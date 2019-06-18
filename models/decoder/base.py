# coding: utf-8 
import math, time, sys
import numpy as np
from pprint import pprint
import tensorflow as tf
from tensorflow.contrib.seq2seq import AttentionWrapper, TrainingHelper, BasicDecoder, dynamic_decode, tile_batch, BeamSearchDecoder
from tensorflow.contrib.rnn import LSTMStateTuple, MultiRNNCell
from tensorflow.python.util import nest

from occult.utils.tf_utils import shape, linear, make_summary, SharedKernelDense
from occult.utils.common import dbgprint, recDotDefaultDict, flatten, flatten_recdict, flatten_batch, dotDict
from occult.models.base import setup_cell, initialize_embeddings
from occult.vocabulary.base import PAD_ID


def get_state_shape(state):
  '''
  Return the size of an encoder-state. 
  If 'state' is a list of states, return that of the first one.
  '''

  def _get_lstm_state_size(state):
    return [shape(state.h, i) for i in range(len(state.h.get_shape()))]

  if nest.is_sequence(state):
    if isinstance(state[0], LSTMStateTuple):
      return _get_lstm_state_size(state[0])
    else:
      return [shape(state[0], i) for i in range(len(state[0].get_shape()))]
  else:
    if isinstance(state, LSTMStateTuple):
      return _get_lstm_state_size(state)
    else:
      return [shape(state, i) for i in range(len(state.get_shape()))]

def _setup_decoder_cell(config, keep_prob):
  return [setup_cell(
      config.cell_type,
      config.rnn_size, 
      config.use_residual,
      keep_prob
    ) for _ in range(config.num_layers)]


class DecoderBase(object):
  def __init__(self, config, keep_prob, dec_vocab, 
               activation=tf.nn.relu,
               scope=None):
    self.keep_prob = keep_prob
    self.scope = scope
    self.activation = activation
    self.config = config
    self.start_token = dec_vocab.word.token2id(dec_vocab.word.BOS)
    self.end_token = dec_vocab.word.token2id(dec_vocab.word.PAD)
    self.beam_width = config.beam_width

    self.embeddings = dotDict()
    self.embeddings = dec_vocab.word.embeddings


  def setup_decoder_cell(self, config, keep_prob, use_beam_search, init_state, 
                         *args, **kwargs):
    raise NotImplementedError

  def decode_train(self, dec_input_tokens, dec_lengths, init_state, 
                   *attention_args,
                   decoder_class=BasicDecoder, 
                   decoder_kwoptions={}):
    '''
    <Args>
    - dec_input_tokens:
    - dec_length:
    - init_state:
    - decoder_class:
    - decoder_options:
    '''
    with tf.variable_scope(self.scope or "Decoder") as scope:
      train_cell, init_state = self.setup_decoder_cell(
        self.config, self.keep_prob, False, init_state, *attention_args)

      self.input_project = tf.layers.Dense(units=self.config.rnn_size, 
                                           name="input_projection",
                                           activation=self.activation)

      if hasattr(self.config, 'use_emb_as_out_proj') and \
         self.config.use_emb_as_out_proj == True:
        # Make the dim of decoder's output be rnn_size to emb_size.
        emb_project = tf.layers.Dense(units=self.config.rnn_size, 
                                      use_bias=False,
                                      activation=None,
                                      name='emb_projection')
        output_kernel = emb_project(self.embeddings)
        output_kernel = tf.transpose(output_kernel)

        self.output_project = SharedKernelDense(units=shape(self.embeddings, 0),
                                                shared_kernel=output_kernel,
                                                use_bias=False,
                                                activation=None,
                                                name='output_projection')
      else:
        self.output_project = tf.layers.Dense(units=shape(self.embeddings, 0), 
                                              name='output_projection',
                                              use_bias=False,
                                              activation=None)
      #use_bias=False, trainable=False)
      # self.output_project = tf.layers.Dense(units=shape(self.embeddings, 0), 
      #                                       name='output_projection')

      with tf.name_scope('Train'):
        inputs = tf.nn.embedding_lookup(self.embeddings, 
                                        dec_input_tokens)
        inputs = self.input_project(inputs)
        inputs = tf.nn.dropout(inputs, self.keep_prob)

        helper = TrainingHelper(inputs, 
                                sequence_length=dec_lengths, 
                                time_major=False)
        train_decoder = decoder_class(train_cell, helper, init_state,
                                      output_layer=self.output_project,
                                      **decoder_kwoptions)
        
        max_dec_len = tf.reduce_max(dec_lengths, name="max_dec_len")
        outputs, final_state, _ = dynamic_decode(
          train_decoder, impute_finished=True,
          maximum_iterations=max_dec_len, scope=scope)
        logits = outputs.rnn_output

        # To prevent the training loss to be NaN.
        logits += 1e-9
        logits = tf.clip_by_value(logits, -20.0, 20.0, name='clip_logits')
        self.train_decoder = train_decoder

    return logits, final_state

  def decode_test(self, init_state, *attention_args, 
                  decoder_class=BeamSearchDecoder,
                  decoder_kwoptions={}):
    with tf.variable_scope(self.scope or "Decoder") as scope:
      with tf.name_scope('Test'):
        batch_size = get_state_shape(init_state)[0]
        test_cell, tiled_init_state = self.setup_decoder_cell(
          self.config, self.keep_prob, True, init_state, *attention_args)

        def lookup_and_project(inputs):
          with tf.name_scope('lookup_and_project'):
            return self.input_project(
              tf.nn.embedding_lookup(self.embeddings, inputs))
        start_tokens = tf.tile(tf.constant([self.start_token], dtype=tf.int32), 
                               [batch_size])
        test_decoder = decoder_class(
          test_cell, lookup_and_project, start_tokens, self.end_token, 
          tiled_init_state,
          self.beam_width, 
          output_layer=self.output_project,
          length_penalty_weight=self.config.length_penalty_weight,
          **decoder_kwoptions
        )

        outputs, final_state, dec_lengths = dynamic_decode(
          test_decoder, impute_finished=False,
          maximum_iterations=self.config.maxlen, scope=scope)
        predictions = outputs.predicted_ids # [batch_size, T, beam_width]
        predictions = tf.transpose(predictions, perm = [0, 2, 1]) # [batch_size, beam_width, T]
        self.test_decoder = test_decoder

    return predictions, final_state


class RNNDecoder(DecoderBase):
  def setup_decoder_cell(self, config, keep_prob, use_beam_search, init_state):
    cells = MultiRNNCell(_setup_decoder_cell(config, keep_prob))
    if use_beam_search:
      init_state = tile_batch(
          init_state, multiplier=self.beam_width)
    return cells, init_state

class AttentionDecoder(DecoderBase):
  # def decode_train(self, dec_input_tokens, dec_lengths, init_state, 
  #                  attention_states, attention_lengths):
  #   with tf.variable_scope(self.scope or "Decoder") as scope:
  #     if self.config.use_byway_attention:
  #       attention_states, attention_lengths = self.add_byway_attn_state(
  #         attention_states, attention_lengths)
  #   return super(AttentionDecoder, self).decode_train(
  #     dec_input_tokens, dec_lengths, init_state, 
  #     attention_states, attention_lengths)

  # def decode_test(self, init_state, attention_states, attention_lengths):
  #   with tf.variable_scope(self.scope or "Decoder") as scope:
  #     if self.config.use_byway_attention:
  #       attention_states, attention_lengths = self.add_byway_attn_state(
  #         attention_states,
  #         attention_lengths)
  #   return super(AttentionDecoder, self).decode_test(init_state, 
  #                                                    attention_states, 
  #                                                    attention_lengths)

  def add_byway_attn_state(self, attention_states, attention_lengths):
    with tf.name_scope('add_byway_attn_state'):
      batch_size = shape(attention_states, 0)
      state_size = shape(attention_states, -1)
      byway_state = tf.get_variable('byway_state', [state_size]) # [state_size]
      byway_state = tf.expand_dims(byway_state, 0) # [1, state_size]
      byway_state = tf.expand_dims(byway_state, 0) # [1, 1, state_size]
      byway_state = tf.tile(byway_state, [batch_size, 1, 1]) # [1, 1, state_size]
      attention_states = tf.concat([byway_state, attention_states], axis=1)
      attention_lengths += 1
      return attention_states, attention_lengths

  def setup_decoder_cell(self, config, keep_prob, use_beam_search, init_state, 
                         attention_states, attention_lengths):
    batch_size = get_state_shape(init_state)[0]
    if use_beam_search:
      attention_states = tile_batch(attention_states, multiplier=self.beam_width)
      init_state = nest.map_structure(lambda s: tile_batch(s, self.beam_width), init_state)
      attention_lengths = tile_batch(attention_lengths, multiplier=self.beam_width)
      batch_size = batch_size * self.beam_width

    attention_size = shape(attention_states, -1)
    attention = getattr(tf.contrib.seq2seq, config.attention_type)(
      attention_size, attention_states,
      memory_sequence_length=attention_lengths)

    def cell_input_fn(inputs, attention):  
      # define cell input function to keep input/output dimension same
      if not config.use_attention_input_feeding:
        return inputs
      attn_project = tf.layers.Dense(config.rnn_size, dtype=tf.float32, 
                                     name='attn_input_feeding',
                                     activation=self.activation)
      return attn_project(tf.concat([inputs, attention], axis=-1))

    cells = _setup_decoder_cell(config, keep_prob)
    if config.top_attention:  # apply attention mechanism only on the top decoder layer
      cells[-1] = AttentionWrapper(cells[-1], attention_mechanism=attention, 
                                   name="AttentionWrapper",
                                   attention_layer_size=config.rnn_size, 
                                   alignment_history=use_beam_search,
                                   initial_cell_state=init_state[-1],
                                   cell_input_fn=cell_input_fn)
      init_state = [state for state in init_state]
      init_state[-1] = cells[-1].zero_state(batch_size=batch_size, 
                                            dtype=tf.float32)
      init_state = tuple(init_state)
      cells = MultiRNNCell(cells)
    else:
      cells = MultiRNNCell(cells)
      cells = AttentionWrapper(cells, attention_mechanism=attention, 
                               name="AttentionWrapper",
                               attention_layer_size=config.rnn_size, 
                               alignment_history=use_beam_search,
                               initial_cell_state=init_state,
                               cell_input_fn=cell_input_fn)
      init_state = cells.zero_state(batch_size=batch_size, dtype=tf.float32) \
                        .clone(cell_state=init_state)
    return cells, init_state



