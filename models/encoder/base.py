# coding: utf-8 
import sys
import tensorflow as tf
from functools import reduce
import numpy as np

from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import LSTMStateTuple

from occult.utils.common import dbgprint, dotDict
from occult.utils.tf_utils import shape, cnn, linear, projection, batch_gather, batch_loop 
from occult.models.base import setup_cell, initialize_embeddings
from occult.vocabulary.base import VocabularyWithEmbedding

def project_state(state, output_size, activation=None):
  with tf.name_scope(sys._getframe().f_code.co_name):
    state_project = tf.layers.Dense(units=output_size, 
                                    dtype=tf.float32,
                                    name="state_projection",
                                    activation=activation)
    if isinstance(state, LSTMStateTuple):
      raise NotImplementedError
    else:
      return state_project(state)


def separate_state(state, n_split):
  '''
  <Args>
  - states: A tuple of tensors or LSTMStateTuple.
  '''
  with tf.name_scope(sys._getframe().f_code.co_name):

    if isinstance(state, LSTMStateTuple):
      raise NotImplementedError
    else:
      return tuple(tf.split(state, n_split, axis=-1))

def merge_state(state, merge_func=tf.concat):
  '''
  <Args>
  - state: A list of state.
  '''

  if merge_func == tf.reduce_mean:
    axis = 0 
  elif merge_func == tf.concat:
    axis = -1
  else:
    raise ValueError('merge_func must be tf.reduce_mean or tf.concat')

  with tf.name_scope(sys._getframe().f_code.co_name):
    if isinstance(state[0], LSTMStateTuple):
      new_c = merge_func([s.c for s in state], axis=axis)
      new_h = merge_func([s.h for s in state], axis=axis)
      state = LSTMStateTuple(c=new_c, h=new_h)
    else:
      state = merge_func(state, axis)
    return state

def reshape_state(state, other_shapes):
  with tf.name_scope(sys._getframe().f_code.co_name):
    if isinstance(state, LSTMStateTuple):
      new_c = tf.reshape(state.c, other_shapes + [shape(state.c, -1)])
      new_h = tf.reshape(state.h, other_shapes + [shape(state.h, -1)])
      state = LSTMStateTuple(c=new_c, h=new_h)
    else:
      state = tf.reshape(state, other_shapes + [shape(state, -1)])
    return state



def setup_rnn(inputs, sequence_length, cell_type, rnn_size, 
              use_residual, keep_prob, num_layers):
  with tf.name_scope(sys._getframe().f_code.co_name):

    cells = tf.contrib.rnn.MultiRNNCell([
      setup_cell(
        cell_type,
        rnn_size, 
        use_residual,
        keep_prob,
      ) for _ in range(num_layers)]) 
    outputs, state = rnn.dynamic_rnn(
      cells, inputs, 
      sequence_length=sequence_length, dtype=tf.float32)
    return outputs, state

def setup_birnn(inputs, sequence_length, cell_type, rnn_size, 
                use_residual, keep_prob):
  with tf.name_scope(sys._getframe().f_code.co_name):
    batch_size = shape(inputs, 0)
    # For 'initial_state' of CustomLSTMCell, different scopes are required in these initializations.
    with tf.variable_scope('fw_cell'):
      cell_fw = setup_cell(cell_type, rnn_size,
                           use_residual,
                           keep_prob=keep_prob)
      initial_state_fw = cell_fw.initial_state(batch_size) if hasattr(cell_fw, 'initial_state') else None

    with tf.variable_scope('bw_cell'):
      cell_bw = setup_cell(cell_type, rnn_size,
                           use_residual,
                           keep_prob=keep_prob)
      initial_state_bw = cell_bw.initial_state(batch_size) if hasattr(cell_bw, 'initial_state') else None

    outputs, state = rnn.bidirectional_dynamic_rnn(
      cell_fw, cell_bw, inputs,
      initial_state_fw=initial_state_fw,
      initial_state_bw=initial_state_bw,
      sequence_length=sequence_length, dtype=tf.float32)
    return outputs, state

def extend_vocab_for_oov(embeddings, inputs, unk_id):
  '''
  Copy the embeddings of OOV to expand word embedding matrix by the number of unique OOV words (mainly for CopyNet)
  <Args>
  - embeddings: A Tensor ([vocab_size, emb_size]).
  - inputs: 
  - unk_id: An integer.
  '''
  with tf.name_scope(sys._getframe().f_code.co_name):
    unk_emb = tf.expand_dims(embeddings[unk_id, :], 0) # [1, emb_size]
    num_oov_words = tf.maximum(tf.reduce_max(inputs) - shape(embeddings, 0), 0)

    oov_embeddings = tf.tile(unk_emb, [num_oov_words, 1]) # [num_oov_words, emb_size]
    extended_embeddings = tf.concat([embeddings, oov_embeddings], axis=0)
  return extended_embeddings

class WordEncoder(object):
  def __init__(self, config, keep_prob, enc_vocab, 
               embeddings=None, scope=None):
    self.cbase = config.cbase
    self.keep_prob = keep_prob
    self.vocab = enc_vocab
    self.scope = scope # to reuse variables

    sys.stderr.write("Initialize word embeddings by the pretrained ones.\n")

    self.embeddings = dotDict()
    self.embeddings.word = enc_vocab.word.embeddings
    if config.cbase:
      self.embeddings.char = enc_vocab.char.embeddings

  def word_encode(self, inputs, extend_vocab=False):
    if inputs is None:
      return inputs

    with tf.variable_scope(self.scope or "WordEncoder"):
      # dbgprint(self.embeddings.word, inputs, 
      #          self.vocab.word.UNK_ID)
      if extend_vocab:
        # Extend vocabulary size by the number of OOV tokens in the inputs.
        word_embeddings = extend_vocab_for_oov(self.embeddings.word, inputs, 
                                               self.vocab.word.UNK_ID)
      else:
        word_embeddings = self.embeddings.word
      outputs = tf.nn.embedding_lookup(word_embeddings, inputs)
      outputs = tf.nn.dropout(outputs, self.keep_prob)
    return outputs

  def char_encode(self, inputs):
    '''
    Args:
    - inputs: [*, max_sequence_length, max_word_length]
    Return:
    - outputs: [*, max_sequence_length, cnn_output_size]
    '''
    if inputs is None:
      return inputs

    with tf.variable_scope(self.scope or "WordEncoder"):
      # Flatten the input tensor to each word (rank-3 tensor).
      with tf.name_scope('flatten'):
        char_repls = tf.nn.embedding_lookup(self.embeddings.char, inputs) # [*, max_word_len, char_emb_size]
        other_shapes = [shape(char_repls, i) for i in range(len(char_repls.get_shape()[:-2]))]

        flattened_batch_size = reduce(lambda x,y: x*y, other_shapes)
        max_sequence_len = shape(char_repls, -2)
        char_emb_size = shape(char_repls, -1)

        flattened_char_repls = tf.reshape(
          char_repls, 
          [flattened_batch_size, max_sequence_len, char_emb_size])

      cnn_outputs = cnn(flattened_char_repls) # [flattened_batch_size, cnn_output_size]
      outputs = tf.reshape(cnn_outputs, other_shapes + [shape(cnn_outputs, -1)]) # [*, cnn_output_size]
      outputs = tf.nn.dropout(outputs, self.keep_prob)
    return outputs

  def get_input_feed(self, batch):
    input_feed = {}
    return input_feed

class SentenceEncoder(object):
  def __init__(self, config, keep_prob, activation=tf.nn.relu, 
               scope=None):
    self.keep_prob = keep_prob
    self.activation = activation
    self.scope = scope
    self.config = config

  def encode(self, inputs, sequence_length): # , merge_func=tf.reduce_mean):
    config = self.config
    with tf.variable_scope(self.scope or "RNNEncoder") as scope:
      if isinstance(inputs, list):
        inputs = [x for x in inputs if x is not None]
        sent_repls = tf.concat(inputs, axis=-1) # [*, max_sequence_len, hidden_size]
      else:
        sent_repls = inputs

      # Flatten the input tensor to a rank 3 tensor ([*, max_sequence_len, hidden_size]), to handle inputs with more than 3 rank. (e.g. context as list of utterances)
      input_hidden_size = shape(sent_repls, -1)
      max_sequence_len = shape(sent_repls, -2)
      other_shapes = [shape(sent_repls, i) for i in range(len(sent_repls.get_shape()[:-2]))]
      flattened_batch_size = reduce(lambda x,y: x*y, other_shapes)

      flattened_shape = [flattened_batch_size, 
                         max_sequence_len, 
                         input_hidden_size]
      flattened_sent_repls = tf.reshape(sent_repls, flattened_shape) 

      flattened_sequence_length = tf.reshape(sequence_length, 
                                             [flattened_batch_size])

      inputs = flattened_sent_repls

      # Project input before the main RNN, to keep the dims of inputs equal to rnn_size in both cases of using birnn or not.
      input_project = tf.layers.Dense(units=self.config.rnn_size, 
                                      dtype=tf.float32,
                                      name="input_projection",
                                      activation=self.activation)
      inputs = input_project(inputs)
      inputs = tf.nn.dropout(inputs, self.keep_prob)

      birnn_state = []
      if self.config.num_layers.birnn > 0:
        for i in range(self.config.num_layers.birnn):
          with tf.variable_scope('BiRNN/L%d' % i):
            use_residual = self.config.use_residual if i > 0 else False
            outputs, state = setup_birnn(inputs, flattened_sequence_length, 
                                         config.cell_type, config.rnn_size, 
                                         config.use_residual, self.keep_prob)
            # Concat and project the outputs and the state from BiRNN to rnn_size.
            state = merge_state(state, tf.concat)
            state = project_state(state, self.config.rnn_size)
            birnn_state.append(state)

            outputs = tf.concat(outputs, axis=-1)
            output_project = tf.layers.Dense(units=self.config.rnn_size, 
                                             dtype=tf.float32,
                                             name="output_projection",
                                             activation=self.activation)
            outputs = output_project(outputs)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            inputs = outputs

      rnn_state = []
      if self.config.num_layers.rnn > 0:
        with tf.variable_scope('RNN'):
          #cells = self.setup_encoder_cell(self.config, self.keep_prob)
          # outputs, state = rnn.dynamic_rnn(
          #   cells, inputs, 
          #   sequence_length=flattened_sequence_length, dtype=tf.float32)
          outputs, rnn_state = setup_rnn(inputs, flattened_sequence_length, 
                                         config.cell_type, config.rnn_size, 
                                         config.use_residual, self.keep_prob, 
                                         config.num_layers.rnn)

      # Turn the shape of outputs and state back.
      outputs = tf.reshape(outputs, other_shapes + [max_sequence_len, shape(outputs, -1)])

      state = list(birnn_state) + list(rnn_state)
      state = tuple([reshape_state(s, other_shapes) for s in state])
    return outputs, state


class MultiEncoderWrapper(SentenceEncoder):
  def __init__(self, encoders):
    """
    Args 
      encoders: A list of SentenceEncoders. The first encoder is regarded as the shared encoder.
    """
    self.encoders = encoders
    self.keep_prob = encoders[0].keep_prob
    self.scope = encoders[0].scope

  def encode(self, wc_sentences, sequence_length, merge_func=tf.concat):
    if not nest.is_sequence(self.encoders):
      return self.encoders.encode(wc_sentences, sequence_length)
    outputs = []
    state = []
    for e in self.encoders:
      word_repls, o, s = e.encode(wc_sentences, sequence_length)
      outputs.append(o)
      state.append(s)
    self.output_shapes = [o.get_shape() for o in outputs]
    self.state_shapes = [s.get_shape() for s in state]
    outputs = merge_func(outputs, axis=-1)
    state = merge_state(state, merge_func=merge_func)
    return word_repls, outputs, state

