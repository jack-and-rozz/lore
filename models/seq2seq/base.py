# coding: utf-8 
import sys
import numpy as np
from pprint import pprint
import tensorflow as tf
from core.models.base import ModelBase
from core.utils.tf_utils import shape, linear, make_summary
from core.utils.common import dbgprint, dotDict, recDotDefaultDict, flatten, flatten_batch, flatten_recdict
#from core.models.seq2seq.evaluation import evaluate, summarize_test_results

class Seq2SeqBase(ModelBase):
  def __init__(self, sess, config, trainer, vocab):
    super(Seq2SeqBase, self).__init__(sess, config, trainer, vocab)

  def get_input_feed(self, batch, is_training):
    raise NotImplementedError

  def encode(self, encoder, source_ph_words, source_ph_chars):
    raise NotImplementedError

  def decode(self, decoder, dec_train_args, dec_test_args):
    raise NotImplementedError

  def add_bos_and_eos(self, target, start_token, end_token):
    with tf.name_scope('add_BOS_and_EOS'):
      # add start_token (end_token) to decoder's input (output).
      batch_size = shape(target, 0)
      with tf.name_scope('start_tokens'):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size])
      with tf.name_scope('end_tokens'):
        end_tokens = tf.tile(tf.constant([end_token], dtype=tf.int32), [batch_size])
      dec_input_tokens = tf.concat([tf.expand_dims(start_tokens, 1), target], axis=1)
      dec_output_tokens = tf.concat([target, tf.expand_dims(end_tokens, 1)], axis=1)
    return dec_input_tokens, dec_output_tokens

  def shift_target_tokens(self, target_tokens, start_token, end_token):
    with tf.name_scope('shift_target_tokens'):
      dec_input_tokens, dec_output_tokens = self.add_bos_and_eos(
        target_tokens, start_token, end_token)

      # Length of target + end_token (or start_token)
      # *NOTE* the end_token must have zero-ID.
      dec_lengths = tf.count_nonzero(
        target_tokens, axis=-1, dtype=tf.int32) + 1
    return dec_input_tokens, dec_output_tokens, dec_lengths

  def get_input_feed(self, batch, is_training):
    input_feed = {}
    batch.response.raw = batch.context.raw
    batch.response.word = batch.context.word

    input_feed[self.is_training] = is_training
    input_feed[self.ph.context.word] = batch.context.word
    if self.ph.context.char is not None:
      input_feed[self.ph.context.char] = batch.context.char
    if batch.response:
      input_feed[self.ph.response] = batch.response.word

    return input_feed

  def tile_batch_for_pred_loss(self, batch, predictions):
    raise NotImplementedError

  def test(self, batches, mode, logger, output_path):
    results = []
    used_batches = []
    sys.stderr.write('Start decoding (%s) ...\n' % mode)
    for i, batch in enumerate(batches):
      input_feed = self.get_input_feed(batch, False)
      # print('<%d>' % i)
      # if i != 654:
      #   continue

      # for k, v in flatten_recdict(batch).items():
      #   if isinstance(v, np.ndarray):
      #     print(k, v.shape)
      #   else:
      #     print(k)
      #   print(v)
      # exit(1)
      output_feed = [
        self.predictions, 
        self.loss_by_example, 
      ]
      # Do decoding and calcurate loss for the gold outputs.
      try:
        predictions, gold_loss = self.sess.run(output_feed, 
                                               input_feed)

      # Calcurate loss for the beam-decoded outputs.
        tiled_batch = self.tile_batch_for_pred_loss(batch, predictions)
        input_feed = self.get_input_feed(tiled_batch, False)
        pred_loss = self.sess.run(self.loss_by_example, input_feed)
      except Exception as e:
        # for DEBUG.
        print(e)
        sys.stdout = sys.stderr
        for k, v in flatten_recdict(tiled_batch).items():
          if type(v) == np.ndarray:
            print(k, v.shape)
          else:
            print(k, type(v))
        exit(1)
      batch_size = predictions.shape[0]
      beam_width = predictions.shape[1]
      pred_loss = np.reshape(pred_loss, [batch_size, beam_width])

      # Flatten the batch and outputs.
      used_batch = flatten_batch(batch)
      for i in range(len(used_batch)):
        used_batch[i].inp_context.sent = [self.vocab.decoder.word.tokens2sent(x) for x in used_batch[i].inp_context.raw]
        used_batch[i].inp_response.sent = self.vocab.decoder.word.tokens2sent(used_batch[i].inp_response.raw)
        #used_batch[i].inp_context.sent = 
      used_batches += used_batch

      results += self.postprocess(predictions, 
                                  gold_loss=gold_loss, 
                                  pred_loss=pred_loss)


    evaluation_results = batches.evaluate(used_batches, self.vocab, results)
    batches.output_tests(used_batches, self.vocab, results, 
                         evaluation_results,
                         output_path=output_path)
    summary_dict = {}
    for metric, score in evaluation_results.items():
      summary_dict['%s/%s/%s' % (self.scopename, mode, metric)] = score
    summary = make_summary(summary_dict)
    return evaluation_results['overall'], summary

  def postprocess(self, predictions, *args, **kwargs):
    raise NotImplementedError

class OneTurnSeq2Seq(Seq2SeqBase):
  def setup_placeholder(self, config):
    '''
    NOTE:
    The tensors fed to placeholders must have no BOS and EOS.
    They are appended to decoder's inputs and outputs through ``self.add_bos_and_eos''.
    '''
    # Placeholders
    with tf.name_scope('Placeholder'):
      ph = recDotDefaultDict()
      # encoder's placeholder
      ph.context.word = tf.placeholder( 
        tf.int32, name='context.word', 
        shape=[None, None]) # [batch_size, n_max_word]
      ph.context.char = tf.placeholder(
        tf.int32, name='context.char',
        shape=[None, None, None])  # [batch_size, n_max_word, n_max_char]

      # decoder's placeholder
      ph.response = tf.placeholder(
        tf.int32, name='response.word', shape=[None, None])
    return ph

  def tile_batch_for_pred_loss(self, batch, predictions):
    '''
    Create a tiled batch to calculate losses for the beam-decoded outputs.
    <Args>
    - batch:
    - predictions: [batch_size, beam_width, max_response_length]
    '''
    batch_size, beam_width, _ = predictions.shape # [batch_size, beam_width, max_len]
    tiled = recDotDefaultDict()
    tiled.context.word = np.reshape(
      np.tile(batch.context.word, [1, beam_width]), 
      [-1] + list(batch.context.word.shape)[1:]) # [batch_size*beam_width, sent_len]
    tiled.context.char = np.reshape(
      np.tile(batch.context.char, [1, beam_width, 1]), 
      [-1] + list(batch.context.char.shape)[1:]) # [batch_size*beam_width, sent_len, word_len]
    tiled.response.word = np.reshape(predictions, [batch_size*beam_width, -1])
    
    # Cut a not fully filled matrix which decoder may generate.
    # (the shape an output matrix is [batch_size, beam_width, max_dec_len], but the maximum length of the generated sequences can be lower than 'max_dec_len'.
    max_dec_len = np.max(np.count_nonzero(tiled.response.word, axis=1))
    tiled.response.word = tiled.response.word[:, :max_dec_len]
    return tiled
 
