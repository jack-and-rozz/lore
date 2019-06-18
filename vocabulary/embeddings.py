# coding: utf-8
import tensorflow as tf
import math


def initialize_embeddings(name, emb_shape, initializer=None, 
                          trainable=True):
  if not initializer:
    initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
  embeddings = tf.get_variable(name, emb_shape, trainable=trainable,
                               initializer=initializer)
  return embeddings

def initialize_word_embeddings(init_embeddings,
                               n_start_tokens,
                               use_pretrained_emb=False,
                               trainable=True):
  use_pretrained_emb = init_embeddings is not None
  with tf.name_scope('word_emb'):
    # Special tokens such as [PAD, UNK, ...] should be trainable.
    initializer = tf.constant_initializer(init_embeddings[:n_start_tokens, :]) if use_pretrained_emb else None
    start_vocabs = initialize_embeddings(
      'start_vocab_emb', init_embeddings[:n_start_tokens, :].shape, 
      initializer=initializer, 
      trainable=True)

    initializer = tf.constant_initializer(init_embeddings[n_start_tokens:, :]) if use_pretrained_emb else None

    word_emb = initialize_embeddings(
      'word_emb', init_embeddings[n_start_tokens:, :].shape, 
      initializer=initializer, 
      trainable=trainable)
    embeddings = tf.concat([start_vocabs, word_emb], axis=0)
  return embeddings

def initialize_char_embeddings(vocab_size, emb_size):
  c_emb_shape = [vocab_size, emb_size] 
  with tf.name_scope('char_emb'):
    embeddings = initialize_embeddings('char_emb', c_emb_shape, trainable=True)
  return embeddings
