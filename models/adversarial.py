#coding: utf-8
import math, time, sys
import tensorflow as tf
from tensorflow.python.framework import ops
from occult.models.base import ModelBase, TestModelBase
from occult.trainers.base import average_gradients
from occult.utils.tf_utils import linear, shape, assign_device
from occult.utils.common import dbgprint

class FlipGradientBuilder(object):
  def __init__(self):
    self.num_calls = 0

  def __call__(self, x, l=1.0):
    grad_name = "FlipGradient%d" % self.num_calls
    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
      #return [tf.neg(grad) * l]
      return [tf.negative(grad) * l]
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": grad_name}):
      y = tf.identity(x)
    self.num_calls += 1
    return y

flip_gradient = FlipGradientBuilder()

class AdversarialBase(ModelBase):
  def __init__(self, sess, config, trainer, vocab, other_models):
    super(AdversarialBase, self).__init__(sess, config, trainer, vocab)
    self.loss, self.gradients = self.define_combination(other_models)

  def define_combination(self, all_models):
    '''
    Define adversarial layers. 
    *Note* this function must be executed after all other models were defined. 
    '''
    adv_models, input_repls, output_label_ids = self.set_label_by_model(all_models) 
    n_labels = max(output_label_ids) + 1
    gradients = []
    # dbgprint(adv_models)
    # dbgprint(input_repls)
    # dbgprint(output_label_ids)

    loss_by_model = []
    gradients_by_model = []
    for model, input_repl, output_id in zip(adv_models, input_repls, output_label_ids):
      # To ensure the adversarial learning using outputs from a task is assigned to the same GPU.
      task_idx = all_models.index(model)
      # device = assign_device(task_idx) 
      # sys.stderr.write('Defining adversarial layer in %s...\n' % (device))
      # with tf.device(device):
      with tf.name_scope('adversarial'):
        hidden = flip_gradient(input_repl)
        for depth in range(self.config.ffnn_depth - 1):
          with tf.variable_scope('hidden%d' % (depth+1)) as scope:
            hidden = linear(hidden, shape(hidden, -1), scope=scope)
            hidden = tf.nn.dropout(hidden, self.keep_prob)

        with tf.variable_scope('output') as scope:
          logits = linear(hidden, n_labels, activation=None, scope=scope)
        #logits = tf.reshape(logits, [-1, n_labels])
        tiled_output_label_id = tf.tile([output_id], [shape(logits, 0)])
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tiled_output_label_id, logits=logits))
        gradients = self.compute_gradients(self.loss_weight * loss)
        loss_by_model.append(loss)
        gradients_by_model.append(gradients)

    loss = tf.reduce_mean(loss_by_model)
    gradients = average_gradients(gradients_by_model)
    return loss, gradients

  def set_label_by_model(self, all_models):
    '''
    '''
    adv_models = []
    input_repls = []
    output_label_ids = []
    label_vocab = {}
    for m in all_models:
      if not hasattr(m, self.config.attr_name.input) or \
         not hasattr(m, self.config.attr_name.output) or \
         isinstance(m, TestModelBase):
        continue
      input_repl = getattr(m, self.config.attr_name.input)
      output_label = getattr(m, self.config.attr_name.output)
      if output_label not in label_vocab:
        label_vocab[output_label] = len(label_vocab)
      output_label_id = label_vocab[output_label]
      adv_models.append(m)
      input_repls.append(input_repl)
      output_label_ids.append(output_label_id)

    if len(adv_models) < 2:
      raise ValueError('At least two labels (tasks, or languages) must be defined to adversarially distinguish them.')

    return adv_models, input_repls, output_label_ids

class TaskAdversarial(AdversarialBase):
  pass

class LangAdversarial(AdversarialBase):
  pass

available_models = {
  TaskAdversarial.__name__: TaskAdversarial,
  LangAdversarial.__name__: LangAdversarial,
}
