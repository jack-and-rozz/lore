#coding: utf-8
from logging import FileHandler
from pprint import pprint
from collections import OrderedDict, defaultdict
import sys, os, random, argparse, re, time
import numpy as np
import tensorflow as tf
sys.path.append(os.getcwd())


from occult.utils.common import dbgprint, dotDict, recDotDict, recDotDefaultDict, flatten, flatten_batch, flatten_recdict, timewatch, str2bool, logManager, get_config, print_config
import occult.dataset as data_libs
import occult.vocabulary as vocab_libs
import occult.trainers as trainer_libs

random.seed(0)
np.random.seed(0)
exit(1)

BEST_CHECKPOINT_NAME = 'model.ckpt.best'
def get_logger(logfile_path=None):
  logger = logManager(handler=FileHandler(logfile_path)) if logfile_path else logManager()
  return logger

class ManagerBase(object):
  @timewatch()
  def __init__(self, args, sess):
    self.sess = sess
    self.model = None
    self.root_path = args.model_root_path
    self.checkpoints_path = args.model_root_path +'/checkpoints'
    self.tests_path = args.model_root_path + '/tests'
    self.summaries_path = args.model_root_path + '/summaries'
    self.create_dir(args)
    self.logger = get_logger(logfile_path=os.path.join(args.model_root_path, 
                                                       args.mode + '.log'))
    self.config = self.load_config(args)
    assert 'vocab' in self.config
    assert 'tasks' in self.config and len(self.config.tasks)

    self.vocab = self.setup_vocab(self.config.vocab)
    self.dataset = self.setup_dataset(self.config, self.vocab)

  def setup_vocab(self, vocab_config):
    # Load pretrained embeddings.
    vocab = dotDict()
    for vocab_type in vocab_config:
      with tf.variable_scope(vocab_type):
        vocab[vocab_type] = dotDict()
        for woken_type, conf in vocab_config[vocab_type].items():
          with tf.variable_scope(token_type):
            vocab_class = getattr(vocab_libs, conf.vocab_class)
            vocab[vocab_type][token_type] = vocab_class(conf)

    return vocab

  def setup_dataset(self, config, vocab):
    # Load Dataset.x
    dataset = recDotDict()
    for k, v in config.tasks.items():
      if 'dataset' in v: # for tasks without data (e.g., adv learning)
        dataset_type = getattr(data_libs, v.dataset.dataset_type)
        dataset[k] = dataset_type(v.dataset, vocab)
    return dataset

  def load_config(self, args):
    config_stored_path = os.path.join(args.model_root_path, 'main.conf')
    if not os.path.exists(config_stored_path) or args.cleanup: #(args.cleanup and args.mode in ['train', 'debug']): 
      config_read_path = args.config_root
      config = get_config(config_read_path)[args.config_type]
      sys.stderr.write("Save the initial config file %s(%s) to %s.\n" % (config_read_path, args.config_type, config_stored_path))
      with open(config_stored_path, 'w') as f:
        sys.stdout = f
        print_config(config)
        sys.stdout = sys.__stdout__
    else: 
      config_read_path = config_stored_path
      config = get_config(config_read_path)
      sys.stderr.write("Found an existing config file, \'%s\'.\n" % (config_stored_path))
    config = recDotDict(config)
    #sys.stderr.write(str(config) + '\n')
    return config

  def create_dir(self, args):
    if not os.path.exists(args.model_root_path):
      os.makedirs(args.model_root_path)
    if not os.path.exists(self.checkpoints_path):
      os.makedirs(self.checkpoints_path)
    if not os.path.exists(self.tests_path):
      os.makedirs(self.tests_path)
    if not os.path.exists(self.summaries_path):
      os.makedirs(self.summaries_path)

  @timewatch()
  def setup_model(self, config, load_best=False):
    if load_best == True:
      checkpoint_path = os.path.join(self.checkpoints_path, BEST_CHECKPOINT_NAME)
    else:
      checkpoint_path = None

    trainer_type = getattr(trainer_libs, config.trainer.trainer_type)

    # Define computation graph.
    if not self.model:
      m = trainer_type(self.sess, config, self.vocab) if not self.model else self.model
      self.model = m
    else:
      m = self.model
      return m

    if not checkpoint_path or not os.path.exists(checkpoint_path + '.index'):
      ckpt = tf.train.get_checkpoint_state(self.checkpoints_path)
      checkpoint_path = ckpt.model_checkpoint_path if ckpt else None

    self.saver = tf.train.Saver(tf.global_variables(), 
                                max_to_keep=config.trainer.max_to_keep)
    self.summary_writer = tf.summary.FileWriter(self.summaries_path, 
                                                self.sess.graph)
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
      sys.stdout.write("Reading model parameters from %s\n" % checkpoint_path)
      self.saver.restore(self.sess, checkpoint_path)
    else:
      sys.stdout.write("Created model with fresh parameters.\n")
      tf.global_variables_initializer().run()

    # Store variable names and vocabulary for debug.
    variables_path = self.root_path + '/variables.list'
    with open(variables_path, 'w') as f:
      variable_names = sorted([v.name + ' ' + str(v.get_shape()) for v in tf.global_variables()])
      variable_names = [name for name in variable_names if not re.search('Adam', name)]
      f.write('\n'.join(variable_names) + '\n')

    self.output_vocab()
    return m

  def output_vocab(self):
    vocab_path = self.root_path + '/vocab.en.word.list'
    if not os.path.exists(vocab_path):
      with open(vocab_path, 'w') as f:
        f.write('\n'.join(self.vocab.encoder.word.rev_vocab) + '\n')

    vocab_path = self.root_path + '/vocab.de.word.list'
    if not os.path.exists(vocab_path) and hasattr(self.config.vocab, 'decoder'):
        with open(vocab_path, 'w') as f:
          f.write('\n'.join(self.vocab.decoder.word.rev_vocab) + '\n')

    vocab_path = self.root_path + '/vocab.en.char.list'
    if not os.path.exists(vocab_path) and \
       hasattr(self.config.vocab.encoder, 'char'):
      with open(vocab_path, 'w') as f:
        f.write('\n'.join(self.vocab.encoder.char.rev_vocab) + '\n')

  def save_model(self, model, save_as_best=False):
    checkpoint_path = self.checkpoints_path + '/model.ckpt'
    self.saver.save(self.sess, checkpoint_path, global_step=model.epoch)
    if save_as_best:
      suffixes = ['data-00000-of-00001', 'index', 'meta']

      # Keep the older best checkpoint to handle failures in saving.
      for sfx in suffixes:
        target_path = self.checkpoints_path + "/%s.%s" % (BEST_CHECKPOINT_NAME, sfx)
        target_path_bak = self.checkpoints_path + "/%s.%s.old" % (BEST_CHECKPOINT_NAME, sfx)
        if os.path.exists(target_path):
          cmd = "mv %s %s" % (target_path, target_path_bak)
          os.system(cmd)

      # Copy the current best checkpoint.
      for sfx in suffixes:
        source_path = self.checkpoints_path + "/model.ckpt-%d.%s" % (model.epoch.eval(), sfx)
        target_path = self.checkpoints_path + "/%s.%s" % (BEST_CHECKPOINT_NAME, sfx)
        if os.path.exists(source_path):
          cmd = "cp %s %s" % (source_path, target_path)
          os.system(cmd)

      # Remove the older one.
      for sfx in suffixes:
        target_path_bak = self.checkpoints_path + "/%s.%s.old" % (BEST_CHECKPOINT_NAME, sfx)
        if os.path.exists(target_path_bak):
          cmd = "rm %s" % (target_path_bak)
          os.system(cmd)

  def train(self):
    model = self.setup_model(self.config)
    if not len(model.tasks):
      raise ValueError('Specify at least 1 task in main.conf.')

    if model.epoch.eval() == 0:
      self.logger.info('Loading dataset...')
      for task_name, d in self.dataset.items():
        train_size, dev_size, test_size = d.size
        size_info = (task_name, str(train_size), str(dev_size), str(test_size))
        self.logger.info('<Dataset size>')
        self.logger.info('%s: %s, %s, %s' % size_info)

    # if isinstance(model, trainers.OneByOne):
    #   self.train_one_by_one(model)
    # else:
    #   self.train_simultaneously(model)
    self.train_simultaneously(model)

    # Do final validation and testing with the best model.
    m = self.test()
    self.logger.info("The model in epoch %d performs best." % m.epoch.eval())

  def train_simultaneously(self, model):
    m = model
    for epoch in range(m.epoch.eval(), self.config.trainer.max_epoch):
      learning_rate = m.learning_rate.eval()

      self.logger.info("Epoch %d: Start: learning_rate: %e" % (epoch, learning_rate))
      batches = self.get_batch('train')
      epoch_time, step_time, train_loss, summary = m.train(batches)

      self.logger.info("Epoch %d (train): epoch-time %.2f, step-time %.2f, loss %s" % (epoch, epoch_time, step_time, " ".join(["%.3f" % l for l in train_loss])))

      batches = self.get_batch('valid')
      epoch_time, step_time, valid_loss, summary = m.valid(batches)

      self.summary_writer.add_summary(summary, m.epoch.eval())
      self.logger.info("Epoch %d (valid): epoch-time %.2f, step-time %.2f, loss %s" % (epoch, epoch_time, step_time, " ".join(["%.3f" % l for l in valid_loss])))

      valid_score = -valid_loss[0] # valid loss of the main task
      save_as_best = False

      task_model = list(m.tasks.values())[0]
      if valid_score >= task_model.max_soccult.eval():
        save_as_best = True
        self.logger.info("Epoch %d (valid): %s max score update (%.3f->%.3f): " % (m.epoch.eval(), task_name, task_model.max_soccult.eval(), valid_score))
        task_model.update_max_score(valid_score)

      t = time.time()
      #save_as_best = self.test_for_valid(m)
      self.test_for_valid(m)
      epoch_time = time.time() - t 
      self.logger.info("Epoch %d (test) : epoch-time %.2f" % (epoch, epoch_time))
      m.add_epoch()
      self.save_model(m, save_as_best=save_as_best)

  # def train_one_by_one(self, model):
  #   '''
  #   '''
  #   m = model
  #   def _run_task(m, task):
  #     epoch = m.epoch.eval()
  #     batches = self.get_batch('train')
  #     self.logger.info("Epoch %d: Start" % m.epoch.eval())
  #     epoch_time, step_time, train_loss, summary = m.train(task, batches)
  #     self.logger.info("Epoch %d (train): epoch-time %.2f, step-time %.2f, loss %.3f" % (epoch, epoch_time, step_time, train_loss))

  #     batches = self.get_batch('valid')
  #     epoch_time, step_time, valid_loss, summary = m.valid(task, batches)
  #     self.summary_writer.add_summary(summary, m.epoch.eval())
  #     self.logger.info("Epoch %d (valid): epoch-time %.2f, step-time %.2f, loss %.3f" % (epoch, epoch_time, step_time, valid_loss))
  #     m.add_epoch()

  #   # Train the model in a reverse order of the important tasks.
  #   task = m.tasks.keys()[1]
  #   for i in range(m.epoch.eval(), int(self.config.trainer.max_epoch/2)):
  #     _run_task(m, task)
  #     save_as_best = self.test_for_valid(model=m, target_tasks=[task])
  #     self.save_model(m, save_as_best=save_as_best)

  #   # Load a model with the best score of WikiP2D task. 
  #   best_ckpt = os.path.join(self.checkpoints_path, BEST_CHECKPOINT_NAME)
  #   m = self.setup_model(self.config, load_best=True)

  #   task = m.tasks.keys()[0]
  #   for epoch in range(m.epoch.eval(), self.config.trainer.max_epoch):
  #     _run_task(m, task)
  #     save_as_best = self.test_for_valid(model=m, target_tasks=[task])
  #     self.save_model(m, save_as_best=save_as_best)

  def test_for_valid(self, model, target_tasks=None):
    '''
    This is a function to show the performance of the model in each epoch.
    If you'd like to run testing again in a different setting from terminal, execute test().
    <Args>
    - model:
    - target_tasks:
    <Return>
    - A boolean, which shows whether the score of the first task in this epoch becomes higher or not.
    '''
    m = model

    tasks = OrderedDict([(k, v) for k, v in m.tasks.items() if not target_tasks or k in target_tasks])
    epoch = m.epoch.eval()
    save_as_best = [False for t in tasks]

    mode = 'valid'
    valid_batches = self.get_batch(mode)
    for i, (task_name, task_model) in enumerate(tasks.items()):
      if task_name not in valid_batches:
        continue
      
      batches = valid_batches[task_name]
      output_path = self.tests_path + '/%s_%s.%03d' % (task_name, mode, epoch)
      valid_score, valid_summary = task_model.test(batches, mode, 
                                                   self.logger, output_path)
      self.summary_writer.add_summary(valid_summary, m.epoch.eval())
    
      # Use evaluation metrics for validation?
      # if valid_score >= task_model.max_soccult.eval():
      #   save_as_best[i] = True
      #   self.logger.info("Epoch %d (valid): %s max score update (%.3f->%.3f): " % (m.epoch.eval(), task_name, task_model.max_soccult.eval(), valid_score))
      #   task_model.update_max_score(valid_score)

    # mode = 'test'
    # test_batches = self.get_batch(mode)
    # for i, (task_name, task_model) in enumerate(tasks.items()):
    #   if not task_name in test_batches:
    #     continue
    #   batches = test_batches[task_name]
    #   output_path = self.tests_path + '/%s_%s.%03d' % (task_name, mode, epoch)
    #   test_score, test_summary = task_model.test(batches, mode, 
    #                                              self.logger, output_path)
    #   self.summary_writer.add_summary(test_summary, m.epoch.eval())

    # Currently select the best epoch by the score on the first task.
    return save_as_best[0] 

  def test(self):
    target_tasks = []
    m = self.setup_model(self.config, load_best=True)
    tasks = OrderedDict([(k, v) for k, v in m.tasks.items() if not target_tasks or k in target_tasks])

    mode = 'valid'
    valid_batches = self.get_batch(mode)
    for i, (task_name, task_model) in enumerate(tasks.items()):
      if task_name not in valid_batches:
        continue
      batches = valid_batches[task_name]
      output_path = self.tests_path + '/%s_%s.best' % (task_name, mode)
      test_score, _ = task_model.test(batches, mode, self.logger, output_path)

    mode = 'test'
    test_batches = self.get_batch(mode)
    for i, (task_name, task_model) in enumerate(tasks.items()):
      if task_name not in test_batches:
        continue
      batches = test_batches[task_name]
      output_path = self.tests_path + '/%s_%s.best' % (task_name, mode)

      test_score, _ = task_model.test(batches, mode, self.logger, output_path)
    return m

  def get_batch(self, mode):
    batches = recDotDict()

    for task_name in self.config.tasks:
      if not task_name in self.dataset:
        continue
      # batch_size = self.config.tasks[task_name].batch_size
      # if mode != 'train':
      #   batch_size /= 5 # Calculating losses for all the predictions expands batch size by beam width, which can cause OOM. (TODO: automatically adjust the batch_size in testing)
      # data = getattr(self.dataset[task_name], mode)
      # #if data.max_rows >= 0:

      # batches[task_name] = data.get_batch(batch_size, 
      #                                     do_shuffle=do_shuffle) 

      data = getattr(self.dataset[task_name], mode)
      #batches[task_name] = data.get_batch()
      batches[task_name] = data

    return batches


def main(args):
  tf_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(allow_growth=True)
  )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    random.seed(0)
    tf.set_random_seed(0)
    manager = ExperimentManager(args, sess)
    if args.mode == 'demo':
      while True:
        utterance = input('> ')
        manager.demo([utterance])
    else:
      getattr(manager, args.mode)()


def get_parser():
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('model_root_path', type=str, help ='')
  parser.add_argument('mode', type=str, help ='')
  parser.add_argument('-ct','--config_type', default='tmp', 
                      type=str, help ='')
  parser.add_argument('-cl', '--cleanup', default=False,
                      action='store_true', help ='')
  parser.add_argument('--debug', default=False,
                      action='store_true', help ='')
  return parser

if __name__ == "__main__":
  # Common arguments are defined in base.py
  parser = get_parser()
  parser.add_argument('-cr','--config_root', 
                      default='occult/configs/main.conf', 
                      help='')

  args = parser.parse_args()
  main(args)
