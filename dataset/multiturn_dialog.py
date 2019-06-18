# coding: utf-8
import os, re, sys, random, copy, time
import numpy as np
from nltk import word_tokenize

from core.dataset.base import padding as _padding, PartitionedDatasetBase, DatasetBase
from core.utils.common import dotDict, recDotDefaultDict, recDotDict, flatten, flatten_recdict, batching_dicts, dbgprint, read_jsonlines, timewatch, separate_path_and_filename
from core.utils import evaluation as eval_util
from core.vocabulary.base import WordVocabularyFromList

EOT = '__eot__'
EOU = '__eou__'
preprocess = lambda dialog: [uttr.strip().split()[:-1] for uttr in dialog.split(' %s ' % EOT)] 

@timewatch()
def load_dialogs(path, max_rows=None, preprocess=lambda x: x.strip()):
  dialogs = []
  for i, l in enumerate(open(path)):
    if max_rows and i >= max_rows:
      break
    dialog = preprocess(l)
    if len(dialog) > 1:
      dialogs.append(dialog)
  return dialogs

@timewatch()
def load_exemplars(path, max_rows=None,
                   preprocess=lambda x: x.strip()):
  exemplars = []
  for i, l in enumerate(open(path)):
    if max_rows and i >= max_rows:
      break
    exemplars.append(preprocess(l))
    #exemplars.append([preprocess(dialog) for dialog in l.strip().split('\t')[:num_exemplars]])
  return exemplars

@timewatch()
def load_exemplar_ids(path, num_exemplars, max_rows=None):
  if max_rows:
    exemplar_ids = []
    for i, l in enumerate(open(path)):
      if i >= max_rows:
        break
      exemplar_ids.append(list(map(int, l.split()))[:num_exemplars])
    return np.array(exemplar_ids) - 1
    #return np.loadtxt(path, dtype=np.int32)[:max_rows, :num_exemplars] - 1
  else:
    return np.loadtxt(path, dtype=np.int32)[:, :num_exemplars] - 1 
  
class _MultiTurnDialogDatasetBase(PartitionedDatasetBase):
  @timewatch()
  def load_data(self, source_path, max_rows):
    sys.stderr.write("Loading dataset from \'%s\'... \n" % source_path)
    data = load_dialogs(source_path, max_rows=max_rows,
                        preprocess=preprocess)
    self.data = self.create_examples(data)
    sys.stderr.write("Finish loading.\n")

  @timewatch()
  def create_examples(self, dialogs):
    examples = [self.create_example(dialog[:-1], dialog[-1], self.vocab, self.maxlen.sent) for dialog in dialogs]
    return examples

  def create_example(self, _context, _response, vocab, max_uttrs):
    '''
    <Args>
    - context: A hierarchical list of turns. Each turn is a list of utterances spoken by the same speaker. In this function, turns are flattened to a list of utterances with speaker labels indicating whether the corresponding utterance was spoken by the addressee or not. 
  e.g. [[utt1-1, utt1-2], [utt2-1], [utt3-1]] 
    -> ([utt1-1, utt1-2, utt2-1, utt3-1], [0,0,1,0])
    - response: A string.
    - max_uttrs: 
    '''
    assert max_uttrs >= 1 
 
    context = _context[-max_uttrs:]
    response = _response

    inp_c_word_ids = [vocab.encoder.word.tokens2ids(uttr) for uttr in context]
    inp_r_word_ids = vocab.decoder.word.tokens2ids(response) if response else []
    return context, inp_c_word_ids, response, inp_r_word_ids

  def create_batch(self, data):
    '''
    Extract examples and pad them, and then create a batch.
    '''
    # batch = recDotDefaultDict()
    # for d in data:
    #   batch = batching_dicts(batch, d) # list of dictionaries to dictionary of lists.
    batch = list(zip(*data))
    batch = self.padding(batch, minlen=self.minlen, maxlen=self.maxlen)
    return batch

  def padding(self, _batch, minlen=None, maxlen=None):
    batch = dotDict()
    batch.inp_context = dotDict()
    batch.inp_response = dotDict()

    inp_context, inp_context_word_ids, inp_response, inp_response_word_ids = _batch

    batch.inp_context.raw = inp_context
    batch.inp_response.raw = inp_response

    # [batch_size, n_max_sent, n_max_word]
    batch.inp_context.word = _padding(inp_context_word_ids,
                                      minlen=[1, minlen.word],
                                      maxlen=[maxlen.sent, maxlen.word])
    # [batch_size, n_max_word]
    if inp_response_word_ids:
      batch.inp_response.word = _padding(inp_response_word_ids,
                                         minlen=[minlen.word],
                                         maxlen=[maxlen.word])
    return batch


  @classmethod
  def evaluate(self_cls, flat_batches, vocab, results) -> dict:
    evaluation_results = {}
    
    # Hypotheses and references given as sentences must be tokenized for evaluation.
    references = [' '.join(word_tokenize(b.inp_response.sent)) for b in flat_batches]
    hypotheses = [' '.join(word_tokenize(r.candidates[0])) for r in results]
    
    evaluation_results['bleu'] = eval_util.calc_bleu(hypotheses, references)
    evaluation_results['dist-1'] = eval_util.calc_dist(hypotheses, 1)
    evaluation_results['dist-2'] = eval_util.calc_dist(hypotheses, 2)
    evaluation_results['length'] = eval_util.calc_length(hypotheses)
    #evaluation_results['cos_sim'] = eval_util.calc_cos_sim(hypotheses, references)
    
    #evaluation_results['overall'] = 1.0 * sum([r.gold_loss for r in results]) / len(results)
    evaluation_results['overall'] = evaluation_results['bleu']
    
    return evaluation_results

  @classmethod
  def print_example(self_cls, example, vocab, test_result=None):
    '''
    Args:
    - example: An example in a batch obtained from 'flatten_batch(batch)'.
    '''
    # if not decorate_func:
    #   decorate_func = lambda text, *args, **kwargs: text
    decorate_func = self_cls.decorate_output
    if example.title:
      print('- Title :', example.title.sent)

    # Print contexts.
    contexts = example.inp_context.sent if example.inp_context.sent else [vocab.encoder.word.tokens2sent(tokens) for tokens in example.inp_context.raw]
    for i, context in enumerate(contexts):
      context = decorate_func(context, vocab.encoder,
                              word_ids=example.inp_context.word[i])
      print ('- Context %d:' % (i), context)



    # Print gold target sentences.
    if example.inp_response:
      response = example.inp_response.sent if example.inp_response.sent else vocab.decoder.word.tokens2sent(example.inp_response.raw) 

      response = decorate_func(response, vocab.decoder,
                               word_ids=example.inp_response.word)
      if test_result and test_result.gold_loss is not None:
        print ('- Reference (loss=%.3f):' % test_result.gold_loss, response)
      else:
        print ('- Reference:', response)

    print()

    # Print exemplars.
    if example.ex_context and example.ex_response:
      ex_contexts = [vocab.encoder.word.tokens2sent(tokens) for tokens in example.ex_context.raw]
      ex_responses = [vocab.decoder.word.tokens2sent(tokens) for tokens in example.ex_response.raw]
      n_exemplars = len(ex_contexts)
      assert len(ex_contexts) == len(ex_responses)
      for i in range(n_exemplars):
        c_exemplar = decorate_func(ex_contexts[i], vocab.encoder,
                                   word_ids=example.ex_context.word[i])
        r_exemplar = decorate_func(ex_responses[i], vocab.decoder,
                                   word_ids=example.ex_response.word[i])
        print ('- Exemplar C-%d: %s' % (i, c_exemplar))
        print ('- Exemplar R-%d: %s' % (i, r_exemplar))
      print()

    # Print decoded target sentences.
    if not test_result:
      return

    if test_result.candidates is not None:
      for i, (candidate, loss) in enumerate(zip(test_result.candidates, test_result.pred_loss)):
        candidate = decorate_func(candidate, vocab.decoder)
        if test_result.pred_loss is not None:
          print ('- Hypothesis%02d (loss=%.3f):' % (i, loss), candidate)
        else:
          print ('- Hypothesis%02d:' % (i), candidate)

  @classmethod
  def output_tests(self_cls, flat_batches, vocab, results, evaluation_results,
                   output_path=None):
    if output_path:
      sys.stderr.write("Output results to \'{}\' .\n".format(output_path))

    # Output the detail of all testing examples.
    sys.stdout = open(output_path + '.summary', 'w') if output_path else sys.stdout
    for i, (b, result) in enumerate(zip(flat_batches, results)):
      _id = '[%04d]' % (i)
      print (_id)
      self_cls.print_example(b, vocab, test_result=result)
      print ('')

    print('<Evaluation Scores>')
    for metric in evaluation_results:
      print('%s\t%.3f' % (metric, evaluation_results[metric]))

    sys.stdout = open(output_path + '.outputs', 'w') if output_path else sys.stdout
    for r in results:
      print(' '.join(word_tokenize(r.candidates[0])))
    sys.stdout = sys.__stdout__

    sys.stdout = open(output_path + '.scores', 'w') if output_path else sys.stdout
    for metric in evaluation_results:
      print('%s,%.3f' % (metric, evaluation_results[metric]))
    sys.stdout = sys.__stdout__
    return evaluation_results


class _MultiTurnDialogTrainDatasetBase(object):
  def filter_example(self, example):
    # Exclude too short or too long responses in training.
    if self.minlen.word and len(example.inp_response.word) < self.minlen.word:
      return False
    if self.maxlen.word and len(example.inp_response.word) > self.maxlen.word:
      return False
    return True

  # def create_examples(self, dialogs, exemplars=None):
  #   examples = []
  #   for dial_idx, dialog in enumerate(dialogs):
  #     for uttr_idx in range(1, len(dialog)):
  #       example = self.create_example(dialog[:uttr_idx], dialog[uttr_idx], 
  #                                     self.vocab, self.maxlen.sent)
  #       if self.filter_example(example):
  #         examples.append(example)
  #   return examples

class _MultiTurnDialogTestDatasetBase(object):
  def filter_example(self, example):
    return True


class _MultiTurnDialogDatasetBaseWithExemplar(_MultiTurnDialogDatasetBase):
  def __init__(self, *args, exemplars=None, **kwargs):
    self.exemplars = exemplars
    super(_MultiTurnDialogDatasetBaseWithExemplar, self).__init__(*args, **kwargs)

  def load_data(self, source_path, max_rows):
    sys.stderr.write("Loading dataset from \'%s\'... \n" % source_path)
    data = load_dialogs(source_path, max_rows=max_rows,
                        preprocess=preprocess)

    source_dir, filename = separate_path_and_filename(source_path)
    exemplar_id_path = self.config.exemplar_id_dir + '/' + filename + self.config.exemplar_id_suffix
    sys.stderr.write("Loading exemplar_ids from \'%s\'... \n" % exemplar_id_path)

  
    self.exemplar_ids = load_exemplar_ids(exemplar_id_path, 
                                          self.num_exemplars,
                                          max_rows=max_rows)
    
    self.data = self.create_examples(data)
    sys.stderr.write("Finish loading.\n")

  def create_examples(self, dialogs):
    exemplar_ids = self.exemplar_ids
    examples = [self.create_example(dialog[:-1], dialog[-1], self.vocab, self.maxlen.sent, exids) for dialog, exids in zip(dialogs, exemplar_ids)]
    return examples

  def create_example(self, _context, _response, vocab, max_uttrs, exemplar_ids):
    example = super(_MultiTurnDialogDatasetBaseWithExemplar, self).create_example(_context, _response, vocab, max_uttrs)
    exemplars = [self.exemplars[_id] for _id in exemplar_ids]
    ex_contexts, ex_responses = [list(x) for x in zip(*exemplars)]
    ex_c_word_ids = [self.vocab.encoder.word.tokens2ids(x) for x in ex_contexts]
    ex_r_word_ids = [self.vocab.decoder.word.tokens2ids(x) for x in ex_responses]

    return list(example) + [ex_contexts, ex_c_word_ids, ex_responses, ex_r_word_ids]


  # def create_batch(self, data):
  #   '''
  #   Extract examples and pad them, and then create a batch.
  #   '''
  #   # batch = recDotDefaultDict()
  #   # for d in data:
  #   #   batch = batching_dicts(batch, d) # list of dictionaries to dictionary of lists.
  #   batched_data = list(zip(*data))
  #   batched_data, instant_vocab = self.assign_ids2unk_in_batch(batched_data)
  #   batch = self.padding(batched_data, minlen=self.minlen, maxlen=self.maxlen)
  #   batch.instant_vocab = instant_vocab
  #   return batch

  # TODO: dynamically change vocab size
  # def assign_ids2unk_in_batch(self, batch):
  #   inp_contexts, inp_c_word_ids, inp_responses, inp_r_word_ids = batch[:4]
  #   ex_contexts, ex_responses = batch[4:6] # [batch_size, num_exemplars, num_words]
  #   rev_vocab = self.vocab.decoder.word.rev_vocab[2:]
  #   all_words_in_batch = flatten(flatten(ex_responses))
  #   all_words_in_batch = set(all_words_in_batch)
  #   unk_words = all_words_in_batch - set(rev_vocab)
  #   instant_vocab = self.vocab.decoder.word
  #   instant_vocab = WordVocabularyFromList(
  #     rev_vocab + list(unk_words), 
  #     base_vocab=self.vocab.decoder.word)

  #   # [batch_size, num_exemplars, num_words]
  #   ex_c_word_ids = [[self.vocab.encoder.word.tokens2ids(x) for x in ex_c] for ex_c in ex_contexts]
  #   ex_r_word_ids = [[instant_vocab.tokens2ids(x) for x in ex_r] for ex_r in ex_responses]
  #   # [batch_size, num_words]
  #   inp_r_word_ids = [instant_vocab.tokens2ids(x) for x in inp_responses]
  #   batched_data = [inp_contexts, inp_c_word_ids, inp_responses, inp_r_word_ids, 
  #                   ex_contexts, ex_c_word_ids, ex_responses, ex_r_word_ids]
  #   return batched_data, instant_vocab

  def padding(self, _batch, minlen=None, maxlen=None):
    batch = super(_MultiTurnDialogDatasetBaseWithExemplar, self).padding(_batch[:4], minlen=minlen, maxlen=maxlen)
    ex_context, ex_c_word_ids, ex_response, ex_r_word_ids = _batch[4:]

    batch.ex_context = dotDict()
    batch.ex_response = dotDict()

    batch.ex_context.raw = ex_context
    batch.ex_response.raw = ex_response

    # [batch_size, n_max_exemplar, n_max_word]


    batch.ex_context.word = _padding(
      ex_c_word_ids,
      minlen=[None, minlen.word],
      maxlen=[None, maxlen.word])
    batch.ex_response.word = _padding(
      ex_r_word_ids,
      minlen=[None, minlen.word],
      maxlen=[None, maxlen.word])
    return batch



class _MultiTurnDialogTrainDataset(
    _MultiTurnDialogDatasetBase, 
    _MultiTurnDialogTrainDatasetBase):
  pass

class _MultiTurnDialogTestDataset(
    _MultiTurnDialogDatasetBase, 
    _MultiTurnDialogTestDatasetBase):
  pass

class _MultiTurnDialogTrainDatasetWithExemplar(
    _MultiTurnDialogDatasetBaseWithExemplar, 
    _MultiTurnDialogTrainDatasetBase):
  pass

class _MultiTurnDialogTestDatasetWithExemplar(
    _MultiTurnDialogDatasetBaseWithExemplar, 
    _MultiTurnDialogTestDatasetBase):
  pass



class MultiTurnDialogDataset(DatasetBase):
  train_class = _MultiTurnDialogTrainDataset
  test_class = _MultiTurnDialogTestDataset

class MultiTurnDialogDatasetWithExemplar(DatasetBase):
  train_class = _MultiTurnDialogTrainDatasetWithExemplar
  test_class = _MultiTurnDialogTestDatasetWithExemplar
  def __init__(self, config, vocab):
    self.vocab = vocab
    exemplar_path = config.source_dir + '/' + config.filename.exemplar
    sys.stderr.write("Loading exemplar candidates from \'%s\'... \n" % exemplar_path)
    self.exemplars = load_exemplars(exemplar_path, 
                                    max_rows=config.max_rows.exemplar,
                                    preprocess=preprocess)
    self.train = self.train_class(vocab,
                                  config,
                                  config.filename.train,
                                  config.max_rows.train,
                                  config.maxlen.train,
                                  config.minlen.train,
                                  config.batch_size.train,
                                  exemplars=self.exemplars,
                                  do_shuffle=True)
    self.valid = self.test_class(vocab,
                                 config,
                                 config.filename.valid,
                                 config.max_rows.valid,
                                 config.maxlen.test,
                                 config.minlen.test,
                                 config.batch_size.test,
                                 exemplars=self.exemplars,
                                 do_shuffle=False)

    self.test = self.test_class(vocab,
                                config,
                                config.filename.test,
                                config.max_rows.test,
                                config.maxlen.test,
                                config.minlen.test,
                                config.batch_size.test,
                                exemplars=self.exemplars,
                                do_shuffle=False)



