#coding: utf-8
from core.utils import common
import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_rank(scores):
  '''
  r : a list of link connection probabilities where a correct one is inserted at the beginning of the corresponding candidates.
  '''
  sorted_idx = np.argsort(scores)[::-1]
  pos_rank = np.where(sorted_idx == 0)[0][0] + 1
  return pos_rank, sorted_idx


def mrr(ranks):
  return sum([1.0 / r for r in ranks]) / float(len(ranks))

def hits_k(ranks, k=10):
  return 100.0 * len([r for r in ranks if r <= 10]) / len(ranks)


def calc_bleu(hypotheses, references):
  '''
  https://www.aclweb.org/anthology/P02-1040
  '''
  assert len(hypotheses) == len(references)
  assert type(hypotheses[0]) == str
  assert type(references[0]) == str
  
  n_data = len(hypotheses)
  bleu = 0
  for hypothesis, reference in zip(hypotheses, references):
    bleu += sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method2) * 100.0
  bleu /= n_data
  return bleu


def calc_dist(hypotheses, dist_n):
  '''
  https://www.aclweb.org/anthology/N16-1014
  '''
  assert dist_n >= 1
  assert type(hypotheses[0]) == str

  n_total_words = 0
  uniq_words = set()
  for hypothesis in hypotheses:
    words_in_hyp = [x for x in hypothesis.split() if x]
    ngrams = [tuple(words_in_hyp[i:i+dist_n]) for i in range(len(words_in_hyp)- dist_n+1)]
    for ngram in ngrams:
      uniq_words.add(ngram)
    n_total_words += len(ngrams)
  return 1.0 * len(uniq_words) / n_total_words


def calc_length(hypotheses):
  assert type(hypotheses[0]) == str

  lengths = [len([x for x in hypothesis.split() if x]) for hypothesis in hypotheses]
  average_length = 1.0 * sum(lengths) / len(lengths)
  return average_length


def calc_cos_sim(hypotheses, references):
  assert len(hypotheses) == len(references)
  assert type(hypotheses[0]) == str
  assert type(references[0]) == str
  raise NotImplementedError


def calc_actent(hypotheses, references):
  assert len(hypotheses) == len(references)
  assert type(hypotheses[0]) == str
  assert type(references[0]) == str
  raise NotImplementedError
