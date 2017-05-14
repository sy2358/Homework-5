import numpy as np
import math
# sigmoid
from scipy.special import expit

# returns word index
def getWordIdx(word):
  idx = np.where(ptb_wtoi == word)[0]
  if len(idx) == 0:
    idx = np.where(ptb_wtoi == "<unk>")[0]
  return idx[0]

# returns embedding given word index
def getEmbedding(idx):
  return embedding_mat[idx]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# given input, hidden layer value of t-1, and cell value of t-1
# calculate new states of the cell
def applyLSTM(x, h, c, bias, weight):
  xh = np.concatenate((x, h), axis=0)
  out = np.add(np.matmul(xh, weight), bias)
  i_c_f_o = np.split(out, 4)
  i = expit(i_c_f_o[0])
  f = expit(i_c_f_o[2])
  o = expit(i_c_f_o[3])
  c_cand = np.tanh(i_c_f_o[1])
  nc = c*f+c_cand*i
  nh = np.tanh(nc)*o
  return nh, nc

def applySoftmax(x, bias, weight):
  return softmax(np.matmul(x, weight)+bias)

# initialize the LSTM
def initLSTM():
  return np.zeros(650), np.zeros(650), np.zeros(650), np.zeros(650)

# one step of LSTM giving word, and the states of the LSTM
# return output layer and new values for the states
def applyLM(widx, h_0, c_0, h_1, c_1):
  emb = getEmbedding(widx)

  # first layer
  new_h_0, new_c_0 = applyLSTM(emb, h_0, c_0, lstm_0_b, lstm_0_w)
  # second layer
  new_h_1, new_c_1 = applyLSTM(new_h_0, h_1, c_1, lstm_1_b, lstm_1_w)
  # softmax
  o = applySoftmax(new_h_1, softmax_b, softmax_w)

  h_0 = new_h_0
  h_1 = new_h_1
  c_0 = new_c_0
  c_1 = new_c_1

  return o, h_0, c_0, h_1, c_1

def probability(p0, sentence):
  # initialize hidden and cells
  h_0, c_0, h_1, c_1 = initLSTM()

  # log probability starting with p0
  log_p = math.log(p0)

  for i in range(len(sentence)-1):
    widx = getWordIdx(sentence[i])
    o, h_0, c_0, h_1, c_1 = applyLM(widx, h_0, c_0, h_1, c_1)

    # next word idx
    nidx = getWordIdx(sentence[i+1])

    log_p = log_p + math.log(o[nidx])

  print(sentence,log_p)
  return log_p

def greedy(word):
  # initialize hidden and cells
  h_0, c_0, h_1, c_1 = initLSTM()

  sentence = [word]

  while not(word=='<eos>'):
    widx = getWordIdx(word)
    o, h_0, c_0, h_1, c_1 = applyLM(widx, h_0, c_0, h_1, c_1)

    # next word idx is the one with highest index in the output layer
    # just exclude <unk>
    o[getWordIdx('unk')]=0
    nidx = np.argmax(o)
    word = ptb_wtoi[nidx]
    sentence.append(word)

  print('generated: ', sentence)

def beam(word, p0, beam):
  # initialize hidden and cells - several of the, since we need to maintain 2 states of the LSTM
  h_0s = []
  c_0s = []
  h_1s = []
  c_1s = []
  log_ps = []
  sentences = []

  # initialize the stack of hypothesis - we have only 1 at the beginning
  # we keep the probability
  for i in range(beam):
    h_0, c_0, h_1, c_1 = initLSTM()
    log_ps.append(math.log(p0))
    h_0s.append(h_0)
    c_0s.append(c_0)
    h_1s.append(h_1)
    c_1s.append(c_1)
    sentences.append([word])

  n = 0
  # while we have not reach <eos> for all search
  while beam>0:
    hypothesis = []
    # explore current hypothesis to generate 'beam' ones
    for i in range(beam):
      widx = getWordIdx(sentences[i][n])
      o, h_0, c_0, h_1, c_1 = applyLM(widx, h_0s[i], c_0s[i], h_1s[i], c_1s[i])
      # next word idx is the one with highest index in the output layer
      # just exclude <unk>
      o[getWordIdx('unk')]=0
      # sort the results to keep only the best
      idxsort = np.argsort(o)
      # push the beam-best hypothesis
      for j in range(beam):
        idx = idxsort[9999-j]
        # calculate global probability and keep the hypothesis
        hypothesis.append([log_ps[j]+math.log(o[idx]), ptb_wtoi[idx], h_0, c_0, h_1, c_1, sentences[i]])

    # we have kept beam*beam hypothesis - sort and keep the best to continue
    hypothesis.sort(key=lambda x: x[0], reverse=True)

    j = 0
    i = 0
    while j < beam:
      log_p = hypothesis[i][0]
      word = hypothesis[i][1]
      h_0 = hypothesis[i][2]
      c_0 = hypothesis[i][3]
      h_1 = hypothesis[i][4]
      c_1 = hypothesis[i][5]
      # copy the sentence
      sentence = list(hypothesis[i][6])
      sentence.append(word)
      if word == '<eos>':
        print('generated: ', sentence)
        beam = beam - 1
      else:
        # check if we don't have the same hypothesis yet
        found = False
        for k in range(j):
          found = found or sentences[k] == sentence
        if not found:
          log_ps[j] = log_p
          h_0s[j] = h_0
          c_0s[j] = c_0
          h_1s[j] = h_1
          c_1s[j] = c_1
          sentences[j] = sentence
          j = j + 1
      i = i + 1
    n = n + 1

print('- loading embedding...')
embedding_mat = np.load('parameters/embedding.npy')

print('- loading word to index')
ptb_wtoi = np.loadtxt('ptb_word_to_id.txt',delimiter='\t',comments = '###',usecols = 0, dtype=bytes).astype(str)

print('- loading LSTM parameters...')
lstm_0_b = np.load('parameters/lstm_0_b.npy')
lstm_0_w = np.load('parameters/lstm_0_w.npy')
lstm_1_b = np.load('parameters/lstm_1_b.npy')
lstm_1_w = np.load('parameters/lstm_1_w.npy')

print('- loading softmax parameters...')
softmax_b = np.load('parameters/softmax_b.npy')
softmax_w = np.load('parameters/softmax_w.npy')

print('- calculate probability on test ...')
probability(0.00135759, ['i', 'have', 'a', 'cat', 'in', 'my', 'home', '<eos>'])
probability(0.00135759, ['i', 'have', 'a', 'cat', 'in', 'my', 'house', '<eos>'])
probability(0.00135759, ['i', 'have', 'a', 'cat', 'in', 'me', 'house', '<eos>'])

print('- greedy generation:')
greedy('some')
greedy('where')

print('- beam search generation (2):')
beam('some', 0.00179434, 2)
beam('where', 0.00179434, 2)

print('- beam search generation (3):')
beam('some', 0.000415237, 3)
beam('where', 0.000415237, 3)

print('- beam search generation (4):')
beam('some', 0.000415237, 4)
beam('where', 0.000415237, 4)
