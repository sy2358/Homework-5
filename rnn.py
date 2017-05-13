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
  c_cand = np.tanh(i_c_f_o[1])
  f = expit(i_c_f_o[2])
  o = expit(i_c_f_o[3])
  nc = c*f + c_cand*i
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

def getPerplexity(sentence):
  # initialize hidden and cells
  h_0, c_0, h_1, c_1 = initLSTM()

  loss = 0

  for i in range(len(sentence)-1):
    widx = sentence[i]
    o, h_0, c_0, h_1, c_1 = applyLM(widx, h_0, c_0, h_1, c_1)

    # calculate the loss for each word
    loss = loss + -math.log(o[sentence[i+1]])

    if (i+1)%1000 == 0:
      print('word ',i+1,' - current ppl:',loss/(i+1))

  print('perplexity: ',loss/len(sentence))

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

print('- read test file')
with open("ptb_test_index.txt") as f:
  test = f.readlines()

test = [int(x.strip()) for x in test]

print('- calculate perplexity on test file')
getPerplexity(test)

