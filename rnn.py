import numpy as np
import math
# sigmoid
from scipy.special import expit

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def applyLSTM(x, h, c, bias, weight):
  xh = np.concatenate((x, h), axis=0)
  out = np.add(np.matmul(xh, weight), bias)
  i_c_f_o = np.split(out, 4)
  i = expit(i_c_f_o[0])
  f = expit(i_c_f_o[2])
  o = expit(i_c_f_o[3])
  c_cand = np.tanh(i_c_f_o[1])
  nc = c*f+c_cand*i
  nh = np.tanh(c)*o
  return nh, nc

def applySoftmax(x, bias, weight):
  return softmax(np.matmul(x, weight)+bias)

def getPerplexity(sentence):
  # initialize hidden and cells
  h_0 = np.zeros(650)
  c_0 = np.zeros(650)
  h_1 = np.zeros(650)
  c_1 = np.zeros(650)

  loss = 0

  for i in range(len(sentence)-1):
    idx = sentence[i]

    emb = embedding_mat[idx]

    # first layer
    new_h_0, new_c_0 = applyLSTM(emb, h_0, c_0, lstm_0_b, lstm_0_w)
    # second layer
    new_h_1, new_c_1 = applyLSTM(new_h_0, h_1, c_1, lstm_1_b, lstm_1_w)
    # softmax
    o = applySoftmax(new_h_1, softmax_b, softmax_w)

    # calculate the loss for each word
    loss = loss + -math.log(o[sentence[i+1]])

    # pass values to following word
    h_0 = new_h_0
    h_1 = new_h_1
    c_0 = new_c_0
    c_1 = new_c_1

  print(sentence,loss/len(sentence))

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

with open("ptb_test_index.txt") as f:
  test = f.readlines()
test = [int(x.strip()) for x in test]

getPerplexity(test)

