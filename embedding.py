import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cosine

def getEmbedding(word):
  idx = np.where(ptb_wtoi == word)[0][0]
  return idx, embedding_mat[idx]

print('- loading embedding...')
embedding_mat = np.load('parameters/embedding.npy')

# [word]
print('- loading word to index')
ptb_wtoi = np.loadtxt('ptb_word_to_id.txt',delimiter='\t',comments = '###',usecols = 0, dtype=bytes).astype(str)

words_1 = ['the','discount','crazy','birthday','just']

print('1) nearest words (displaying 5 nearest)')

for w in words_1:
  idx, embedding = getEmbedding(w)
  # calculate cosine distance to all other idx and sort by distance
  cosine_dist=sorted([(cosine(embedding, embedding_mat[j]), j) for j in range(len(ptb_wtoi)) if not j == idx])
  # display the top 5 closest - exclude the first one which is the word itself
  print(w,'==>',[ptb_wtoi[j] for d,j in cosine_dist[0:5]])

print()
print('2) word vector calculation (displaying 5 nearest)')

analogy = [['king', 'male', 'female'],
           ['breakfast', 'morning', 'evening'],
           ['japan','tokyo','paris']]

for ana in analogy:
  idxa, embeddinga = getEmbedding(ana[0])
  idxb, embeddingb = getEmbedding(ana[1])
  idxc, embeddingc = getEmbedding(ana[2])
  embedding = embeddinga-embeddingb+embeddingc

  # calculate cosine distance to all other idx and sort by distance
  cosine_dist=sorted([(cosine(embedding, embedding_mat[j]), j) for j in range(len(ptb_wtoi)) if j not in [idxa,idxb,idxc]])
  # display the top 5 closest
  print(ana[0]+'-'+ana[1]+'+'+ana[2],'==>',[ptb_wtoi[j] for d,j in cosine_dist[0:5]])
