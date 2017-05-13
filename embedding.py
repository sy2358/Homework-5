import numpy as np
from scipy.spatial.distance import cosine
import sys

# if setting 'large' in commandline, use word embedding trained on very large corpus
embedding = ['parameters/embedding.npy', 'ptb_word_to_id.txt']
if len(sys.argv)>1 and sys.argv[1]=='large':
  embedding = ['parameters/embedding-large.npy', 'large_word_to_id.txt']

# returns word index
def getWordIdx(word):
  idx = np.where(ptb_wtoi == word)[0]
  if len(idx) == 0:
    idx = np.where(ptb_wtoi == "<unk>")[0]
  return idx[0]

# returns embedding given word index
def getEmbedding(idx):
  return embedding_mat[idx]

print('- loading embedding...')
embedding_mat = np.load(embedding[0])

print('- loading word to index')
ptb_wtoi = np.loadtxt(embedding[1],delimiter='\t',comments = '###',usecols = 0, dtype=bytes).astype(str)

words_1 = ['the', 'discount', 'crazy', 'birthday', 'just']

print('1) nearest words (displaying 5 nearest)')

for w in words_1:
  idx, embedding = getEmbedding(getWordIdx(w))
  # calculate cosine distance to all other idx and sort by distance
  cosine_dist=sorted([(cosine(embedding, embedding_mat[j]), j) for j in range(len(ptb_wtoi)) if not j == idx])
  # display the top 5 closest
  print(w,'==>',[ptb_wtoi[j] for d,j in cosine_dist[0:5]])

print()
print('2) word vector calculation (displaying 5 nearest)')

analogy = [['king', 'male', 'female'],
           ['breakfast', 'morning', 'evening'],
           ['japan','tokyo','london']]

for ana in analogy:
  idxa, embeddinga = getEmbedding(getWordIdx(ana[0]))
  idxb, embeddingb = getEmbedding(getWordIdx(ana[1]))
  idxc, embeddingc = getEmbedding(getWordIdx(ana[2]))
  embedding = embeddinga-embeddingb+embeddingc

  # calculate cosine distance to all other idx and sort by distance
  cosine_dist=sorted([(cosine(embedding, embedding_mat[j]), j) for j in range(len(ptb_wtoi)) if j not in [idxa,idxb,idxc]])
  # display the top 5 closest
  print(ana[0]+'-'+ana[1]+'+'+ana[2],'==>',[ptb_wtoi[j] for d,j in cosine_dist[0:5]])
