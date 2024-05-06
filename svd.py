# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

la = np.linalg

corpus = ["woman a",
          "tôi thích NLP .",
          "tôi ghét ở một_mình"]

words = []
for sentences in corpus:
    words.extend(sentences.split())

words = list(set(words))
words.sort()

X = np.zeros([len(words), len(words)])

for sentences in corpus:
    tokens = sentences.split()
    for i, token in enumerate(tokens):
        if(i == 0):
            X[words.index(token), words.index(tokens[i + 1])] += 1
        elif(i == len(tokens) - 1):
            X[words.index(token), words.index(tokens[i - 1])] += 1
        else:
            X[words.index(token), words.index(tokens[i + 1])] += 1
            X[words.index(token), words.index(tokens[i - 1])] += 1

print(X)

U, s, Vh = la.svd(X, full_matrices=False)

plt.xlim(-1, 1)
plt.ylim(-1, 1)

for i in range(len(words)):
    plt.text(U[i, 0], U[i, 1], words[i])

plt.show()