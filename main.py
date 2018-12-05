import numpy as np
import pandas as pd

from keras.layers import Dense, SimpleRNN, GRU, LSTM, Embedding, Input, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import merge, TimeDistributed, Lambda, Flatten, Activation, RepeatVector, Permute, Bidirectional, Conv1D, Average, average
from keras import backend as K
from keras.layers.merge import Dot

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

from gensim.models import KeyedVectors

import os
import sys

import collections
import matplotlib.pyplot as plt

#local files
import preprocess
import models

# print(os.listdir("all/"))
path = "all/"

### set parameters 
sentence_len = 70
embedding_size=100


feature_selection = sys.argv[1]
model = sys.argv[2]

### preprocessing data
# X_train, y_train, X_val, y_val, te_vec, vocab_size = preprocess.preprocessing_data(path=path, feature_selection="tokenize", sentence_len = sentence_len, embedding_size = embedding_size)

X_train, y_train, X_val, y_val, te_vec, vocab_size = preprocess.preprocessing_data(path=path, feature_selection=feature_selection, sentence_len = sentence_len, embedding_size = embedding_size, preprocess=True)

### build model
# pred_test_y = models.model('BernoulliNB', vocab_size=vocab_size, sentence_len=sentence_len, embedding_size=embedding_size, tr_vec=X_train, tr_ans=y_train, val_vec=X_val, val_ans=y_val ,te_vec=te_vec )
pred_test_y = models.model(model, vocab_size=vocab_size, sentence_len=sentence_len, embedding_size=embedding_size, tr_vec=X_train, tr_ans=y_train, val_vec=X_val, val_ans=y_val ,te_vec=te_vec )
# print(model.history.history['acc'])

### plot the learning rate
# plt.plot(model.history.history['acc'])
# plt.plot(model.history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


submission=pd.read_csv(path + "sample_submission.csv")
submission['prediction']= pd.DataFrame(pred_test_y)

submission.to_csv('submission.csv', index=False)

