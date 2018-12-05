import numpy as np
import pandas as pd
# from keras.layers import Dense, SimpleRNN, GRU, LSTM, Embedding
# from keras.models import Sequential
from keras.preprocessing.text import Tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

import os
import re
print(os.listdir("all/"))

path = "all/"

raw_data = pd.read_csv(path + 'train_data_2', encoding='utf-8')
raw_data_test = pd.read_csv(path + 'test_data_2', encoding='utf-8')

# raw_data = pd.read_csv(path + 'train.csv', encoding='utf-8')
# raw_data_test = pd.read_csv(path + 'test.csv', encoding='utf-8')


print(raw_data.shape)
print(raw_data.columns)
# print('\n')
# print(raw_data.head(5))

stop = stopwords.words('english')

print("Removing Punctuation and turning into lower case")
raw_data['question_text'] = raw_data['question_text'].str.replace('[^\w\s]','')
raw_data['question_text'] = raw_data['question_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
raw_data['question_text'] = raw_data['question_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


raw_data_test['question_text'] = raw_data_test['question_text'].str.replace('[^\w\s]','')
raw_data_test['question_text'] = raw_data_test['question_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
raw_data_test['question_text'] = raw_data_test['question_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))


question_len = []

count = 0
for i in (raw_data['question_text'].values):
	# print(i)
	re.sub(r'[^\x00-\x7F]+',' ', i)
	# print(i.split(' '))
	question_len.append(len(i.split(' ')))
	# for j in i.split(' '):
	# 	test = ",".join(j) 
	# print (test)
	# if len(i.split(' ')) == 0:
	# count += 1

# for i in (raw_data_test['question_text']):
# 	if len(i.split(' ')) == 0:
# 		count += 1

# print(count)
pd = pd.Series(question_len)
# print(pd.columns)
question_text_distribution = pd.groupby(pd.values).agg(['count']).to_dict()
print(question_text_distribution['count'])

width = 0.5

import matplotlib.pyplot as plt
labels = list(question_text_distribution['count'].keys())
values = list(question_text_distribution['count'].values())


p = plt.bar(labels, values, width)
plt.ylabel('Counts')
plt.xticks(labels)
plt.show()


# print("Removing Punctuation and turning into lower case")
# raw_data_test['question_text'] = raw_data_test['question_text'].str.replace('[^\w\s]','')
# raw_data_test['question_text'] = raw_data_test['question_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# raw_data_test['question_text'] = raw_data_test['question_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# target_distribution = (raw_data.groupby(['target']).agg(['count']).to_dict())[(u'qid', 'count')]
# print(target_distribution)

# width = 0.35

# import matplotlib.pyplot as plt
# labels = list(target_distribution.keys())
# values = list(target_distribution.values())

# print(values)

# p = plt.bar(labels, values, width)
# plt.ylabel('Counts')
# plt.xticks(labels, ('0', '1'))
# plt.show()

# print(raw_data['question_text'].head())
# all_raw_data_text = raw_data.question_text.values.tolist() + raw_data.question_text.values.tolist()


# print ('initilize tf vectorizer ...')
# vectorizer = TfidfVectorizer(use_idf=False, stop_words='english')
# vectorizer.fit(all_raw_data_text)

# print ('transform data to tfidf vector ...')
# tr_vec = vectorizer.transform(raw_data.question_text)
# te_vec = vectorizer.transform(raw_data_test.question_text)
# print(tr_vec.shape)
# print(te_vec.shape)

