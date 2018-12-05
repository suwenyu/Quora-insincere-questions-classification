import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

from gensim.models import KeyedVectors

import os
import collections


def preprocessing_data(path, feature_selection, sentence_len, embedding_size, preprocess):
	# path = "all/"

	# raw_data = pd.read_csv(path+'train_data_2', encoding='utf-8')
	# raw_data_test = pd.read_csv(path+'test_data_2', encoding='utf-8')

	raw_data = pd.read_csv(path+'train.csv', encoding='utf-8')
	raw_data_test = pd.read_csv(path+'test.csv', encoding='utf-8')

	print(raw_data.shape)
	print(raw_data.columns)
	# print('\n')
	# print(raw_data.head(5))

	stop = stopwords.words('english')

	if preprocess:
		print("Removing Punctuation and turning into lower case")
		raw_data['question_text'] = raw_data['question_text'].str.replace('[^\w\s]','')
		raw_data['question_text'] = raw_data['question_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
		raw_data['question_text'] = raw_data['question_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

		raw_data_test['question_text'] = raw_data_test['question_text'].str.replace('[^\w\s]','')
		raw_data_test['question_text'] = raw_data_test['question_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
		raw_data_test['question_text'] = raw_data_test['question_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))


	print(raw_data['question_text'].head())
	# print(aaa)
	all_raw_data_text = raw_data.question_text.values.tolist() + raw_data_test.question_text.values.tolist()
	# all_raw_data_text = raw_data.question_text.values.tolist() 

	# print("loading word2vec embedding")

	# EMBEDDING_FILE = path + 'embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
	# embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)



	print("calculate tokenize...")

	if feature_selection=="tokenize":
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(all_raw_data_text)
		tr_vec = tokenizer.texts_to_sequences(raw_data.question_text.values)
		te_vec = tokenizer.texts_to_sequences(raw_data_test.question_text.values)

		tr_vec = pad_sequences(tr_vec, maxlen=sentence_len)
		te_vec = pad_sequences(te_vec, maxlen=sentence_len)


		vocab_size = len(tokenizer.word_index)+1

	# embedding_matrix = np.zeros((vocab_size, 300))
	# for word, i in tokenizer.word_index.items():
	# 	if word in embeddings_index:
	# 		embedding_vector = embeddings_index.get_vector(word)
	# 		embedding_matrix[i] = embedding_vector
	if feature_selection=="tfidf_tokenize":
		toke = "unigram"
		if toke == "unigram":
			ngram_range=(1,1)
		else:
			ngram_range=(2,2)
		tokenizer = TfidfVectorizer(use_idf=True,ngram_range=ngram_range)
		tokenizer.fit(all_raw_data_text)

		tr_vec = tokenizer.transform(raw_data.question_text)
		te_vec = tokenizer.transform(raw_data_test.question_text)
		vocab_size = len(tokenizer.get_feature_names())+1
	# max_idf = max(tokenizer.idf_)
	# print(max_idf)
	# word2weight = collections.defaultdict( lambda: max_idf, [(w, tokenizer.idf_[i]) for w, i in tokenizer.vocabulary_.items()])
	# print(word2weight)
	# print("success change tf-idf to w2v")
	print("Success finishing feature selection...")

	tr_ans = raw_data.target.values

	print("Split data into validation data with same random state...")
	X_train, X_val, y_train, y_val = train_test_split(tr_vec, tr_ans, test_size=0.2, random_state=0)

	return X_train, y_train, X_val, y_val, te_vec, vocab_size

# X_train, X_test, y_train, y_train, tr_ans, vocab_size = preprocessing_data(path="all/", feature_selection="tokenize", sentence_len = 70, embedding_size = 100)

