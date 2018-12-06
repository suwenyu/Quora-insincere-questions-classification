from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

from keras.layers import Dense, SimpleRNN, GRU, LSTM, Embedding, Input, Dropout
from keras.models import Sequential
from keras.layers import merge, TimeDistributed, Lambda, Flatten, Activation, RepeatVector, Permute, Bidirectional, Conv1D, Average, average
from keras.layers.merge import Dot
from keras import backend as K

from keras.models import model_from_json

import pickle
import os


def selfattentiveModel(vocab_size, sentence_len, embedding_size, tr_vec, tr_ans, val_vec, val_ans ,te_vec):
	inp = Input(shape=(sentence_len,))
	# x = Embedding(max_num_words, seq_len, weights=[embedding_matrix])(inp)
	x = Embedding(input_dim = vocab_size, 
					input_length = sentence_len, 
					output_dim = embedding_size)(inp)

	x = Bidirectional(LSTM(128, return_sequences=True))(x)
	attention = TimeDistributed(Dense(256, activation='tanh'))(x)
	# attention = Permute([2, 1])(x)
	# print(attention.shape)

	attention = Dropout(0.1)(attention)
	attention = Dense(350,input_shape=(256,70,))(attention)
	attention = Activation('tanh')(attention)
	attention = Dropout(0.1)(attention)
	attention = Dense(30,input_shape=(350,))(attention)

	attention = Activation('softmax')(attention)
	# attention = Permute([2, 1])(attention)

	# print(attention.shape)

	# attention = Conv1D(filters=1, kernel_size=350, activation='tanh')(x)
	# attention = Conv1D(filters=350, kernel_size=30, activation='linear')(attention)
	# attention = Lambda(lambda x: K.softmax(x, axis=1), name="attention_vector")(attention)
	weighted_sequence_embedding = Dot(axes=[1, 1], normalize=False)([attention, x])
	print(weighted_sequence_embedding.shape)
	lstm_em = Lambda(lambda x: K.sum(x, axis=1))(weighted_sequence_embedding)
	# outp = Lambda(lambda x: K.l2_normalize(K.sum(x, axis=1)))(weighted_sequence_embedding)
	# outp = Dense(1,input_shape=(256,))(outp)
	# outp = average(outp)
	lstm_em = Dense(64, activation="relu")(lstm_em)
	lstm_em = Dropout(0.1)(lstm_em)
	outp = Dense(1, activation="sigmoid")(lstm_em)

	model = Model(inputs=inp, outputs=outp)
	print(model.summary())

	from keras.optimizers import Adam
	adam = Adam(lr=0.001)

	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	
	if os.path.isfile('save/selfattentiveModel.json'):
		json_file = open('save/selfattentiveModel.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
	else:
		model.fit(tr_vec, tr_ans,batch_size=512 ,epochs=2, validation_data=(val_vec, val_ans), verbose=True)
		model_json = model.to_json()
		with open("save/selfattentiveModel.json", "w") as json_file:
			json_file.write(model_json)

	print ('make validation predictions ...')
	print(model.evaluate(x=val_vec, y=val_ans, batch_size=1024, verbose=1))
	
	#clf_predictions = clf.predict_proba(te_vec)
	preds = model.predict(te_vec, batch_size=1024)
	pred_test_y = (preds>0.35).astype(int)
	return pred_test_y


def lstmModel(vocab_size, sentence_len, embedding_size, tr_vec, tr_ans, val_vec, val_ans ,te_vec):
	inp = Input(shape=(sentence_len,))
	# x = Embedding(max_num_words, seq_len, weights=[embedding_matrix])(inp)
	x = Embedding(input_dim = vocab_size, 
					input_length = sentence_len, 
					output_dim = embedding_size)(inp)

	x = LSTM(128, return_sequences=False)(x)
	lstm_em = Dense(64, activation="relu")(x)
	lstm_em = Dropout(0.1)(lstm_em)
	outp = Dense(1, activation="sigmoid")(lstm_em)


	model = Model(inputs=inp, outputs=outp)
	print(model.summary())

	from keras.optimizers import Adam
	adam = Adam(lr=0.001)

	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	# model.fit(tr_vec, tr_ans,batch_size=512 ,epochs=2, validation_data=(val_vec, val_ans), verbose=True)

	if os.path.isfile('save/lstmModel.json'):
		json_file = open('save/lstmModel.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
	else:
		model.fit(tr_vec, tr_ans,batch_size=512 ,epochs=2, validation_data=(val_vec, val_ans), verbose=True)
		model_json = model.to_json()
		with open("save/lstmModel.json", "w") as json_file:
			json_file.write(model_json)

	print ('make validation predictions ...')
	print(model.evaluate(x=val_vec, y=val_ans, batch_size=1024, verbose=1))

	#clf_predictions = clf.predict_proba(te_vec)
	preds = model.predict(te_vec, batch_size=1024)
	pred_test_y = (preds>0.35).astype(int)
	return pred_test_y


def bernoulliNB(tr_vec, tr_ans, val_vec, val_ans, te_vec, feature_selection):
	from sklearn.naive_bayes import BernoulliNB

	# print(os.path.isfile('save/bernoulliNB_'+feature_selection+'.pickle'))
	if os.path.isfile('save/bernoulliNB_'+feature_selection+'.pickle'):
		with open('save/bernoulliNB_'+feature_selection+'.pickle', 'rb') as f:
			clf = pickle.load(f)
			print(clf.score(val_vec, val_ans))
	else:
		clf = BernoulliNB(alpha=1.0)
		clf.fit(tr_vec, tr_ans)
		print(clf.score(val_vec, val_ans))
		
		with open('save/bernoulliNB_'+feature_selection+'.pickle', 'wb') as f:
			pickle.dump(clf, f)


	print ('make predictions ...')
	#clf_predictions = clf.predict_proba(te_vec)
	preds = clf.predict(te_vec)
	pred_test_y = (preds>0.35).astype(int)
	return pred_test_y

def complementNB(tr_vec, tr_ans, val_vec, val_ans, te_vec):
	from sklearn.naive_bayes import ComplementNB
	clf = ComplementNB()
	clf.fit(tr_vec, tr_ans)
	print(clf.score(val_vec, val_ans))

	print ('make predictions ...')
	#clf_predictions = clf.predict_proba(te_vec)
	preds = clf.predict(te_vec)
	pred_test_y = (preds>0.35).astype(int)
	return pred_test_y

def multinomialNB(tr_vec, tr_ans, val_vec, val_ans, te_vec):
	from sklearn.naive_bayes import MultinomialNB
	clf = multinomialNB(alpha=1)
	clf.fit(tr_vec, tr_ans)
	print(clf.score(val_vec, val_ans))

	print ('make predictions ...')
	#clf_predictions = clf.predict_proba(te_vec)
	preds = clf.predict(te_vec)
	pred_test_y = (preds>0.5).astype(int)
	return pred_test_y


def model(modelname, feature_selection, vocab_size, sentence_len, embedding_size, tr_vec, tr_ans, val_vec, val_ans ,te_vec ):
	# Building an LSTM model
	print("Building Model")
	# if modelname == "MultinomialNB":
	# 	predict = multinomialNB(tr_vec, tr_ans, val_vec, val_ans ,te_vec, feature_selection)
	# 	return predict
	# if modelname == "ComplementNB":
	# 	predict = complementNB(tr_vec, tr_ans, val_vec, val_ans ,te_vec)
	# 	return predict
	if modelname == "BernoulliNB":
		predict = bernoulliNB(tr_vec, tr_ans, val_vec, val_ans ,te_vec, feature_selection)
		return predict
	if modelname == "SELFAttentive":
		predict = selfattentiveModel(vocab_size, sentence_len, embedding_size, tr_vec, tr_ans, val_vec, val_ans ,te_vec)
		return predict


	if modelname=="LSTM":
		predict = lstmModel(vocab_size, sentence_len, embedding_size, tr_vec, tr_ans, val_vec, val_ans ,te_vec)
		return predict


