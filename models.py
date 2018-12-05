from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

from keras.layers import Dense, SimpleRNN, GRU, LSTM, Embedding, Input, Dropout
from keras.models import Sequential
from keras.layers import merge, TimeDistributed, Lambda, Flatten, Activation, RepeatVector, Permute, Bidirectional, Conv1D, Average, average
from keras.layers.merge import Dot
from keras import backend as K

def lstmattentionModel(vocab_size, sentence_len, embedding_size, tr_vec, tr_ans, val_vec, val_ans ,te_vec):
	inp = Input(shape=(sentence_len,))
	# x = Embedding(max_num_words, seq_len, weights=[embedding_matrix])(inp)
	x = Embedding(input_dim = vocab_size, 
					input_length = sentence_len, 
					output_dim = embedding_size)(inp)

	x = LSTM(128, return_sequences=True)(x)
	
	# compute weight of the sequence
	attention = TimeDistributed(Dense(1, activation='tanh'))(x) 
	attention = Flatten()(attention)
	attention = Activation('softmax')(attention)
	attention = RepeatVector(128)(attention)
	attention = Permute([2, 1])(attention)

	# apply the attention
	sent_representation = Dot(axes=[1,1])([x, attention])
	sent_representation = Lambda(lambda x: K.sum(x, axis=1))(sent_representation)

	# probabilities = Dense(3, activation='softmax')(sent_representation)	

	lstm_em = Dense(64, activation="relu")(sent_representation)
	lstm_em = Dropout(0.1)(lstm_em)
	outp = Dense(1, activation="sigmoid")(lstm_em)


	model = Model(inputs=inp, outputs=outp)
	print(model.summary())

	from keras.optimizers import Adam
	adam = Adam(lr=0.001)

	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	model.fit(tr_vec, tr_ans,batch_size=512 ,epochs=2, validation_data=(val_vec, val_ans), verbose=True)

	print ('make predictions ...')
	#clf_predictions = clf.predict_proba(te_vec)
	preds = model.predict(te_vec, batch_size=1024)
	pred_test_y = (preds>0.35).astype(int)
	return pred_test_y

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
	attention = Permute([2, 1])(attention)

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

	model.fit(tr_vec, tr_ans,batch_size=512 ,epochs=2, validation_data=(val_vec, val_ans), verbose=True)

	print ('make predictions ...')
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

	model.fit(tr_vec, tr_ans,batch_size=512 ,epochs=2, validation_data=(val_vec, val_ans), verbose=True)

	print ('make predictions ...')
	#clf_predictions = clf.predict_proba(te_vec)
	preds = model.predict(te_vec, batch_size=1024)
	pred_test_y = (preds>0.35).astype(int)
	return pred_test_y

def SVM(tr_vec, tr_ans, val_vec, val_ans, te_vec):
	from sklearn.svm import SVC
	import matplotlib.pyplot as plt
	import numpy as np

	score = []
	for i in range(1,10):
		print(i)
		clf = SVC(gamma='auto', C=i)
		clf.fit(tr_vec, tr_ans)
		score.append(clf.score(val_vec, val_ans))
	np.asarray(score)

	fig, ax = plt.subplots( )
	line = ax.plot([i for i in range(1,10)], score)
	ax.legend()
	plt.show()
	

	clf = SVC(gamma='auto', C=np.argmax(score)+1)
	clf.fit(tr_vec, tr_ans)
	print ('make predictions ...')
	#clf_predictions = clf.predict_proba(te_vec)
	preds = clf.predict(te_vec)
	pred_test_y = (preds>0.35).astype(int)
	return pred_test_y



	# C_range_new = [0.001,0.01,0.1,1,10,100]
	# gamma_range_new = [0.01, 0.1, 1, 10, 100]
	# param_grid = dict(gamma=gamma_range_new, C=C_range_new)
	
	# from sklearn.metrics import fbeta_score, make_scorer
	# from sklearn.model_selection import GridSearchCV
	# grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
	# grid.fit(val_vec, val_ans)

	# scores = grid.cv_results_['mean_test_score'].reshape(len(C_range_new),
	#                                                      len(gamma_range_new))

	# import matplotlib.pyplot as plt
	# import numpy as np
	# plt.figure(figsize=(8, 6))
	# plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
	# plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
	# plt.xlabel('gamma')
	# plt.ylabel('C')
	# plt.colorbar()
	# plt.xticks(np.arange(len(gamma_range_new)), gamma_range_new, rotation=45)
	# plt.yticks(np.arange(len(C_range_new)), C_range_new)
	# plt.title('Validation accuracy')
	# plt.show()
def bernoulliNB(tr_vec, tr_ans, val_vec, val_ans, te_vec):
	from sklearn.naive_bayes import BernoulliNB
	clf = BernoulliNB(alpha=1.0)
	clf.fit(tr_vec, tr_ans)
	print(clf.score(val_vec, val_ans))

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


def model(modelname, vocab_size, sentence_len, embedding_size, tr_vec, tr_ans, val_vec, val_ans ,te_vec ):
	# Building an LSTM model
	print("Building Model")
	if modelname == "MultinomialNB":
		predict = multinomialNB(tr_vec, tr_ans, val_vec, val_ans ,te_vec)
		return predict
	if modelname == "ComplementNB":
		predict = complementNB(tr_vec, tr_ans, val_vec, val_ans ,te_vec)
		return predict
	if modelname == "BernoulliNB":
		predict = bernoulliNB(tr_vec, tr_ans, val_vec, val_ans ,te_vec)
		return predict
	if modelname == "SELFAttentive":
		predict = selfattentiveModel(vocab_size, sentence_len, embedding_size, tr_vec, tr_ans, val_vec, val_ans ,te_vec)
		return predict
	if modelname=="LSTMAttention":
		predict = lstmattentionModel(vocab_size, sentence_len, embedding_size, tr_vec, tr_ans, val_vec, val_ans ,te_vec)
		return predict

	if modelname=="LSTM":
		predict = lstmModel(vocab_size, sentence_len, embedding_size, tr_vec, tr_ans, val_vec, val_ans ,te_vec)
		return predict

	if modelname=="SVM":
		predict = SVM(tr_vec, tr_ans, val_vec, val_ans, te_vec)
		return predict
