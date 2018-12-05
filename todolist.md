#### Baseline Model
Naive Bayes
- no processing data and use INT to represent words(val: 0.8216)
- no processing data and tf-idf(unigram) with BernoulliNB (val acc:0.9320, test acc:0.522)
- no processing data and tf-idf(bigram) with BernoulliNB (val acc: 0.9430, test acc:0.270)

- remove stop words and punciation and use INT to represent words (val acc:0.838815)
- remove stop words and punciation and tf(val acc:0.9424, test acc:0.520) or tf-idf (val acc:0.9424, test acc:0.540) with BernoulliNB
- remove stop words and punciation and tf-idf(bigram) with BernoulliNB (val acc: 0.9398 test acc:0.058)

- word2vec encoding BernoulliNB (**not yet**)
can change SVM to NB(SVM too slow)
- ComplementNB (val acc: 0.8924, test acc: 0.41)
sklearn.naive_bayes.ComplementNB ( It is particularly suited for imbalanced data sets)(???)
update: bigram(**bad**)

#### Deep learning
- LSTM with tokenize (**ok**) 
- LSTM with pre-train embedding(**not yet**)
- LSTM with attention (**ok**) 0.626
- LSTM with self attentive (**ok**)
- Graph LSTM weights

#### Graph
- training accuracy and validation accuracy
- imbalance data (**ok**)
- seqence length (**ok**)

#### Modular
- seperate program into data processing, features selection, and Model (**ok**)
- Save to the pre-trained model 

