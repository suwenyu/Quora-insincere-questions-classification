#### Introduction to Machine Learning HW5
##### Wen-Yuh Su    UIN:671937912

- Environment: 
```
> python3 --version
```
3.6.5

- Install require packages: 
```
> pip3 install -r requirements.txt
```

- Dataset
put under the all directory
```
> ./all/trainsmall.csv
> ./all/testsmall.csv
```

- Run the main.py
```
> python3 main.py [feature_selection_method] [model_name]
```

Feature_selection_method:
1. tokenize
2. tfidf_tokenize

Models:
1. BernoulliNB
2. LSTM (can only use tokenize)
3. SELFAttentive (can only use tokenize)
For exmaple,
```
> python3 main.py tokenize BernoulliNB
```
It will print out the validation accuracy.


#####Results
1. Output results are not the same to the writing parts because those are run on the Kaggle kernel which is better efficient than my computer.
2. In addition, I did not do any test locally. I submitted all the results on the Kaggle platform to get the real testing results.



