# Quora Insincere Questions Classification

Detect toxic content to improve online conversations

### Run the code

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


# Results
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-0lax"></th>
    <th class="tg-0lax"></th>
    <th class="tg-0lax"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-fn5d">Model</td>
    <td class="tg-0lax">Setup</td>
    <td class="tg-0lax">Validation Accuracy</td>
    <td class="tg-0lax">Testing Accuracy</td>
  </tr>
  <tr>
    <td class="tg-fn5d">TF-IDF with Naive Bayes</td>
    <td class="tg-0lax">no processing data</td>
    <td class="tg-0lax">93.20%</td>
    <td class="tg-0lax">52.2%</td>
  </tr>
  <tr>
    <td class="tg-fn5d">TF-IDF with Naive Bayes</td>
    <td class="tg-0lax">remove stop words, punctuation</td>
    <td class="tg-0lax">94.24%</td>
    <td class="tg-0lax">54.0%</td>
  </tr>
  <tr>
    <td class="tg-0pky">WordEmbedding with LSTM</td>
    <td class="tg-0lax">remove stop words, punctuation</td>
    <td class="tg-0lax">95.35%</td>
    <td class="tg-0lax">62.1%</td>
  </tr>
  <tr>
    <td class="tg-0pky">Word Embedding with LSTM and attentive structure</td>
    <td class="tg-0lax">remove stop words,<br>punctuation</td>
    <td class="tg-0lax">95.45%</td>
    <td class="tg-0lax">61.20%</td>
  </tr>
</tbody>
</table>


### Notice
1. Output results are not the same to the writing parts because those are run on the Kaggle kernel which is better efficient than my computer.
2. In addition, I did not do any test locally. I submitted all the results on the Kaggle platform to get the real testing results.



