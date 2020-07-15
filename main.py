# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:53:36 2020

@author: Aswin
"""

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt 
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline

train = pd.read_csv("C:/Users/Aswin/CS224n/Twitter Sentiment Analysis/train.csv")
test = pd.read_csv("C:/Users/Aswin/CS224n/Twitter Sentiment Analysis/test.csv")

#print(train.shape, test.shape)

combined = train.append(test, ignore_index = True)
#removing the '@user' tags from the sentences
train['cleaned tweet'] = train['tweet'].apply(lambda x : ' '.join([word for word in x.split() if 
                                                                         not word.startswith('@')]))
test['cleaned tweet'] = test['tweet'].apply(lambda x : ' '.join([word for word in x.split() if 
                                                                         not word.startswith('@')]))

#collecting the hashtags from the neutral words (so as to see which hashtags correspond to non racist/sexist sentiments)
neutralWords = ' '.join([word for word in train['cleaned tweet'][train['label'] == 0]])
hashtags = [ht for ht in neutralWords.split() if ht.startswith('#')]
for i in range(len(hashtags)):
    hashtags[i] = hashtags[i][1:]

hashtag_freq = nltk.FreqDist(hashtags)
ht_df = pd.DataFrame({'Hashtag' : list(hashtag_freq.keys()), 'Count' : list(hashtag_freq.values())})

#Select top 20 most frequent hashtags and plot them   
most_frequent = ht_df.nlargest(columns = 'Count', n = 20) 
plt.figure(figsize = (16, 5))
ax = sns.barplot(data = most_frequent, x = 'Hashtag', y = 'Count')
ax.set(ylabel = 'Count')
plt.show()

#repeating the similar steps for racist/sexist words
negativeWords = ' '.join([word for word in train['cleaned tweet'][train['label'] == 1]])
hashtagsNeg = [ht for ht in negativeWords.split() if ht.startswith('#')]
for i in range(len(hashtagsNeg)):
    hashtagsNeg[i] = hashtagsNeg[i][1:]

hashtagNeg_freq = nltk.FreqDist(hashtagsNeg)
htNeg_df = pd.DataFrame({'Hashtag' : list(hashtagNeg_freq.keys()), 'Count' : list(hashtagNeg_freq.values())})

#Select top 20 most frequent words used and plot them
most_frequent = htNeg_df.nlargest(columns = 'Count', n = 20) 
plt.figure(figsize = (16, 5))
ax = sns.barplot(data = most_frequent, x = 'Hashtag', y = 'Count')
ax.set(ylabel = 'Count')
plt.show()

wc = WordCloud(width = 800, height = 500, max_font_size = 100).generate(neutralWords)
plt.figure(figsize = (12, 8))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.show()

wc = WordCloud(width = 800, height = 500, max_font_size = 100).generate(negativeWords)
plt.figure(figsize = (12, 8))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.show()

X_train, X_val, y_train, y_val = train_test_split(train['cleaned tweet'], train['label'], test_size = 0.25,
                                                      random_state = 42)
vect = CountVectorizer().fit(X_train)
X_train_vect = vect.transform(X_train)

#Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vect, y_train)
pred_nb = nb.predict(vect.transform(X_val))
print('F1 Score of Naive Bayes Model: ', f1_score(y_val, pred_nb))

#Logistic Regression
lr = LogisticRegression(C = 50)
lr.fit(X_train_vect, y_train)
pred_lr = lr.predict(vect.transform(X_val))
print('F1 score of Logistic Regression Model: ', f1_score(y_val, pred_lr))

#TF-IDF
vect = TfidfVectorizer().fit(X_train)
X_train_vect = vect.transform(X_train)
lr.fit(X_train_vect, y_train)
pred_tfidf = lr.predict(vect.transform(X_val))
print('F1 score of TF-IDF model: ', f1_score(y_val, pred_tfidf))

#Continuous Bag of Words
vect = CountVectorizer(max_df = 0.9, min_df = 1).fit(X_train)
X_train_vect = vect.transform(X_train)
lr.fit(X_train_vect, y_train)
pred_bow = lr.predict(vect.transform(X_val))
print('F1 score of Bag of Words model: ', f1_score(y_val, pred_bow))


#HyperParameter Tuning
'''
We get best parameters as C = 50 and min_df = 1
pipe = make_pipeline(CountVectorizer(), LogisticRegression())
param_grid = {"logisticregression__C": [0.01, 0.1, 1, 10, 50, 100],
              "countvectorizer__min_df": [1,2,3]}
grid = GridSearchCV(pipe, param_grid, cv = 5, scoring = 'f1', n_jobs = -1)
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
'''

#Building our final model
vect = TfidfVectorizer().fit(X_train)
X_train_vect = vect.transform(X_train)

model = LogisticRegression(C = 50)
model.fit(X_train_vect, y_train)
pred = model.predict(vect.transform(X_val))
print('F1: ', f1_score(y_val, pred))

X_test = test['cleaned tweet']
test_pred = model.predict_proba(vect.transform(X_test))
preds = np.where(test_pred[:, 1] > 0.35, 1, 0)
res = pd.DataFrame(data = {'id' : test['id'], 'label' : preds})
res.to_csv('finalmodel.csv', index = False)