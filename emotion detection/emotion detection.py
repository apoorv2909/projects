# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:20:38 2020

@author: Apoorv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel("data.xlsx",header = None)
dataset = dataset[0].str.split("<",expand = True)
dataset = dataset[1].str.split(">",expand = True)

x = dataset[1]
y = dataset[0]

emotion = ["anger","disgust","fear","happy","sad","shame","surprise"]

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
corpus = []
for i in range(0,len(x)):
    review = re.sub('[^a-zA-Z]'," ",x[i])
    review = review.lower()
    review = review.split()
    stemmer = SnowballStemmer("english",ignore_stopwords = True)
    review = [stemmer.stem(word) for word in review if word not in set(stopwords.words("english"))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
row = cv.get_feature_names()


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


from sklearn.svm import SVC
classifier = SVC(C = 2,kernel = "linear",random_state = 0)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_pred,Y_test)

from sklearn.model_selection import GridSearchCV
parameters = {'C':[1,5],'kernel':["linear"]}
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = "accuracy",n_jobs = -1)
grid_search.fit(X_train,Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

def accuracy():
    acc = 0
    for i in range(0,7):
        acc = acc + cm[i,i]/319
    return(acc)
accuracy()

def find_emotion(line):
    ln = re.sub('[^a-zA-Z]',' ',line)
    ln = ln.lower()
    ln = ln.split()
    ln_ps = SnowballStemmer("english",ignore_stopwords = True)
    ln = [ln_ps.stem(word) for word in ln if word not in set(stopwords.words("english"))]
    ln = " ".join(ln)
    ln = [ln]
    ln_X = cv.transform(ln).toarray()
    ln_pred = classifier.predict(ln_X)
    ln_emotion = emotion[ln_pred[0]]
    return(ln_emotion)
    
find_emotion("you look upset")
