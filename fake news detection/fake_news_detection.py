# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:16:11 2020

@author: Apoorv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv("news.csv")
dataset.dropna(axis = 0)


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset["label"] = labelencoder.fit_transform(dataset["label"])

labels = {0:"Fake",1:"Real"}
Y = dataset["label"]

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
x1 = []
x2 = []
for i in range(0,6335):
    review = re.sub("[^a-zA-Z]"," ",dataset["title"][i])
    review = (review.lower()).split()
    ps = PorterStemmer()
    review = [ps.stem(words) for words in review if words not in set(stopwords.words("english"))]
    review = " ".join(review)
    x1.append(review)
    review1 = re.sub("[^a-zA-Z]"," ",dataset["text"][i])
    review1 = (review1.lower()).split()
    review1 = [ps.stem(words) for words in review1 if words not in set(stopwords.words("english"))]
    review1 = " ".join(review1)
    x2.append(review1)
    
from sklearn.feature_extraction.text import CountVectorizer
cv1 = CountVectorizer()
X1 = cv1.fit_transform(x1).toarray()
cv2 = CountVectorizer()
X2 = cv2.fit_transform(x2).toarray()

X1_word_sum = []
for i in range(0,6884):
    X1_word_sum.append(X1[:,i].sum())
    
from sklearn.model_selection import train_test_split
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X1,Y,test_size = 0.2,random_state = 0)
X2_train,X2_test,Y2_train,Y2_test = train_test_split(X2,Y,test_size = 0.2,random_state = 0)
    

from sklearn.svm import SVC
classifier1 = SVC(C = 1,kernel = 'rbf')
classifier1.fit(X1_train,Y1_train)
Y1_pred = classifier1.predict(X1_test)

classifier2 = SVC(C = 1,kernel = 'linear')
classifier2.fit(X2_train,Y2_train)
Y2_pred = classifier2.predict(X2_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y1_test,Y1_pred)

cm2 = confusion_matrix(Y2_test,Y2_pred)



