# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:57:01 2022

@author: roshan
"""

import pandas as pd

table = pd.read_csv('C:/Users/roshan/OneDrive/Desktop/Machine Learning Assignment/abalone.csv')

#print(table.head())
#print(table.head())
sexMapping= {"I": 0, "M": 1,"F": 2}

table['Sex']=table['Sex'].map(sexMapping)
 
#print(table.head())
x=table.iloc[:, :8]
y=table.iloc[:,8]
print(y)
#print(y)
from sklearn.model_selection import train_test_split

trainX,testX,trainY,testY=train_test_split(x,y)


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn import tree
classifier=tree.DecisionTreeClassifier()
classifier.fit(trainX,trainY)

predictedVal=classifier.predict(testX)
from sklearn.metrics import accuracy_score
print(accuracy_score(testY,predictedVal))

