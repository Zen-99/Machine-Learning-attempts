# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:34:44 2022

@author: roshan
"""
import pandas as pd

table = pd.read_csv('C:/Users/roshan/OneDrive/Desktop/Machine Learning Assignment/CASP.csv')

#print(table.head()) 

#seperate inputs and outputs of the given data set.
x=table.iloc[:,1:]
y=table.iloc[:,0]

#print(x)

from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY=train_test_split(x,y)

from sklearn.linear_model import LinearRegression
#Here we create a model with train data
CASP_Model=LinearRegression().fit(trainX,trainY)

from sklearn.metrics import mean_absolute_error 
predicted_values = CASP_Model.predict(testX)
error=mean_absolute_error(testY, predicted_values)

print("Error is:",error,"%")

