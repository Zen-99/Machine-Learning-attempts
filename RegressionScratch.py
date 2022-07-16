# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 08:08:22 2022

@author: roshan
"""

import pandas as pd
import numpy as np

table = pd.read_csv('C:/Users/roshan/OneDrive/Desktop/Machine Learning Assignment/CASP.csv')

x=table.iloc[:,1:]
y=table.iloc[:,0]

from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY=train_test_split(x,y)

#print(trainX.columns)
x1=trainX["F1"]
x1=np.array(x1)
x2=trainX["F2"]
x2=np.array(x2)
x3=trainX["F3"]
x3=np.array(x3)
x4=trainX["F4"]
x4=np.array(x4)
x5=trainX["F5"]
x5=np.array(x5)
x6=trainX["F6"]
x6=np.array(x6)
x7=trainX["F7"]
x7=np.array(x7)
x8=trainX["F8"]
x8=np.array(x8)
x9=trainX["F9"]
x9=np.array(x9)

y=trainY
y=np.array(y)

n=len(x1)
#
XMat=np.ones((n,1))

x1New=np.reshape(x1,(n,1))
x2New=np.reshape(x2,(n,1))
x3New=np.reshape(x3,(n,1))
x4New=np.reshape(x4,(n,1))
x5New=np.reshape(x5,(n,1))
x6New=np.reshape(x6,(n,1))
x7New=np.reshape(x7,(n,1))
x8New=np.reshape(x8,(n,1))
x9New=np.reshape(x9,(n,1))

xNew=np.append(XMat,x1New,axis=1)
xNew=np.append(xNew,x2New,axis=1)
xNew=np.append(xNew,x3New,axis=1)
xNew=np.append(xNew,x4New,axis=1)
xNew=np.append(xNew,x5New,axis=1)
xNew=np.append(xNew,x6New,axis=1)
xNew=np.append(xNew,x7New,axis=1)
xNew=np.append(xNew,x8New,axis=1)
xNew=np.append(xNew,x9New,axis=1)



XT=np.transpose(xNew)

XTdotxNew=XT.dot(xNew)

inverse=np.linalg.inv(XTdotxNew)
mult=XT.dot(y)

theta=inverse.dot(mult)

beta0=theta[0]
beta1=theta[1]
beta2=theta[2]
beta3=theta[3]
beta4=theta[4]
beta5=theta[5]
beta6=theta[6]
beta7=theta[7]
beta8=theta[8]
beta9=theta[9]

def trainingAlgorithm(beta0,beta1,beta2,beta3,beta4,beta5,beta6,beta7,beta8,beta9,X1,X2,X3,X4,X5,X6,X7,X8,X9):
    predictedValue=beta0+X1*beta1+(X2*beta2)+(X3*beta3)+(X4*beta4)+(X5*beta5)+(X6*beta6)+(X7*beta7)+(X8*beta8)+(X9*beta9)
    return predictedValue

X1=testX["F1"]
X1=np.array(X1)
X2=testX["F2"]
X2=np.array(X2)
X3=testX["F3"]
X3=np.array(X3)
X4=testX["F4"]
X4=np.array(X4)
X5=testX["F5"]
X5=np.array(X5)
X6=testX["F6"]
X6=np.array(X6)
X7=testX["F7"]
X7=np.array(X7)
X8=testX["F8"]
X8=np.array(X8)
X9=testX["F9"]
X9=np.array(X9)

Y=testY
Y=np.array(Y)

prediction=np.zeros(len(X1))

i=0

while(i<len(X1)):
    prediction[i]=trainingAlgorithm(beta0,beta1,beta2,beta3,beta4,beta5,beta6,beta7,beta8,beta9,X1[i],X2[i],X3[i],X4[i],X5[i],X6[i],X7[i],X8[i],X9[i])
    i=i+1

 
from sklearn.metrics import mean_absolute_error 

error=mean_absolute_error(testY, prediction)  
print(error)