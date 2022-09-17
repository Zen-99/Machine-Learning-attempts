# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BRg2zw9c1sb9iqZJOttH39Abtjpl9RSA
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

table = pd.read_csv('/content/drive/MyDrive/Machine Learning/car.csv')

print(table.head())

buyingMapping= {"vhigh": 3, "high": 2,"med": 1,"low":0}
table['Buying']=table['Buying'].map(buyingMapping)
maintMapping={"vhigh": 3, "high": 2,"med": 1,"low":0}
table['Maint']=table['Maint'].map(maintMapping)
lugMapping={"small":0,"med":1,"big":2}
table['Lug_root']=table['Lug_root'].map(lugMapping)
safetyMapping={"low":0,"med":1,"high":2}
table['Saftey']=table['Saftey'].map(safetyMapping)
doorsMapping={"2":2,"3":3,"4":4,"5more":5}
table['Doors']=table['Doors'].map(doorsMapping)
personMapping={"2":1,"4":2,"more":3}
table['Persons']=table['Persons'].map(personMapping)

table.head()

x=table.iloc[:, :6]
y=table.iloc[:,6]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=4,bootstrap=True,ccp_alpha=0.00)

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)

from sklearn import metrics

print("Accuracy of the model is:",metrics.accuracy_score(y_test, y_pred))

predicted=clf.predict([[1,2,3,1,1,1]])
print(predicted)

featureImportance = pd.Series(clf.feature_importances_,index=['Buying','Maint','Doors','Persons','Lug_root','Saftey']).sort_values(ascending=False)
featureImportance

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# Creating a bar plot
sns.color_palette("Paired")
sns.barplot(x=featureImportance, y=featureImportance.index)

# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Requirements')
plt.title("Important Features")
plt.legend()
plt.show()