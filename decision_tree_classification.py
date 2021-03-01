#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 22:01:12 2021

@author: dev__7
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data=pd.read_csv('/home/dev__7/work/Machine Learning Codes and Datasets/udemy/Part 3 - Classification/Section 19 - Decision Tree Classification/Python/Social_Network_Ads.csv')

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

# print(classifier.predict(sc.transform([[52,150000]])))

y_pred=classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
