#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:13:31 2020

@author: dev__7
"""

import numpy as np
import pandas as pd 
import matplotlib as plt

data=pd.read_csv('/home/demon/study/Machine Learning Codes and Datasets/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()

x_train=scale.fit_transform(x_train)
x_test=scale.fit_transform(x_test)

 

from sklearn.linear_model import LogisticRegression

LogisticRegression=LogisticRegression(random_state=0)

LogisticRegression.fit(x_train, y_train)

# prediction=LogisticRegression.predict(scale.transform([[30,87000]]))

y_pred=LogisticRegression.predict(x_test)
# print(y_pred)
# print(y_test)

# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# score=LogisticRegression.score(x_test, y_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test, y_pred)
print(cm)

a=accuracy_score(y_test, y_pred)
print(a)


#visualising result

from matplotlib.colors import ListedColormap

x_set,y_set=scale.inverse_transform(x_train),y_train

