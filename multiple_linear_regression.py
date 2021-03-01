#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 04:37:37 2020

@author: dev__7
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('/home/demon/project/Machine Learning Codes and Datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(ct.fit_transform(x))


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1))


# plt.scatter(x_train,y_train,color='red')
# plt.plot(x_train,y_pred,color='green')
# plt.title('profit')
# plt.xlabel('')
# plt.ylabel('')

#backward elimination

import statsmodels.formula.api as sm
import statsmodels.api as sm1
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5,6]]
regressor_OLS=sm1.OLS(x,x_opt).fit()
regressor_OLS.summary()



