#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 04:58:12 2020

@author: dev__7
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset=pd.read_csv('/home/demon/project/Machine Learning Codes and Datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv')

x=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1:2].values

#spliting dataset into traning_set and test_set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#traning the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
# print(regressor)

#predicting the test set result
# y_pred=regressor.predict(x_test)
x_pred=regressor.predict(x_train)

#visualising the traning set
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,x_pred,color='green')
plt.title('salary vs experience(tranning_set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#visualising the test set result
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,x_pred,color='blue')
plt.title('salary vs experience(test_set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
    

