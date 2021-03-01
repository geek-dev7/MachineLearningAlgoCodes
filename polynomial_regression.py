#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:36:56 2020

@author: dev__7
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('/home/demon/project/Machine Learning Codes and Datasets/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
x=dataset.iloc[:,1:-1]
y=dataset.iloc[:,-1]

#traning the linear regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x, y)

#traning the polynomialregression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly_reg=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly_reg,y)

#visualising the linear_reg
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('lin_reg_model')
plt.xlabel('position_label')
plt.ylabel('salary')
plt.show

#visualising the polynomial_reg
plt.scatter(x, y, color='red')
plt.plot(x,lin_reg_2.predict(x_poly_reg)  ,color='green')
plt.title(label='poly_reg_model')
plt.xlabel('position_label')
plt.ylabel('salary')
plt.show

#predicting new result with LinearRegression
l=lin_reg.predict([[6.5]])

#predicting new result with polnomialyRegression
p=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))