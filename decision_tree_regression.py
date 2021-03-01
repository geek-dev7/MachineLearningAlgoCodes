#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 02:37:51 2020

@author: dev__7
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('/home/demon/project/Machine Learning Codes and Datasets/Part 2 - Regression/Section 8 - Decision Tree Regression/Python/Position_Salaries.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
# y=y.values.reshape(len(y),1)

#traning dataset on decision tree
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor()
regressor.fit(x,y)

predict=regressor.predict([[6.5]])
# predict=predict.reshape((len(predict),1))
# print(predict)

x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))

plt.scatter(x,y, color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='green')
# plt.plot(x_grid,predict,color='blue')
plt.title(label='decision_tree_regression')
plt.xlabel('position_label')
plt.ylabel('salary')
plt.show('True')