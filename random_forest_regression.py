#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:04:11 2020

@author: dev__7
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# dataset=pd.read_csv('/home/demon/project/Machine Learning Codes and Datasets/Part 2 - Regression/Section 9 - Random Forest Regression/Python/Position_Salaries.csv')
# x=dataset.iloc[:,1:-1].values
# y=dataset.iloc[:,-1].values

dataset=pd.read_csv('/home/demon/study/training/machine_learning_traning_implementation/dataset/for_random_forest.csv')
x=dataset.drop(['Survived'],axis=1)
y=dataset['Survived']

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10)

regressor.fit(x,y)

p=regressor.predict([[6.5]])

# x_grid=np.arange(min(x),max(x),0.1)
# x_grid=x_grid.reshape((len(x_grid),1))

# plt.scatter(x,y, color='red')
# plt.plot(x_grid,regressor.predict(x_grid),color='green')
# # plt.plot(x_grid,predict,color='blue')
# plt.title(label='decision_tree_regression')
# plt.xlabel('position_label')
# plt.ylabel('salary')
# plt.show('True')