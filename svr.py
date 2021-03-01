#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 23:43:02 2020

@author: dev__7
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('/home/demon/project/Machine Learning Codes and Datasets/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Python/Position_Salaries.csv')

x=dataset.iloc[:,1:-1]
y=dataset.iloc[:,-1]
y=y.values.reshape(len(y),1)

from sklearn.preprocessing import StandardScaler 
sc_x=StandardScaler()  
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)


from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)

#transform predction on scaler of x
tf_x=sc_x.transform([[6.5]])
# use tf_x transformed predction value on base of x scaler in prediction
p_x=regressor.predict(tf_x)
# then we use that predicted value of p_sc_x into inverse transformation on sc_y
sc_y.inverse_transform(p_x)

# sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x)),color='green')
plt.title(label='svr')
plt.xlabel('position_label')
plt.ylabel('salary')
plt.show('False')

#visualising the svr (on higher resolution)
x_grid=np.arange(min(sc_x.inverse_transform(x)),max(sc_x.inverse_transform(x)),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(x_grid,sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))),color='green')
plt.title(label='svr')
plt.xlabel('position_label')
plt.ylabel('salary')
plt.show('True')