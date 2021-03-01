#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:04:35 2020

@author: dev__7
"""

import pandas as pd 
import matplotlib.pyplot  as plt
import numpy as np

dataset=pd.read_csv('/home/demon/project/Machine Learning Codes and Datasets/Machine Learning A-Z (Model Selection)/Regression/Data.csv')

def decisionTree():
    x=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,-1].values
    # y=y.values.reshape1(len(y),1)
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    #traning dataset on decision tree
    from sklearn.tree import DecisionTreeRegressor
    regressor=DecisionTreeRegressor()
    regressor.fit(x_train,y_train)

    # predict=regressor.predict([[6.5]])
    # predict=predict.reshape((len(predict),1))
    # print(predict)
    
    
    #prediction
    
    y_pred=regressor.predict(x_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1))
    
    return y_test,y_pred
    # x_grid=np.arange(min(x),max(x),0.1)
    # x_grid=x_grid.reshape((len(x_grid),1))
    
    # plt.scatter(x,y, color='red')
    # plt.plot(x_grid,regressor.predict(x_grid),color='green')
    # # plt.plot(x_grid,predict,color='blue')
    # plt.title(label='decision_tree_regression')
    # plt.xlabel('position_label')
    # plt.ylabel('salary')
    # plt.show('True')
    
    

def multipleLinearRegression():
    x=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,-1].values


    # from sklearn.compose import ColumnTransformer
    # from sklearn.preprocessing import OneHotEncoder
    # ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
    # x=np.array(ct.fit_transform(x))
    
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    
    from sklearn.linear_model import LinearRegression
    regressor=LinearRegression()
    regressor.fit(x_train,y_train)
    
    y_pred=regressor.predict(x_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1))
    
    return y_test,y_pred
        
    # plt.scatter(x_train,y_train,color='red')
    # plt.plot(x_train,y_pred,color='green')
    # plt.title('profit')
    # plt.xlabel('')
    # plt.ylabel('')
        
    #backward elimination
        
    # import statsmodels.formula.api as sm
    # import statsmodels.api as sm1
    # x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
    # x_opt=x[:,[0,1,2,3,4,5,6]]
    # regressor_OLS=sm1.OLS(x,x_opt).fit()
    # regressor_OLS.summary()
    
    
def polynomialRegression():
    x=dataset.iloc[:,:-1]
    y=dataset.iloc[:,-1]

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    
    #traning the linear regression model on the whole dataset
    from sklearn.linear_model import LinearRegression
    # lin_reg=LinearRegression()
    # lin_reg.fit(x, y)
    
    #traning the polynomialregression model on the whole dataset
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg=PolynomialFeatures(degree=4)
    x_poly=poly_reg.fit_transform(x_train)
    lin_reg=LinearRegression()
    lin_reg.fit(x_poly,y_train)
    
    # #visualising the linear_reg
    # plt.scatter(x,y,color='red')
    # plt.plot(x,lin_reg.predict(x),color='blue')
    # plt.title('lin_reg_model')
    # plt.xlabel('position_label')
    # plt.ylabel('salary')
    # plt.show
    
    # #visualising the polynomial_reg
    # plt.scatter(x, y, color='red')
    # plt.plot(x,lin_reg_2.predict(x_poly_reg)  ,color='green')
    # plt.title(label='poly_reg_model')
    # plt.xlabel('position_label')
    # plt.ylabel('salary')
    # plt.show

    #predicting new result with LinearRegression
    # l=lin_reg.predict([[6.5]])
    
    #predicting new result with polnomialyRegression
    # p=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
    
    #predicting values
    y_pred=lin_reg.predict(poly_reg.transform(x_test))
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.values.reshape(len(y_test),1)),axis=1))
    return y_test,y_pred


def randonForestRegression():
    x=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,-1].values

    from sklearn.ensemble import RandomForestRegressor
    regressor=RandomForestRegressor(n_estimators=10)
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    
    regressor.fit(x_train,y_train)
    
    #prediction
    y_pred=regressor.predict(x_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_pred),1)),axis=1))
    
    return y_pred,y_test
    # p=regressor.predict([[6.5]])

    # x_grid=np.arange(min(x),max(x),0.1)
    # x_grid=x_grid.reshape((len(x_grid),1))
    
    # plt.scatter(x,y, color='red')
    # plt.plot(x_grid,regressor.predict(x_grid),color='green')
    # # plt.plot(x_grid,predict,color='blue')
    # plt.title(label='decision_tree_regression')
    # plt.xlabel('position_label')
    # plt.ylabel('salary')
    # plt.show('True')    
    

def svr():
    x=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,-1].values
    y=y.reshape(len(y),1)
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    
    from sklearn.preprocessing import StandardScaler 
    sc_x=StandardScaler()  
    sc_y=StandardScaler()
    x_train=sc_x.fit_transform(x_train)
    y_train=sc_y.fit_transform(y_train)


    from sklearn.svm import SVR
    regressor=SVR(kernel='rbf')
    regressor.fit(x_train,y_train)
    
    #transform predction on scaler of x
    # tf_x=sc_x.transform([[6.5]])
    # use tf_x transformed predction value on base of x scaler in prediction
    # p_x=regressor.predict(tf_x)
    # then we use that predicted value of p_sc_x into inverse transformation on sc_y
    # sc_y.inverse_transform(p_x)
    
    # sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
    
    # plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
    # plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x)),color='green')
    # plt.title(label='svr')
    # plt.xlabel('position_label')
    # plt.ylabel('salary')
    # plt.show('False')
        
    # #visualising the svr (on higher resolution)
    # x_grid=np.arange(min(sc_x.inverse_transform(x)),max(sc_x.inverse_transform(x)),0.1)
    # x_grid=x_grid.reshape((len(x_grid),1))
    # plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
    # plt.plot(x_grid,sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))),color='green')
    # plt.title(label='svr')
    # plt.xlabel('position_label')
    # plt.ylabel('salary')
    # plt.show('True')
    
    #predction
    y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(x_test)))
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
    return y_test,y_pred
    
def evluatingModelResult(y_test,y_pred):
    from sklearn.metrics import r2_score
    print(r2_score(y_test, y_pred))
    

    
# y_test,y_pred=multipleLinearRegression()
# y_test,y_pred=svr()
# y_test,y_pred=polynomialRegression()
y_test,y_pred=randonForestRegression()
# y_test,y_pred=decisionTree()

evluatingModelResult(y_test,y_pred)