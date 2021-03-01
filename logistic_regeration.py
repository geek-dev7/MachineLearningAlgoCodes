#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:40:07 2020

@author: demon__7
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

digits=load_digits()

print("image_data_shape", digits.data.shape)
print("label_data_shape", digits.target.shape)

plt.figure(figsize=(20,4))
for index, (image,label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1,5,index + 1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('traning: %i\n'%label, fontsize=20) 
    
x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.27,random_state=100)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression

logic_reg=LogisticRegression()
logic_reg.fit(x_train,y_train)

# print(logic_reg.predict(x_test[0].reshape(1,-1)))

print(logic_reg.predict((x_test[0:10])))
prediction=logic_reg.predict(x_test)
score=logic_reg.score(x_test, y_test)
print(score)




index=0
classifiedindex=[]
for predict,actual in zip(prediction,y_test):
    if predict==actual:
        classifiedindex.append(index)
    index+=1
plt.figure(figsize=(20,3))

for plotindex,wrong in enumerate(classifiedindex[0:4]):
    plt.subplot(1,4,plotindex+1)
    plt.imshow(np.reshape(x_test[wrong],(8,8)),cmap=plt.cm.gray)
    plt.title("predicted:{},actual:{}".format(prediction[wrong],y_test[wrong]),fontsize=20 )
    
    