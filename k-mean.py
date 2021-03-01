#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:17:50 2020

@author: demon__7
"""

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()
import numpy as np 
from sklearn.datasets.samples_generator import make_blobs
X,y_true=make_blobs(n_samples=300,centers=4,cluster_std=0.60,random_state=0)
plt.scatter(X[:,0],X[:,1],s=50)

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans=kmeans.predict(X)

from sklearn.metrics import pairwise_distances_argmin
def find_clusters(X, n_clusters, rseed=2):
    rng=np.random.RandomState(rseed)
    i=rng.permutation(X.shape[0])[:n_clusters]
    centers=X[i]
    while True:
        labels=pairwise_distances_argmin(X, centers)
        new_centers=np.array([X[labels==i].mean(0) 
                             for i in range(n_clusters)])
        centers, labels=find_clusters(X,4)
        plt.scatter(X[:,0], X[:,1],c=labels,
                    s=50,cmap='viridis')
        if np.all(centers==new_centers):
            break
        centers=new_centers
    return centers, labels
    centers,labels=find_clusters(X,4)
    plt.scatter(X[:,0], X[:,1],c=y_kmeans,s=50,cmap='viridis')
    plt.scatter(centers[:,0],centers[:,1],c='black',s=200,alpha=0.5);

from sklearn.datasets import load_sample_image
india=load_sample_image("flower.jpg")
