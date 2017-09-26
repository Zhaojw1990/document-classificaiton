#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:51:27 2017

@author: zhao
"""
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import f1_score
import numpy as np


    
    
#==============================================================================
# def bestSvm(X,y):    
#     
#     
#     clf = svm.SVC()
#     kf = KFold(n_splits=2, random_state=None, shuffle=False)
#     for train_index, test_index in kf.split(X):
#         print("TRAIN:", train_index, "TEST:", test_index)
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         print y_train
#         clf.fit(X_train,y_train)
#         cc = clf.score(X_test,y_test)
#         print cc
# 
#==============================================================================




if __name__ == "__main__":
    
    neg_raw = np.load("vectors_neg_500.npy")
    pos_raw = np.load("vectors_pos_500.npy")

    tag = []
    neg = []
    pos = []

    for i in neg_raw:
        neg.append(i[:500])
        tag.append(i[-1:])

    for i in pos_raw:
        neg.append(i[:500])
        tag.append(i[-1:])

    X = neg+pos
    Y = tag
    clf = svm.SVC().fit(X,Y)
    cc = clf.score(X,Y)
    print cc