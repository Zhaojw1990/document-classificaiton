#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:28:40 2017

@author: zhao
"""

import numpy as np
import math
from nltk import FreqDist
def loadfiles():
    np_gini = np.load("gini.npy")
    dic = {}
    for i in np_gini:
        tf_neg = float(i[0])
        tf_pos = float(i[1])
        tf = float(i[2])
        gini_score = float(i[3])
        if gini_score != 1:
            score = 1/(1-gini_score) + math.log(tf/20000)
        else:
            score = 1+ math.log10(tf/30000)
        dic[i[4]] = score,tf,tf_pos,tf_neg,gini_score

    dic_fs = sorted(dic.iteritems(),key = lambda x:x[1], reverse = True) 

    np_tokens = np.load("tf.npy")
    
    return dic,dic_fs[:3000],np_tokens

def feature_to_vector(dic_all_ftv,dic_ftv,tokens_ftv):
    vector_ftv = []
    for document in tokens_ftv:
        fdist = FreqDist(document)
        vec = []
        for words in dic_ftv:
            word = words[0]
            m = 0
            if fdist[word] != 0:
                m = dic_all_ftv[word][0]
            vec.append(m)
        vector_ftv.append(vec) 
            
            
            
            
        
    return vector_ftv
    
    

    
def feature_tagging(vec):
 
    neg = []
    pos = []
    par = len(vec)/2
    neg = vec[:par]
    pos = vec[-par:]
    for i in neg:
        i.append(0)
    for i in pos:
        i.append(1)
    return neg,pos

    
if __name__ == "__main__":
    
    dic_all,dic_sort,tpl = loadfiles()
    tokens = []
    for i in tpl:
        token = []
        for j in i:
            token.append(j[1])
        tokens.append(token)
        
    vectors = feature_to_vector(dic_all,dic_sort,tokens)
    np_vectors_neg,np_vectors_pos = feature_tagging(vectors)
    np.save("vectors_neg_500",np_vectors_neg)
    np.save("vectors_pos_500",np_vectors_pos)
