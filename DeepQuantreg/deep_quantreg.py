#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:57:38 2020

@author: jiayichen
"""


import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import eager_learning_phase_scope
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import scipy.stats

from .utils import weighted_loss, get_ci, get_mse, get_ql

class output:
    def __init__(self, predQ, lower, upper, ci, mse, ql):
        self.predQ = predQ
        self.lower = lower
        self.upper = upper
        self.ci = ci
        self.mse = mse
        self.ql = ql

  
def predict_with_uncertainty(model, testdata, ci=0.95, n_iter=100):
    func = K.function([model.input] , [model.output])
    with eager_learning_phase_scope(value=1):
        result = []
        for i in range(n_iter):
            print
            result.append(func([testdata]))
            
        result = np.array(result)
        predmean = result.mean(axis=0).reshape(-1,)
        predsd = result.std(axis=0).reshape(-1,)
        lowerCI = predmean - scipy.stats.norm.ppf(1-0.5*(1-ci))*predsd
        upperCI = predmean + scipy.stats.norm.ppf(1-0.5*(1-ci))*predsd
        return np.exp(predmean), np.exp(lowerCI), np.exp(upperCI)

    
def deep_quantreg(train_df,test_df,layer=2,node=300,n_epoch=100,bsize=64,acfn="sigmoid",opt="Adam",uncertainty=True,dropout=0.2,tau=0.5,verbose = 0):
    X_train = train_df["X"]
    Y_train = train_df["Y"]
    E_train = train_df["E"]
    W_train = train_df["W"]
    n1 = np.shape(Y_train)[0]
    u1 = np.max(Y_train[E_train==0])
    
    X_test = test_df["X"]
    Y_test = test_df["Y"]
    E_test = test_df["E"]
    n2 = np.shape(Y_test)[0]
    u2 = np.max(Y_train[E_train==0])
    
    
    model = Sequential()
    for i in range(layer):
        if i==0:
            model.add(Dense(node, input_dim = X_train.shape[1], activation = acfn)) # Hidden 1
            model.add(Dropout(dropout))
        else:
           model.add(Dense(node, activation = acfn)) # Hidden 2 
           model.add(Dropout(dropout))
           
    model.add(Dense(1, activation = 'linear')) # Output
    model.compile(loss = weighted_loss(W_train, tau), metrics=['mse'],optimizer = opt)
    model.fit(X_train,np.log(Y_train),verbose = verbose, epochs = n_epoch, batch_size = bsize)

    Qpred = np.exp(model.predict(X_train))
    Qpred = np.reshape(Qpred, n1)
    ci = get_ci(Y_train,Qpred,E_train)
    mse = get_mse(np.log(Y_train),np.log(Qpred),E_train)
    ql = get_ql(np.log(Y_train),np.log(Qpred),E_train,np.log(u1))
    
    if uncertainty==False: 
        Qpred2 = np.exp(model.predict(X_test))
        Qpred2 = np.reshape(Qpred2, n2)
        lowerCI, upperCI = None, None
    else:
        Qpred2, lowerCI, upperCI = predict_with_uncertainty(model,X_test)

    
    ci2 = get_ci(Y_test,Qpred2,E_test)
    mse2 = get_mse(np.log(Y_test),np.log(Qpred2),E_test)
    ql2 = get_ql(np.log(Y_test),np.log(Qpred2),E_test,np.log(u2))
        
    print( 'Concordance Index for training dataset:', ci)
    print( 'Concordance Index for test dataset:', ci2)
    
    o = output(Qpred2, lowerCI, upperCI, ci2, mse2, ql2)


    return o
        



