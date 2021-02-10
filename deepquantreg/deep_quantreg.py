#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:37:57 2021

@author: jiayichen
"""

import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout

import pandas as pd

from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter

from sklearn import preprocessing
import scipy.stats


   
def get_weights(time, delta):
    kmf = KaplanMeierFitter()
    kmf.fit(durations = time, event_observed = 1-delta, timeline=time)
    km = np.array(kmf.survival_function_.KM_estimate)
    km[km == 0] = 0.005
    w = np.array(delta/km)
    return w


def get_group_weights(time, delta, trt):
    n = np.shape(time)[0]
    w = np.zeros((n, ))
    w1 = get_weights(time[trt==0], delta[trt==0])
    w2 = get_weights(time[trt==1], delta[trt==1])
    w[trt==0] = w1
    w[trt==1] = w2
    return w



def huber(y_true, y_pred, eps=0.001):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < eps

  squared_loss = tf.keras.backend.square(error) / (2*eps)
  linear_loss  = tf.keras.backend.abs(error) - 0.5 * eps

  return tf.where(cond, squared_loss, linear_loss)

    
    
def weighted_loss(weights,tau, eps=0.001):
    def loss(y_true, y_pred):
        e = huber(y_true, y_pred)
        #e = y_true - tf.math.exp(y_pred)
        e = weights*e
        return K.mean(K.maximum(tau*e,(tau-1)*e))   
    return loss



def get_mse(obsT, predT, delta):
    temp = np.multiply(delta, (obsT-predT)**2)
    mse = np.sum(temp)/np.sum(delta)
    return mse


def get_ci(obsT, predT, delta):
    ci = concordance_index(obsT,predT,delta)
    return ci

def get_ql(obsT, predT, delta, tau, u):
    t = np.minimum(obsT,u)
    e = t - predT
    temp = np.maximum(tau*e,(tau-1)*e)
    ql = np.mean(temp)
    return ql


def predict_with_uncertainty(model, testdata, ci=0.95,n_iter=100):
    func = K.function(model.inputs + [K.learning_phase()], model.outputs)

    result = []

    for i in range(n_iter):
        result.append(func([testdata] + [1])[0])
        
    result = np.array(result)
    predmean = result.mean(axis=0).reshape(-1,)
    predsd = result.std(axis=0).reshape(-1,)
    lowerCI = predmean - scipy.stats.norm.ppf(1-0.5*(1-ci))*predsd
    upperCI = predmean + scipy.stats.norm.ppf(1-0.5*(1-ci))*predsd
    return np.exp(predmean), np.exp(lowerCI), np.exp(upperCI)



def organize_data (df,time="OT",event="ind",trt=None):
    E=np.array(df[event])
    Y=np.array(df[time])
    X=np.array(df.drop([event, time], axis = 1))
    X=X.astype('float64')
    scaler=preprocessing.StandardScaler().fit(X) #standardize
    X=scaler.transform(X)
    if trt == None:
        W = get_weights(Y, E)
    
    if  trt != None:
        trt=np.array(df[trt])
        W = get_group_weights(Y, E, trt)
        
    return {
        'Y' : Y,
        'E' : E,
        'X' : X,
        'W' : W
    }


class output:
    def __init__(self, predQ, lower, upper, ci, mse):
        self.predQ = predQ
        self.lower = lower
        self.upper = upper
        self.ci = ci
        self.mse = mse
        
        
    

def deep_quantreg(train_df,test_df,layer=2,node=300,n_epoch=100,bsize=64,acfn="sigmoid",opt="Adam",uncertainty=True,dropout=0.2,tau=0.5,verbose = 0):
    X_train = train_df["X"]
    Y_train = train_df["Y"]
    E_train = train_df["E"]
    W_train = train_df["W"]
    n1 = np.shape(Y_train)[0]
    
    X_test = test_df["X"]
    Y_test = test_df["Y"]
    E_test = test_df["E"]
    n2 = np.shape(Y_test)[0]
    
    
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
    ci = concordance_index(Y_train,Qpred,E_train)
    mse = get_mse(Y_train,Qpred,E_train)
    
    if uncertainty==False: 
        Qpred2 = np.exp(model.predict(X_test))
        Qpred2 = np.reshape(Qpred2, n2)
        lowerCI, upperCI = None, None
    else:
        Qpred2, lowerCI, upperCI = predict_with_uncertainty(model,X_test)

    
    ci2 = concordance_index(Y_test,Qpred2,E_test)
    mse2 = get_mse(Y_test,Qpred2,E_test)
        
    print( 'Concordance Index for training dataset:', ci)
    print( 'MSE for training dataset:', mse)
    print( 'Concordance Index for test dataset:', ci2)
    print( 'MSE for test dataset:', mse2)
    
    o = output(Qpred2, lowerCI, upperCI, ci2, mse2)


    return o
        
