#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:57:38 2020

@author: jiayichen
"""

import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
#import csv


from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter

from sklearn import preprocessing


def heythere():
    print("hey there")
   
def get_weights(time, delta):
    kmf = KaplanMeierFitter()
    kmf.fit(durations = time, event_observed = 1-delta)
    km = np.array(kmf.survival_function_.KM_estimate)
    n = np.shape(time)[0]
    nkm = np.shape(km)[0]
    if n != nkm:
        km = km[:-1]
    km[km == 0] = 0.005
    w = np.array(delta/(n*km))
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
        e = huber(y_true, tf.math.exp(y_pred))
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



    

def deep_quantreg(train_df,test_df,layer=2,node=300,n_epoch=100,bsize=64,acfn="sigmoid",opt="Adam",tau=0.5,verbose = 0):
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
    if layer == 1:
        model.add(Dense(node, input_dim = X_train.shape[1], activation = acfn)) # Hidden 1
        model.add(Dense(1, activation = 'linear')) # Output
    if layer == 2 :    
        model.add(Dense(node, input_dim = X_train.shape[1], activation = acfn)) # Hidden 1
        model.add(Dense(node, activation = acfn)) # Hidden 2 
        model.add(Dense(1, activation = 'linear')) # Output
    if layer == 3 :    
        model.add(Dense(node, input_dim = X_train.shape[1], activation = acfn)) # Hidden 1
        model.add(Dense(node, activation = acfn)) # Hidden 2 
        model.add(Dense(node, activation = acfn)) # Hidden 3
        model.add(Dense(1, activation = 'linear')) # Output
    
    model.compile(loss = weighted_loss(W_train, tau), metrics=['mse'],optimizer = opt)
    model.fit(X_train,Y_train,verbose = verbose, epochs = n_epoch, batch_size = bsize)

    Qpred = np.exp(model.predict(X_train))
    Qpred = np.reshape(Qpred, n1)
    ci = concordance_index(Y_train,Qpred,E_train)
    mse = get_mse(Y_train,Qpred,E_train)
    
    Qpred2 = np.exp(model.predict(X_test))
    Qpred2 = np.reshape(Qpred2, n2)
    ci2 = concordance_index(Y_test,Qpred2,E_test)
    mse2 = get_mse(Y_test,Qpred2,E_test)
        
    print( 'Concordance Index for training dataset:', ci)
    print( 'MSE for training dataset:', mse)
    print( 'Concordance Index for test dataset:', ci2)
    print( 'MSE for test dataset:', mse2)


    return {
        'train_pred' : Qpred,
        'test_pred' : Qpred2
    }



