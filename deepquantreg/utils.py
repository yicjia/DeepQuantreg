#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 21:23:40 2021

@author: jiayichen
"""
import tensorflow as tf
import numpy as np
from keras import backend as K

from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter

from sklearn import preprocessing


   
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
