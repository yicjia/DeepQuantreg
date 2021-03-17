#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:47:49 2020

@author: jiayichen
"""


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

from .utils import weighted_loss
        

def hyper_tuning(train_df,layers,nodes,dropout,activation, optimizer, bsize, n_epochs, n_cv=5, n_jobs=1, tau=0.5):
    X_train = train_df["X"]
    Y_train = train_df["Y"]
    #E_train = train_df["E"]
    W_train = train_df["W"]
    
    def create_model(layers,nodes,activation,optimizer,dropout):
        model = Sequential()
        for i in range(layers):
            if i==0:
                model.add(Dense(nodes, input_dim = X_train.shape[1]))
                model.add(Activation(activation))
                model.add(Dropout(dropout))
            else:
                model.add(Dense(nodes, activation = activation)) 
                model.add(Activation(activation))
                model.add(Dropout(dropout))
    
        model.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='linear'))
        model.compile(optimizer=optimizer,loss=weighted_loss(W_train, tau),metrics=['mse'])
        return model

    model = KerasRegressor(build_fn = create_model,verbose = 0)
    param_grid = dict(layers=layers, nodes=nodes, activation = activation, optimizer = optimizer, dropout=dropout, batch_size = bsize, epochs=n_epochs)
    grid = GridSearchCV(estimator=model,param_grid=param_grid,cv = 2, n_jobs=1)
    grid_result = grid.fit(X_train,np.log(Y_train))
    #print(grid_result.best_score_,grid_result.best_params_)
    return(grid_result.best_params_)




