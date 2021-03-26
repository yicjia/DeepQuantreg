# DeepQuantreg

DeepQuantreg implements a deep neural network to the quantile regression for survival data with right censoring, which is adjusted by the inverse of the estimated censoring distribution in the check function.

DeepQuantreg shows that the deep learning method could be flexible enough to predict nonlinear patterns more accurately compared to the traditional quantile regression, emphasizing practicality of the method for censored survival data. 

Reference: Jia, Y., & Jeong, J. H. (2020). Deep Learning for Quantile Regression: DeepQuantreg. arXiv preprint arXiv:2007.07056.
- Paper: https://arxiv.org/abs/2007.07056


## Installation:

### From source

Download a local copy of DeepQuantreg and install from the directory:

	git clone https://github.com/yicjia/DeepQuantreg.git
	cd DeepQuantreg
	pip install .

### Dependencies

Tensorflow, Keras, lifelines, sklearn, scipy, and all of their respective dependencies. 

## Example

First, open Python and import the pacakge:

    from DeepQuantreg import utils as utils
    from DeepQuantreg import deep_quantreg as dq
    import pandas as pd

Then, read in the datasets and organize them into DeepQuantreg form. 

    train_dataset_fp = "./data/traindata.csv"
    train_df = pd.read_csv(train_dataset_fp)
    train_df = utils.organize_data(train_df,time="OT",event="ind",trt="x2")

    test_dataset_fp = "./data/testdata.csv"
    test_df = pd.read_csv(test_dataset_fp)
    test_df = utils.organize_data(test_df,time="OT",event="ind",trt="x2")


DeepQuantreg can be trained and predict using the following code: 

    result = dq.deep_quantreg(train_df,test_df,layer=2,node=300,n_epoch=200,bsize=64,tau=0.5)


You can get the predicted quantiles and its prediction interval by calling:
    
    result.predQ
    result.lower
    result.upper
    
You can get the prediction performance by calling:
    
    result.ci
    result.mse
    result.ql

  

The figure shows the prediction from DeepQuantreg:
![PIplot](https://user-images.githubusercontent.com/58962571/108008760-19669a00-6fcf-11eb-803b-478652281d77.PNG)

## Hyperparameter tuning

It is recommanded to run hyperparameter tuning before training your model. In DeepQuantreg, grid search method is used for hyperparameter tuning. Specify your search space, for example:

    layers = [2,3]
    nodes=[100,300]
    dropout=[0.2,0.4]
    activation = ["sigmoid","relu"]
    optimizer = ["adam","nadam"]
    bsize = [64,128]
    n_epochs = [50,100,200]
    
    from DeepQuantreg import hyperparameter_tuning as tune
    tune.hyper_tuning(train_df,layers,nodes,dropout,activation, optimizer, bsize, n_epochs, n_cv=5, tau=0.5)


##

## Function: deep_quantreg

### Usage
deep_quantreg(train_df,test_df,layer=2,node=300,n_epoch=50,bsize=64,acfn="sigmoid",opt="Adam",uncertainty=True,dropout=0.2,tau=0.5,verbose=0)

### Parameters
* *train_df* :	the training dataset after organize into DeepQuantreg form.
* *test_df* :	the test dataset after organize into DeepQuantreg form.
* *layer* :	the number of hidden layers, the defualt is 2. 
* *node* :	the number of hidden nodes for each layer, the defualt is 300. 
* *n_epoch* :	the number of epochs, the defualt is 100. 
* *bsize* :	the batch size, the default is 64.
* *acfn* :	the activation function, the default is sigmoid.
* *opt* :	the optimizor, the default is Adam
* *uncertainty* :	whether to get prediction uncertainty, the default is True. dropout layer must be enable if choose to get prediction uncertainty.
* *dropout* :	the dropout rate, the default is 0.2
* *tau* :	the quantiles, the default is the median (0.5).

### Attributes
* *predQ* :	the predicted conditional quantiles.
* *lower* :	the lower bound of the 95 percent prediction interval
* *upper* :	the upper bound of the 95 percent prediction interval
* *ci* :	the C-index between the predicted quantiles and the observed event time.
* *mse* :	the MSE between the predicted quantiles and the observed event time over event observations.


## Function: hyper_tuning

### Usage
hyper_tuning(train_df,layers,nodes,dropout,activation, optimizer, bsize, n_epochs, n_cv=5, n_jobs=1, tau=0.5)

### Parameters
* *train_df* :	the training dataset after organize into DeepQuantreg form.
* *layers* :	the search space for number of hidden layers.
* *nodes* :	the search space for the number of hidden nodes for each layer. 
* *dropout* :	the search space for the dropout rate.
* *activation* :	the search space for the activation function.
* *optimizer* :	the search space for the optimizor
* *bsize* :	the search space for the batch size.
* *n_epoch* :	the search space for the number of epochs. 
* *n_cv* :	number of fold cross validation, the default is 5. 
* *n_jobs* :	number of jobs to run in parallel, the default is 1.
* *tau* :	the quantiles, the default is the median (0.5).

### Returns
hyper_tuning returns a dictionary of the best estimator that was chosen by the search



## Function: organize_data

### Usage
organize_data(df,time,event,trt)

### Parameters
* *df* :	the input dataset. Should contain a time column, a event indicator column and columns of covaraites 
* *time* :	the follow-up time
* *event* :	the event indicator
* *trt* :	the treatment group if you want to compute different KM estimators for different groups. The default is None.

### Returns
organize_data returns a dictionary containing "Y": the follow-up time, "E": the event indicator, "X": the covariates matrix, and "W": IPCW weights

