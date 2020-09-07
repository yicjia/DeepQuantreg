# DeepQuantreg

DeepQuantreg implements a deep neural network to the quantile regression for survival data with right censoring, which is adjusted by the inverse of the estimated censoring distribution in the check function.

DeepQuantreg shows that the deep learning method could be flexible enough to predict nonlinear patterns more accurately compared to the traditional quantile regression even in low-dimensional data, emphasizing on practicality of the method for censored survival data. 

Paper: https://arxiv.org/abs/2007.07056


## Installation:

### From source

Download a local copy of DeepQuantreg and install from the directory:

	git clone https://github.com/yicjia/DeepQuantreg.git
	cd DeepQuantreg
	pip install .

### Dependencies

Tensorflow, Keras, lifelines, sklearn, and all of their respective dependencies. 

## Example

First, open Python and import the pacakge:

    from DeepQuantreg import deep_quantreg as dq
    import pandas as pd

Then, read in the datasets and organize them into DeepQuantreg form. 

    train_dataset_fp = "./data/traindata.csv"
    train_df = pd.read_csv(train_dataset_fp)
    train_df = dq.organize_data(train_df,time="OT",event="ind",trt="x2")

    test_dataset_fp = "./data/testdata.csv"
    test_df = pd.read_csv(test_dataset_fp)
    test_df = dq.organize_data(test_df,time="OT",event="ind",trt="x2")


DeepQuantreg can be trained and predict using the following code: 

    model = dq.deep_quantreg(train_df,test_df,layer=2,node=300,n_epoch=50,bsize=64,acfn="sigmoid",opt="Adam",tau=0.5,verbose=0)

It prints out the C-index and Modified MSE. However, you can also get them using the code:
    
    ci = dq.get_ci(test_df["Y"],model["test_pred"],test_df["E"])
    MSE = dq.get_mse(test_df["Y"],model["test_pred"],test_df["E"])


## Function: organize_data

### Usage
organize_data(df,time="OT",event="ind",trt="x2")

### Arguments
* *df* :	the input dataset. Should contain a time column, a event indicator column and columns of covaraites 
* *time* :	the follow-up time
* *event* :	the event indicator
* *trt* :	the treatment group if you want to compute different KM estimators for different groups. The default is None.

### Values
organize_data returns a dictionary containing "Y": the follow-up time, "E": the event indicator, "X": the covariates matrix, and "W": IPCW weights


## Function: deep_quantreg

### Usage
deep_quantreg(train_df,test_df,layer=2,node=300,n_epoch=50,bsize=64,acfn="sigmoid",opt="Adam",tau=0.5,verbose=0)

### Arguments
* *train_df* :	the training dataset after organize into DeepQuantreg form.
* *test_df* :	the test dataset after organize into DeepQuantreg form.
* *layer* :	the number of hidden layers, the defualt is 2. Currently, DeepQuantreg only allows up to 3 hidden layers.
* *node* :	the number of hidden nodes for each layer, the defualt is 300. 
* *n_epoch* :	the number of epochs, the defualt is 100. 
* *bsize* :	the batch size, the default is 64.
* *acfn* :	the activation function, the default is sigmoid.
* *opt* :	the optimizor, the default is Adam
* *tau* :	the quantiles, the default is the median (0.5).

### Values
deep_quantreg returns a dictionary containing "train_pred": the predicted quantiles based on the training data, "test_pred": the predicted quantiles based on the test data.

## Function: get_ci and get_mse

### Usage
get_ci(obsT, predT, delta)

get_mse(obsT, predT, delta)

### Arguments
* *obsT* :	the follow-up time, "Y" from organize_data().
* *predT* :	the predicted quantile from deep_quantreg().
* *delta* :	the event indicator, "E" from organize_data().

### Values
get_ci returns the C-index between the predicted quantiles and the observed event time.

get_mse returns the MSE between the predicted quantiles and the observed event time.


