import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm
from math import sqrt

import env 
import wrangle_zillow
import split_scale_zillow
import features_zillow


def modeling_function(x_train,y_train):
    predictions=pd.DataFrame({'actual':y_train.home_value}).reset_index(drop=True)
    
    #model 1
    lm1=LinearRegression()
    lm1.fit(x_train,y_train)
    lm1_predictions=lm1.predict(x_train)
    predictions['lm1']=lm1_predictions

    #baseline model
    predictions['baseline'] = y_train.home_value.mean()
    
    return predictions


def plot_residuals(x, y):
    '''
    Plots the residuals of a model that uses x to predict y. Note that we don't
    need to make any predictions ourselves here, seaborn will create the model
    and predictions for us under the hood with the `residplot` function.
    '''
    return sns.residplot(x, y)

