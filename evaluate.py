import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from math import sqrt
import matplotlib.pyplot as plt

def plot_residuals(x, y, df):
    """
    Takes: 
           x, feature string
           y, target string
           df
    Returns: 
           plot of the residuals from linear regression
           of y on x
    """
    sns.residplot(x, y, df)
    plt.title('Residuals')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


def regression_errors(y, yhat):
    """
    Takes: 
          target variable y, Pandas Series
          yhat, calculated regression Pandas Series
    Returns:
          SSE, ESS, TSS, MSE, RMSE
    """
    from math import sqrt
    SSE = ((yhat - y) ** 2).sum()
    MSE = SSE / len(y)
    ESS = sum((yhat - y.mean())**2)
    TSS = SSE + ESS
    RMSE = sqrt(MSE)
    print(f'SSE: {round(SSE, 3)}')
    print(f'ESS: {round(ESS, 3)}')
    print(f'TSS: {round(TSS, 3)}')
    print(f'MSE: {round(MSE, 3)}')
    print(f'RMSE: {round(RMSE,3)}')
    return SSE, ESS, TSS, MSE, RMSE


def baseline_mean_errors(y):
    """
    Takes: 
          target variable y, Pandas Series
    Returns:
          SSE, MSE, RMSE for baseline values
    """
    bSSE = ((y.mean() - y) ** 2).sum()
    bMSE = bSSE / len(y)
    bRMSE = bMSE ** .5
    print(f'baseline SSE: {round(bSSE, 3)}')
    print(f'baseline MSE: {round(bMSE, 3)}')
    print(f'baseline RMSE: {round(bRMSE,3)}')
    return bSSE, bMSE, bRMSE


def better_than_baseline(y, yhat):
    """
    Takes: 
          y variable, Pandas Series
          yhat variable, Pandas Series
    Returns:
          boolean value evaluating statement
              - model performs better
              than baseline
    """
    SSE = ((yhat - y) ** 2).sum()
    bSSE = ((y.mean() - y) ** 2).sum()
    return SSE < bSSE


def model_significance(ols_model):
    """Takes:
             ols model
       Returns:
             r2
             p_value
    """
    p_value = ols_model.f_pvalue
    r2 = ols_model.rsquared
    print(f'The independent variable explains {r2:.2%} of the variance in our dependent variable.')
    print(f'Our p_value is {round(p_value,4)}')
    return r2, p_value