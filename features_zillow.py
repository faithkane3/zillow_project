import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandarScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regresssion
import statsmodel.api as sm 
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection impofrt RFE 

def select_kbest_freg(x, y, k):
    """Removes all but highest scoring features
    Takes:
          k - int: number of features
          x - df of features
          y - df of target
    Returns:
          list of column names of highest scoring features
    """
    f_selector = SelectKBest(f_regression, k).fit(x, y)
    f_support = f_selector.get_support()
    f_feature = x.loc[:,f_support].columns.tolist()
    return f_feature

def ols_backware_elimination(x, y):
    """Removes all but highest scoring features
    Takes:
          x - df of features
          y - df of target
    Returns:
          list of column names of highest scoring features
    """
    cols = list(x_train.columns)
    while (len(cols) > 0):
        p = []
        x_1 = x[cols]
        x_1 = sm.add_constant(x_1)
        model = sm.OLS(y, x_1).fit()
        p = model.pvalues
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax > 0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    return cols 


def lasso_cs_coef(x, y):
    """
    Takes:
          df of features
          df of target
    Returns:
          coefficients for each feature
          plots the features with their weights
    """
    reg = LassoCV().fit(x, y)
    coef = pd.Series(reg.coef_, index = x.columns)
    imp_coef = coef.sort_values()
    plot = imp_coef.plot(kind = "barh")
    plt.show()
    return coef

    
def optimumal_number_of_features(x, y):   
    """
    Takes:
          x_train: Pandas df
          y_train: Pandas df
    Returns:
          int: number of optimum features
    """ 
    features_range = range(1, len(x.columns)+1)
    high_score = 0
    number_of_features = 0
    for n in features_range):
        model = LinearRegression()
        train_rfe = RFE(model, n).fit_transform(x, y)
        model.fit(train_rfe, y)
        score = model.score(train_rfe, y)
        if(score > high_score):
            high_Score = score
            number_of_features = n
    return number_of_features, score