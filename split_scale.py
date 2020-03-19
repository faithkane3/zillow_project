import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

def split_my_data(df, train_pct=0.70, seed=123):
    train, test = train_test_split(df, train_size=train_pct, random_state=seed)
    return train, test

def standard_scaler(df):
    """
    Takes in df with numeric values only
    Returns scaler, train_scaled, test_scaled df
    """
    train, test = split_my_data(df)
    scaler = StandardScaler().fit(train)
    train_scaled = (pd.DataFrame(scaler.transform(train), 
                    columns=train.columns.values)
                    .set_index([train.index.values]))
    test_scaled = (pd.DataFrame(scaler.transform(test), 
                    columns=test.columns.values)
                   .set_index([test.index.values]))
    return scaler, train_scaled, test_scaled

def telco_standard_scaler(df):
    """
    Takes in telco df
    Returns df with numeric columns scaled
    """
    df.set_index('customer_id', inplace=True)
    train, test = split_my_data(df)
    scaler = StandardScaler().fit(train)
    train_scaled = (pd.DataFrame(scaler.transform(train), 
                    columns=train.columns.values)
                    .set_index([train.index.values]))
    test_scaled = (pd.DataFrame(scaler.transform(test), 
                    columns=test.columns.values)
                   .set_index([test.index.values]))
    train_scaled.reset_index(inplace=True)
    train_scaled.rename(columns={'index': 'customer_id'}, inplace=True)
    test_scaled.reset_index(inplace=True)
    test_scaled.rename(columns={'index': 'customer_id'}, inplace=True)
    return scaler, train_scaled, test_scaled

