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


def scale_inverse(scaler, train_scaled, test_scaled):
    train_unscaled = (pd.DataFrame(scaler.inverse_transform(train_scaled), 
                    columns=train_scaled.columns.values)
                    .set_index([train_scaled.index.values]))
    test_unscaled = (pd.DataFrame(scaler.inverse_transform(test_scaled), 
                    columns=test_scaled.columns.values)
                   .set_index([test_scaled.index.values]))
    return train_unscaled, test_unscaled


def telco_scale_inverse(scaler, train_scaled, test_scaled):
    train_scaled.set_index('customer_id', inplace=True)
    test_scaled.set_index('customer_id', inplace=True)
    train_unscaled = (pd.DataFrame(scaler.inverse_transform(train_scaled), 
                    columns=train_scaled.columns.values)
                    .set_index([train_scaled.index.values]))
    test_unscaled = (pd.DataFrame(scaler.inverse_transform(test_scaled), 
                    columns=test_scaled.columns.values)
                   .set_index([test_scaled.index.values]))
    train_unscaled.reset_index(inplace=True)
    train_unscaled.rename(columns={'index': 'customer_id'}, inplace=True)
    test_unscaled.reset_index(inplace=True)
    test_unscaled.rename(columns={'index': 'customer_id'}, inplace=True)
    return train_unscaled, test_unscaled


def uniform_scaler(df):
    train, test = split_my_data(df)
    scaler = (QuantileTransformer(n_quantiles=100, 
                                  output_distribution='uniform', 
                                  random_state=123, copy=True)
                                  .fit(train))
    train_scaled = (pd.DataFrame(scaler.transform(train), 
                                 columns=train.columns.values)
                                .set_index([train.index.values]))
    test_scaled = (pd.DataFrame(scaler.transform(test), 
                                columns=test.columns.values)
                               .set_index([test.index.values]))
    return scaler, train_scaled, test_scaled


def gaussian_scaler(df):
    train, test = split_my_data(df)
    scaler = (PowerTransformer(method='yeo-johnson', 
                               standardize=False, 
                               copy=True)
                              .fit(train))
    train_scaled = (pd.DataFrame(scaler.transform(train), 
                                 columns=train.columns.values)
                                .set_index([train.index.values]))
    test_scaled = (pd.DataFrame(scaler.transform(test), 
                                columns=test.columns.values)
                               .set_index([test.index.values]))
    return scaler, train_scaled, test_scaled


def min_max_scaler(df):
    train, test = split_my_data(df)
    scaler = (MinMaxScaler(copy=True, 
                           feature_range=(0,1))
                          .fit(train))
    train_scaled = (pd.DataFrame(scaler.transform(train), 
                                 columns=train.columns.values)
                                .set_index([train.index.values]))
    test_scaled = (pd.DataFrame(scaler.transform(test), 
                                columns=test.columns.values)
                               .set_index([test.index.values]))
    return scaler, train_scaled, test_scaled


def iqr_robust_scaler(df):
    train, test = split_my_data(df)
    scaler = (RobustScaler(quantile_range=(25.0,75.0), 
                           copy=True, 
                           with_centering=True, 
                           with_scaling=True)
                          .fit(train))
    train_scaled = (pd.DataFrame(scaler.transform(train), 
                                 columns=train.columns.values)
                                .set_index([train.index.values]))
    test_scaled = (pd.DataFrame(scaler.transform(test), 
                                columns=test.columns.values)
                               .set_index([test.index.values]))
    return scaler, train_scaled, test_scaled