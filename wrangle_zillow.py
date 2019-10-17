import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from util import get_db_url

def get_data_from_sql():
    query = """
    SELECT bedroomcnt, 
       bathroomcnt,
       calculatedfinishedsquarefeet,
       taxamount,
       taxvaluedollarcnt
       propertylandusedesc, 
       fips
    FROM predictions_2017
    JOIN properties_2017 USING(id)
    JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE (transactiondate >= '2017-05-01' AND transactiondate <= '2017-06-30') 
	    AND propertylandusetypeid = '261' 
	    OR (propertylandusetypeid = '279' AND propertylandusedesc='Single Family Residential')
    ORDER BY fips
    LIMIT 100;
    """
    df = pd.read_sql(query, get_db_url('zillow'))
    return df


def clean_data(df):
    df["fips_number"] = df.fips
    df = df.drop(columns=["fips"])
    df = df.dropna()
    df.fips_number = df.astype(int)
    df.propertylandusedesc = df.astype("category")
    df.bedroomcnt = df.astype("int")
    df.bathroomcnt = df.astype("int")
    return df
       
def wrangle_zillow():
    df = get_data_from_sql()
    df = clean_data(df)
    return df 
