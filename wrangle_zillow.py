import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from util import get_db_url

def get_data_from_sql():
    query = """
    SELECT bedroomcnt as bedrooms, 
       bathroomcnt as bathrooms,
       calculatedfinishedsquarefeet as square_feet,
       taxamount as taxes,
       taxvaluedollarcnt as home_value,
       propertylandusedesc, 
       fips as fips_number
    FROM predictions_2017
    JOIN properties_2017 USING(id)
    JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE (transactiondate >= '2017-05-01' AND transactiondate <= '2017-06-30') 
        AND propertylandusetypeid IN ('261', '279') 
        AND bedroomcnt > 0
        AND bathroomcnt > 0
        AND calculatedfinishedsquarefeet > 0 
        AND taxamount > 0
        AND taxvaluedollarcnt > 0
        AND fips > 0
    ORDER BY fips;
    """
    df = pd.read_sql(query, get_db_url('zillow'))
    return df


def clean_data(df):
    df = df.dropna()
    df["fips_number"] = df["fips_number"].astype(int)
    df["propertylandusedesc"] = df["propertylandusedesc"].astype("category")
    # df["bedroomcnt"] = df["bedroomcnt"].astype("int")
    # df["bathroomcnt"] = df["bathroomcnt"].astype("int")
    df["square_feet"] = df["square_feet"].astype("int")
    return df
       
def wrangle_zillow():
    df = get_data_from_sql()
    df = clean_data(df)
    return df 
