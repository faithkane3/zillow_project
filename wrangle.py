import pandas as pd
import numpy as np

from env import host, user, password

def get_db_url(db_name):
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"


def get_data_from_sql():
    query = """
    SELECT customer_id, monthly_charges, tenure, total_charges
    FROM customers
    WHERE contract_type_id = 3;
    """
    df = pd.read_sql(query, get_db_url('telco_churn'))
    return df

def wrangle_telco():
    df = get_data_from_sql()
    df.tenure.replace(0, 1, inplace=True)
    df.total_charges.replace(' ', df.monthly_charges, inplace=True)
    df.total_charges = df.total_charges.astype(float)
    return df