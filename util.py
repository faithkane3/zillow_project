from env import host, user, password

# function to contact database
def get_db_url(db_name):
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"

# function strips spaces, replaces them with NANs, and converts them to a float
def space_to_float(df, column):
    df.column = df.total_charges.str.strip().replace(" ",np.nan).astype(float)
    return df

# function to clean up some dirty columns
def wrangle_df(df, col):
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df[col] = df[col].astype('float')
    df = df.dropna()
    return df

# function to return a df using a sql query to database
def read_sql_query(db, query, url):
    url = get_db_url(db)
    df = pd.read_sql(query, url)
    return df

# function to summarize df
def tell_me_about(df):
    print("DataFrame Shape:\n")
    print(df.shape)
    print("\nInfo about:\n")
    print(df.info())
    print("\nDescribe:\n")
    print(df.describe())
    print("\nPreview:\n")
    print(df.head())