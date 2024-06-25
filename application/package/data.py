import pandas as pd
from application.package.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - removing observations with missing features
    - removing potential duplicate observations
    - set columns to the correct dtype
    """
    # Remove missing information
    df = df.dropna(how='any', axis=0)

    # Remove duplicate observations
    df = df.drop_duplicates()

    # Setting the correct dtypes
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

    print("✅ data cleaned")

    return df
