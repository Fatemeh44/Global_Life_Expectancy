import pandas as pd

def prepare_data(df, column_name):
    df_prophet = df[['Year', column_name]].rename(columns={'Year': 'ds', column_name: 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
    return df_prophet