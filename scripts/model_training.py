from prophet import Prophet
import pandas as pd

def train_model(df):
    df_prophet = df[['Year', 'Life expectancy']].rename(columns={'Year': 'ds', 'Life expectancy': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    return model