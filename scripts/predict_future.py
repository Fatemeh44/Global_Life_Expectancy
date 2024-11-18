from prophet import Prophet
import pandas as pd

def predict_future(model, start_year, end_year):
    future_years = pd.DataFrame({'ds': pd.date_range(f'{start_year}-01-01', f'{end_year}-01-01', freq='Y')})
    forecast = model.predict(future_years)
    return forecast[['ds', 'yhat']].rename(columns={'ds': 'Year', 'yhat': 'Life expectancy'})