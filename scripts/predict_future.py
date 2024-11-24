import pandas as pd
import numpy as np

def predict_future(model, start_year, end_year):
    future = model.make_future_dataframe(periods=(end_year - start_year + 1), freq='Y')
    forecast = model.predict(future)
    future_data = forecast[(forecast['ds'].dt.year >= start_year) & (forecast['ds'].dt.year <= end_year)]
    future_data = future_data[['ds', 'yhat']]
    return future_data