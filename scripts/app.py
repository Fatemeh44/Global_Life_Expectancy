import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

from data_import import filter_data
from data_analysis import prepare_data
from model_training import train_model
from predict_future import predict_future

file_path = 'C:/Users/FATI/PycharmProjects/Life-Expectancy-of-5-EU/data/raw/life_expectancy.csv'

def get_all_countries(file_path):
    df = pd.read_csv(file_path)
    df['Country'] = df['Country'].str.strip()
    countries = df['Country'].unique().tolist()
    return countries

def train_model(df):
    """Train a Prophet model for time series forecasting."""
    df_prophet = df[['Year', 'Life expectancy']].rename(columns={'Year': 'ds', 'Life expectancy': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    return model

def predict_future(model, start_year, end_year):
    future_years = pd.DataFrame({'ds': pd.date_range(f'{start_year}-01-01', f'{end_year}-01-01', freq='Y')})
    forecast = model.predict(future_years)
    return forecast[['ds', 'yhat']].rename(columns={'ds': 'Year', 'yhat': 'Life expectancy'})

countries = get_all_countries(file_path)

st.title('Future Life Expectancy Prediction')

selected_country = st.selectbox('Select a Country', ['All Countries'] + countries)

cleaned_data = filter_data(file_path, countries)

if selected_country != 'All Countries':
    country_data = cleaned_data[cleaned_data['Country'] == selected_country]
    if not country_data.empty:
        model = train_model(country_data)
        future_data = predict_future(model, 2024, 2055)
        st.write(f'Life Expectancy Prediction for {selected_country}')
    else:
        st.warning(f"No data available for {selected_country}")
else:
    model = train_model(cleaned_data)
    future_data = predict_future(model, 2024, 2055)
    st.write('Life Expectancy Prediction for All Countries')

fig = px.line(future_data, x='Year', y='Life expectancy',
              title=f'Life Expectancy Prediction for {selected_country if selected_country != "All Countries" else "All Countries"}')
fig.update_layout(yaxis=dict(range=[55, 125]), xaxis_title='Year', yaxis_title='Life Expectancy')

st.plotly_chart(fig)