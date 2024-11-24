import streamlit as st
import pandas as pd
import plotly.graph_objs as go

from data_import import filter_data
from data_analysis import prepare_data
from model_training import train_model
from predict_future import predict_future

file_path = 'data/raw/life_expectancy.csv'


def get_all_countries(file_path):
    df = pd.read_csv(file_path)
    df['Country'] = df['Country'].str.strip()
    countries = df['Country'].unique().tolist()
    print(f"this are the countries {countries}")
    return countries


def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal weight'
    else:
        return 'Overweight'


def apply_bmi_colors(df):
    df['Category'] = df['yhat'].apply(categorize_bmi)
    return df


countries = get_all_countries(file_path)

st.title('Future Life Expectancy and BMI Prediction')

selected_country = st.selectbox('Select a Country', ['All Countries'] + countries)
data_column = st.selectbox('Select data to predict', ['Life expectancy', 'BMI'])

cleaned_data = filter_data(file_path, countries)

if selected_country != 'All Countries':
    country_data = cleaned_data[cleaned_data['Country'] == selected_country]
    if not country_data.empty:
        df_prophet = prepare_data(country_data, data_column)
        model = train_model(df_prophet)
        future_data = predict_future(model, 2024, 2055)
        st.write(f'{data_column} Prediction for {selected_country}')
    else:
        st.warning(f"No data available for {selected_country}")
else:
    df_prophet = prepare_data(cleaned_data, data_column)
    model = train_model(df_prophet)
    future_data = predict_future(model, 2024, 2055)
    st.write(f'{data_column} Prediction for All Countries')

fig = go.Figure()

if data_column == 'BMI':
    future_data = apply_bmi_colors(future_data)

    for category, color in [('Underweight', 'yellow'), ('Normal weight', 'green'), ('Overweight', 'red')]:
        category_data = future_data[future_data['Category'] == category]
        fig.add_trace(go.Scatter(x=category_data['ds'], y=category_data['yhat'],
                                 mode='lines', name=category, line=dict(color=color, width=4)))

    # Plotting transitions
    for i in range(1, len(future_data)):
        if future_data['Category'].iloc[i] != future_data['Category'].iloc[i - 1]:
            prev = future_data.iloc[i - 1]
            curr = future_data.iloc[i]
            fig.add_trace(go.Scatter(x=[prev['ds'], curr['ds']], y=[prev['yhat'], curr['yhat']],
                                     mode='lines', line=dict(color='black', width=2), showlegend=False,
                                     hoverinfo='skip'))

else:
    fig.add_trace(go.Scatter(x=future_data['ds'], y=future_data['yhat'],
                             mode='lines', name=data_column, line=dict(color='blue', width=4)))

fig.update_layout(
    title=f'{data_column} Prediction for {selected_country if selected_country != "All Countries" else "All Countries"}',
    xaxis_title='Year',
    yaxis_title=data_column,
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(255, 255, 255, 1)',
    title_font=dict(size=20, family='Arial', color='black'),
    xaxis=dict(
        title='Year',
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black'
        ),
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        tickfont=dict(
            family='Arial, sans-serif',
            size=14,
            color='black'
        )
    ),
    yaxis=dict(
        title=data_column,
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black'
        ),
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        tickfont=dict(
            family='Arial, sans-serif',
            size=14,
            color='black'
        )
    ),
    legend=dict(
        font=dict(
            family='Arial, sans-serif',
            size=14,
            color='black'
        )
    ),
    margin=dict(l=40, r=40, t=40, b=40),
    showlegend=True,
)

# Adding a border effect
border_color = 'gray'
border_width = 2

fig.add_shape(type="rect",
              xref="paper", yref="paper",
              x0=0, y0=0, x1=1, y1=1,
              line=dict(color=border_color, width=border_width))

st.plotly_chart(fig)