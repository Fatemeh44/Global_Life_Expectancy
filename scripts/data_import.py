import pandas as pd

file_path = 'C:/Users/FATI/PycharmProjects/Life-Expectancy-of-5-EU/data/raw/life_expectancy.csv'
countries = ['Switzerland', 'Italy', 'France', 'Germany', 'United Kingdom of Great Britain and Northern Ireland']

def filter_data(file_path, countries):
    df = pd.read_csv(file_path)
    df['Country'] = df['Country'].str.strip()
    df.fillna(0, inplace=True)
    filtered_df = df[df['Country'].isin(countries)]
    return filtered_df

filtered_data = filter_data(file_path, countries)
print("Countries in the filtered data:", filtered_data['Country'].unique())
print(filtered_data)