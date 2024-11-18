def prepare_data(df):
    X = df[['Year']]
    Y = df[['Life expectancy']]
    return X,Y