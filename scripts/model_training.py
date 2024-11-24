from prophet import Prophet

def train_model(df_prophet):
    model = Prophet()
    model.fit(df_prophet)
    return model