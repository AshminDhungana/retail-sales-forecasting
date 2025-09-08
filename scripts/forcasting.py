#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import joblib


# In[5]:


df = pd.read_csv("../data/walmart_features.csv")
last_row = df.iloc[-1]
last_7_sales = df['weekly_sales'].tail(7).tolist()

# Define the new date for forecasting. Let's assume the last date was 2012-10-26 and we want to forecast 2012-11-02.
last_date = pd.to_datetime(df['date'].iloc[-1])
next_date = last_date + pd.DateOffset(weeks=1)

# Create a new DataFrame for the future data point
X_future = pd.DataFrame({
    'store': [last_row['store']],
    'holiday_flag': [0], # Assume no holiday
    'temperature': [last_row['temperature']], # Use the last known temperature
    'fuel_price': [last_row['fuel_price']], # Use the last known fuel price
    'cpi': [last_row['cpi']],
    'unemployment': [last_row['unemployment']],

    # Lag features based on the last available weekly sales values
    'lag_1': [last_7_sales[-1]],
    'lag_2': [last_7_sales[-2]],
    'lag_3': [last_7_sales[-3]],
    'lag_4': [last_7_sales[-4]],
    'lag_5': [last_7_sales[-5]],
    'lag_6': [last_7_sales[-6]],
    'lag_7': [last_7_sales[-7]],

    # Rolling mean/std features
    'rolling_mean_4': [df['weekly_sales'].tail(4).mean()],
    'rolling_std_4': [df['weekly_sales'].tail(4).std()],
    'rolling_mean_12': [df['weekly_sales'].tail(12).mean()],

    # Date-related features for the next week
    'year': [next_date.year],
    'month': [next_date.month],
    'weekofyear': [next_date.weekofyear],
    'dayofweek': [next_date.dayofweek],

    # Lagged holiday and rolling features of other variables
    'holiday_lag1': [last_row['holiday_flag']],
    'temperature_rolling4': [df['temperature'].tail(4).mean()],
    'fuel_price_rolling4': [df['fuel_price'].tail(4).mean()],
    'cpi_rolling4': [df['cpi'].tail(4).mean()],
    'unemployment_rolling4': [df['unemployment'].tail(4).mean()],
})

print("Future Data Point (X_future):\n", X_future)


# In[6]:


# Load the trained model and make a prediction
xgb_best_model = joblib.load("../models/xgb_best_model.pkl")
forecast = xgb_best_model.predict(X_future)

print(f"\nForecast for the next week's sales: {forecast[0]:.2f}")

