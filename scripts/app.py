import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_best_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "walmart_features.csv")
# Load model
xgb_best_model = joblib.load(MODEL_PATH)

# Load dataset
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

# Sidebar inputs
st.sidebar.header("Forecast Settings")
n_weeks = st.sidebar.slider("Weeks to Forecast", 1, 12, 4)

# Last known values
last_row = df.iloc[-1]
last_7_sales = df['weekly_sales'].tail(7).tolist()
last_date = df['date'].iloc[-1]

# Prepare future dataframe
future_dates = [last_date + pd.DateOffset(weeks=i) for i in range(1, n_weeks + 1)]
preds = []

current_sales = last_7_sales.copy()

for i in range(n_weeks):
    X_future = pd.DataFrame({
        'store': [last_row['store']],
        'holiday_flag': [0],
        'temperature': [last_row['temperature']],
        'fuel_price': [last_row['fuel_price']],
        'cpi': [last_row['cpi']],
        'unemployment': [last_row['unemployment']],
        'lag_1': [current_sales[-1]],
        'lag_2': [current_sales[-2]],
        'lag_3': [current_sales[-3]],
        'lag_4': [current_sales[-4]],
        'lag_5': [current_sales[-5]],
        'lag_6': [current_sales[-6]],
        'lag_7': [current_sales[-7]],
        'rolling_mean_4': [pd.Series(current_sales[-4:]).mean()],
        'rolling_std_4': [pd.Series(current_sales[-4:]).std()],
        'rolling_mean_12': [pd.Series(current_sales[-7:]).mean()],
        'year': [future_dates[i].year],
        'month': [future_dates[i].month],
        'weekofyear': [future_dates[i].isocalendar().week],
        'dayofweek': [future_dates[i].dayofweek],
        'holiday_lag1': [0],
        'temperature_rolling4': [df['temperature'].tail(4).mean()],
        'fuel_price_rolling4': [df['fuel_price'].tail(4).mean()],
        'cpi_rolling4': [df['cpi'].tail(4).mean()],
        'unemployment_rolling4': [df['unemployment'].tail(4).mean()],
    })

    forecast = xgb_best_model.predict(X_future)[0]
    preds.append(forecast)
    current_sales.append(forecast)

# Plot results
st.title("Walmart Sales Forecasting")
st.write(f"Forecasting next {n_weeks} weeks")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['date'].tail(20), df['weekly_sales'].tail(20), label="Historical")
ax.plot(future_dates, preds, marker='o', label="Forecast", color="red")
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Weekly Sales")
st.pyplot(fig)

st.write("Forecasted Values:")
st.write(pd.DataFrame({"Date": future_dates, "Predicted Sales": preds}))

# Create forecast DataFrame
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": preds
})

# Show table
st.write("Forecasted Values:")
st.write(forecast_df)

# Download button
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Forecasts as CSV",
    data=csv,
    file_name="walmart_forecast.csv",
    mime="text/csv",
)
