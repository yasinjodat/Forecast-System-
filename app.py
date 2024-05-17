import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from models import Arima_model,Sarima_model,lstm_model,Hybrid_ann_arima

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns='Vol.',inplace=True)
    df['Date']=pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)
    df.sort_index(ascending=True,inplace=True)
    df['High']=df['High'].str.replace(',','').astype(float)
    df['Low']=df['Low'].str.replace(',','').astype(float)
    df['Open']=df['Open'].str.replace(',','').astype(float)
    df['Price']=df['Price'].str.replace(',','').astype(float)
    df['Change %']=df['Change %'].str.replace('%','').astype(float)
    df['Change %']=df['Change %']/100
    df.index.freq=pd.infer_freq(df.index)
    return df


# Example dataset paths (these should be replaced with actual paths)
datasets = {
    "Dataset 1": "C:/Users/hp/Documents/University/Semester 8/Data Mining Lab/Project/data/S&P 500 Historical Data.csv",
    "Dataset 2": "C:/Users/hp/Documents/University/Semester 8/Data Mining Lab/Project/data/AEP_hourly.csv"
}


st.title('Time Series Forecasting System')

st.sidebar.title("Select Dataset")
dataset_name = st.sidebar.selectbox("Dataset", list(datasets.keys()))
df = load_data(datasets[dataset_name])


st.sidebar.title("Model Selection")
model_type = st.sidebar.radio("Choose Model", ('ARIMA','SARIMA', 'LSTM', 'Hybrid ARIMA-ANN'))


st.write("### Dataset")
st.write(df.head())


st.line_chart(df)

# Parameters for ARIMA
if model_type in ['ARIMA','Hybrid ARIMA-ANN']:
    p = st.sidebar.number_input('p (AR order)', min_value=0, value=1)
    d = st.sidebar.number_input('d (Differencing order)', min_value=0, value=1)
    q = st.sidebar.number_input('q (MA order)', min_value=0, value=1)
if model_type in ['SARIMA']:
    p = st.sidebar.number_input('p (AR order)', min_value=0, value=1)
    d = st.sidebar.number_input('d (Differencing order)', min_value=0, value=1)
    q = st.sidebar.number_input('q (MA order)', min_value=0, value=1)
    seasonal_period = st.sidebar.number_input('Seasonal Period', min_value=1, value=12)
    P = st.sidebar.number_input('P (Seasonal AR order)', min_value=0, value=1)
    D = st.sidebar.number_input('D (Seasonal differencing order)', min_value=0, value=1)
    Q = st.sidebar.number_input('Q (Seasonal MA order)', min_value=0, value=1)

# Execute Forecast Button
if st.sidebar.button("Execute Forecast"):
    # Placeholder for results
    st.write("## Forecast Results")

    if model_type == 'ARIMA':
        arr_test_forecast_series,test_forecast_arima,train,test,mse,rmse=Arima_model(df,p,d,q)
        # model = ARIMA(df['Open'], order=(p, d, q), seasonal_order=(P, D, Q, seasonal_period))
        # model_fit = model.fit(disp=False)
        # forecast = model_fit.forecast(steps=12)
        plt.figure(figsize=(14,7))
        plt.plot(train, label='Training Data')
        plt.plot(test, label='Actual Data', color='orange')
        plt.plot(arr_test_forecast_series, label='Forecasted Data', color='green')
        plt.fill_between(test.index, 
                        test_forecast_arima.conf_int().iloc[:, 0], 
                        test_forecast_arima.conf_int().iloc[:, 1], 
                        color='k', alpha=.15)
        plt.title('ARIMA Model Evaluation')
        plt.xlabel('Date')
        plt.ylabel('Open Price')
        plt.legend()
        plt.show()
        st.write("### ARIMA Model Forecast")
        st.pyplot(plt)

        print('RMSE:', rmse)
        
       
        # st.line_chart(forecast)
        
        st.write(f'Mean Squared Error: {mse}')
        st.write(f'Root Mean Squared Error: {rmse}')
    elif model_type == 'LSTM':
        # ANN Model Placeholder - Add your ANN model code here
        st.write("### ANN Model Forecast")
        mse,mae,rmse,r2=lstm_model(df)
        st.write(f'Mean Squared Error: {mse}')
        st.write(f'Root Mean Squared Error: {rmse}')
        st.write(f'Mean Absolute Error: {mae}')
        st.write(f'R-squared (R2) Score: {r2}')
        
    elif model_type == 'Hybrid ARIMA-ANN':
        # Hybrid ARIMA-ANN Model Placeholder - Add your hybrid model code here
        st.write("### Hybrid ARIMA-ANN Model Forecast")
        mse,mae,rmse,r2=Hybrid_ann_arima(df,p,d,q)
        st.write(f'Mean Squared Error: {mse}')
        st.write(f'Root Mean Squared Error: {rmse}')
        st.write(f'Mean Absolute Error: {mae}')
        st.write(f'R-squared (R2) Score: {r2}')
        

    elif model_type == 'SARIMA':
        forecast,mse,rmse=Sarima_model(df,p,d,q,P,D,Q,seasonal_period)
        plt.figure(figsize=(12, 6))
        plt.plot(df['Open'], label='Actual')
        plt.plot(forecast, label='Forecast')
        plt.title('Actual vs Forecast')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        st.pyplot(plt)
        st.write(f'Mean Squared Error: {mse}')
        st.write(f'Root Mean Squared Error: {rmse}')
