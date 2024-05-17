import pandas as pd 
import numpy as np
import math 
import string
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


def Arima_model(df,p,d,q):
    df_diff=df['Open'].diff().dropna()
    train_size = int(len(df_diff) * 0.8)
    train, test = df_diff[0:train_size], df_diff[train_size:len(df_diff)]
    ar_model_train = ARIMA(train, order=(p,d,q))
    model_train_fit_arima = ar_model_train.fit()
    test_forecast_arima=model_train_fit_arima.get_forecast(steps=(len(test)))
    arr_test_forecast_series = pd.Series(test_forecast_arima.predicted_mean, index=test.index)
    mse = mean_squared_error(test, arr_test_forecast_series)
    rmse = mse**0.5
    return arr_test_forecast_series,test_forecast_arima,train,test,mse,rmse


def Sarima_model(df,p,d,q,P,D,Q,seasonal_period):
    model = SARIMAX(df['Open'], order=(p, d, q), seasonal_order=(P, D, Q, seasonal_period))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=12)
    mse = mean_squared_error(df['Open'], forecast)
    rmse = mse**0.5
    return forecast,mse,rmse


def lstm_model(df):
    scaler=MinMaxScaler()
    normaldf= pd.DataFrame(scaler.fit_transform(df), columns=df.columns,index=df.index)
    # Split the data into training and test sets
    train_size = int(len(normaldf) * 0.8)
    train, test = normaldf[0:train_size], normaldf[train_size:len(normaldf)]

    # Create sequences for the neural network
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data.iloc[i:i + seq_length].values)
            y.append(data.iloc[i + seq_length].values)
        return np.array(X), np.array(y)

    seq_length = 10
    X_train, y_train = create_sequences(train, seq_length)
    X_test, y_test = create_sequences(test, seq_length)
    # Define the model
    model = Sequential()
    model.add(LSTM(200, return_sequences=True, input_shape=(seq_length, X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(170, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(X_train.shape[2]))  # Output layer matches number of features

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Summary of the model
    model.summary()
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    history = model.fit(X_train, y_train, epochs=50, batch_size=50, validation_data=(X_test, y_test), callbacks=[early_stop])
    # history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    loss = model.evaluate(X_test, y_test)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Inverse scaling of predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)

    # Inverse scaling of actual values
    y_train_inv = scaler.inverse_transform(y_train)
    y_test_inv = scaler.inverse_transform(y_test)
    mse = mean_squared_error(y_test_inv, test_predictions)
    mae = mean_absolute_error(y_test_inv, test_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, test_predictions)

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)

    plt.figure(figsize=(14, 5))
    plt.plot(df.index[train_size + seq_length:], y_test_inv[:, 0], label='Actual')
    plt.plot(df.index[train_size + seq_length:], test_predictions[:, 0], label='Predicted')
    plt.title('Actual vs Predicted')
    plt.legend()
    st.pyplot(plt)
    return mse,mae,rmse,r2



def Hybrid_ann_arima(df,p,d,q):
    data=df['Open']
  
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    arima_order = (p, d, q)
    arima_model = ARIMA(train, order=arima_order)
    arima_fit = arima_model.fit()

  
    arima_forecast = arima_fit.forecast(steps=len(test))
    arima_residuals = test - arima_forecast
 
    X = arima_forecast.values.reshape(-1, 1)
    y = arima_residuals.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

 
    ann_model = Sequential([
        Dense(256, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    ann_model.compile(optimizer='adam', loss='mean_squared_error')
    ann_model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=1)


    ann_residuals_pred = ann_model.predict(scaler.transform(X))

    final_forecast = arima_forecast.values + ann_residuals_pred.flatten()

  

    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.values, label='Original Data')
    plt.plot(test.index, arima_forecast.values, label='ARIMA Forecast')
    plt.plot(test.index[:len(final_forecast)], final_forecast, label='Combined Forecast')
    plt.legend()
    st.pyplot(plt)
    mse = mean_squared_error(test.values, final_forecast)
    mae = mean_absolute_error(test.values, final_forecast)
    rmse = np.sqrt(mse)
    r2 = r2_score(test.values, final_forecast)
    return mse,mae,rmse,r2



