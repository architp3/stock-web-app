import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
from st_aggrid import AgGrid
from keras.models import load_model
import tensorflow as tf


# Project Title
st.markdown("<h1 style='text-align: center; color: grey;'>Stock Trend Analysis and Prediction</h1>", unsafe_allow_html=True)

# Users choice for years of stock data and companies
base_year = 2010
min_diff = 3
current_year = int(datetime.strftime(datetime.today(), '%Y'))
end_year = current_year
start_year = st.slider('Choose a Start Date', min_value=base_year, max_value = end_year)
start = f'{start_year}-01-01'
end = datetime.strftime(datetime.today(), '%Y-%m-%d')

tickers = pd.read_csv('tickers.csv')
tickers = tickers.reset_index()
tickers = tickers.drop(columns=['index','Unnamed: 0'])

stock = st.selectbox('Choose a Stock Ticker:', tickers['ticker'])


col1, col2, col3 , col4, col5 = st.columns(5)

# Center button

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    start_button = st.button('Apply Changes')

if start_button:
    st.text(f'Showing trend of {stock} for years {start_year}-{end_year}:')
    df = yf.download(str(stock), start=start, end=end)
    df = df.reset_index()
    df.set_index(['Date'], inplace=True)
    st.dataframe(df.describe())

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    
    # Show plot of rolling mean for 100 days
    roll_100 = df['Close'].rolling(100).mean()
    fig1, ax1 = plt.subplots()
    ax1.plot(df['Close'], label='Original Price')
    ax1.plot(roll_100, color='red', label='Rolling 100 Price')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Stock Price ($)')
    ax1.legend()
    st.markdown(f'<p style="font-size:20px;text-align:center;">{stock} stock trend from {start} to {end}</p>', unsafe_allow_html=True)
    st.pyplot(fig1)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    # Show plot of original, rolling mean of 100 days and rolling mean of 200 days
    roll_200 = df['Close'].rolling(200).mean()
    fig2, ax2 = plt.subplots()
    ax2.plot(df['Close'], label='Original Price')
    ax2.plot(roll_200, color='green', label='Rolling 200 Price')
    ax2.plot(roll_100, color='red', label='Rolling 100 Price')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Stock Price ($)')
    ax2.legend()
    st.markdown(f'<p style="font-size:20px;text-align:center;">{stock} stock trend from {start} to {end}</p>', unsafe_allow_html=True)
    st.pyplot(fig2)


    # Create training and test set using a 70-30 split.
    dataTrain = pd.DataFrame(df['Close'][:int(len(df) * 0.70)])
    dataTest = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

    mmscale = MinMaxScaler()
    training = mmscale.fit_transform(dataTrain)

    X_train = []
    y_train = []

    period = 100
    for i in range(period, len(training)):
        X_train.append(training[i-period:i])
        y_train.append(training[i][0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    model = tf.keras.models.load_model('pred_model.keras')

    # Model Info (Optional)

    X_test = []
    y_test = []
    
    past_100 = dataTrain.tail(100)
    print(type(past_100))
    test_input = pd.concat([past_100, dataTest], ignore_index=True)
    final_data = mmscale.fit_transform(test_input)

    for j in range(period, len(final_data)):
        X_test.append(final_data[j - period:j])
        y_test.append(final_data[j][0])
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    predicted = model.predict(X_test)
    scaler = mmscale.scale_
    predicted = predicted * 1/scaler
    y_test = y_test * 1/scaler

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    # Predicted vs Orginal Stock Prices Graph
    fig3, ax3 = plt.subplots()
    ax3.plot(y_test, label='Original Price')
    ax3.plot(predicted, color='red', label='Predicted Price')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Stock Price ($)')
    ax3.legend()
    st.markdown(f'<p style="font-size:20px;text-align:center;">{stock} stock trend from {start} to {end}</p>', unsafe_allow_html=True)
    st.pyplot(fig3)
    
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    # Model Performance Metrics
    st.markdown(f"<h3 style='text-align: left; color: white;'>Model Statistics</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: red;'>Mean Squared Error: {mean_squared_error(predicted,y_test)}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: red;'>Mean Absolute Error: {mean_absolute_error(predicted,y_test)}</p>", unsafe_allow_html=True)
    


