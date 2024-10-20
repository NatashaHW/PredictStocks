import streamlit as st
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')

# Custom function to replace Lambda layer (needed for model loading)
def expand_dims_custom(x):
    return tf.expand_dims(x, axis=1)

# Function to load the correct model and weights based on stock selection
def load_model(stock_choice):
    if stock_choice == 'ASII':
        model_path = 'Model/Model_ASII_LSTMSentiment1e_Architecture.keras'
        weight_path = 'Model/model_lstm_asii_sentiment1e.weights.h5'
    elif stock_choice == 'BMRI':
        model_path = 'Model/Model_BMRI_LSTMSentiment2a_Architecture.keras'
        weight_path = 'Model/model_lstm_bmri_sentiment2a.weights.h5'
    elif stock_choice == 'TLKM':
        model_path = 'Model/Model_TLKM_LSTMSentiment2f_Architecture.keras'
        weight_path = 'Model/model_lstm_tlkm_sentiment2f.weights.h5'
    elif stock_choice == 'BBRI':
        model_path = 'Model/Model_BBRI_LSTMSentiment1k_Architecture.keras'
        weight_path = 'Model/model_lstm_bbri_sentiment1k.weights.h5'
    elif stock_choice == 'BBCA':
        model_path = 'Model/Model_BBCA_LSTMSentiment1e_Architecture.keras'
        weight_path = 'Model/model_lstm_bbca_sentiment1e.weights.h5'
    else:
        raise ValueError("Invalid stock choice")

    model = tf.keras.models.load_model(model_path, custom_objects={'expand_dims_custom': expand_dims_custom})
    model.load_weights(weight_path)
    return model

# Streamlit App
def main():
    st.title('Stock Price Prediction with Sentiment')
    st.write("This app predicts the next day's stock price based on time series data and sentiment analysis using an LSTM model.")

    # User chooses the stock they want to predict (ASII, BMRI, or TLKM)
    stock_choice = st.selectbox('Choose a Stock', ['BBCA', 'BBRI', 'BMRI', 'ASII', 'TLKM'])
    st.write(f"Predicting for: {stock_choice}")

    # Input data for the last 5 days of stock prices
    st.subheader('Input Stock Data (Last 5 Days)')
    stock_data = []
    for i in range(5):
        stock_input = st.number_input(f'Stock Price Day {i+1}', min_value=0.0, value=1000.0, step=1.0)
        stock_data.append(stock_input)

    # Input sentiment data for each of the last 5 days
    st.subheader('Input Sentiment Data (Last 5 Days)')
    sentiment_data = []
    for i in range(5):
        sentiment_input = st.selectbox(f'Sentiment Day {i+1} (1 = Positive, 0 = Neutral, -1 = Negative)', [-1, 0, 1], index=1, key=f'sentiment_{i}')
        sentiment_data.append(sentiment_input)

    if st.button('Predict Next Day Price'):
        # Load the model based on the stock choice
        model = load_model(stock_choice)

        # Prepare the input data for the model
        stock_array = np.array(stock_data).reshape(1, 5, 1)  # 5 days of stock data
        sentiment_array = np.array(sentiment_data).reshape(1, 5, 1)  # 5 days of sentiment data
        # Make the prediction
        prediction = model.predict([stock_array, sentiment_array])
        predicted_price = prediction[0][0]

        # Display the prediction result
        st.success(f'Predicted Stock Price for the Next Day ({stock_choice}): {predicted_price:.2f}')
        
if __name__ == '__main__':
    main()
