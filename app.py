# app.py
import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("Crypto Liquidity Predictor")

ma7 = st.number_input("7-day MA")
ma14 = st.number_input("14-day MA")
vol = st.number_input("Volatility")
lr = st.number_input("Liquidity Ratio")

if st.button("Predict Liquidity"):
    input_data = np.array([[ma7, ma14, vol, lr]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Liquidity: {prediction:.4f}")
