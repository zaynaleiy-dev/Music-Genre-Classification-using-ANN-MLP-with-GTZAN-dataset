import streamlit as st
import pandas as pd
import joblib

# Load saved model, encoder, and scaler
mlp = joblib.load("mlp_genre_model.pkl")
le = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎵 Music Genre Classifier (ANN)")
st.write("Upload a CSV row of features to predict the genre")

uploaded_file = st.file_uploader("Upload features CSV", type=["csv"])

if uploaded_file is not None:
    # Read uploaded features
    user_data = pd.read_csv(uploaded_file)
    
    # Scale features
    X = scaler.transform(user_data)
    
    # Predict
    prediction = mlp.predict(X)
    genre = le.inverse_transform(prediction)
    
    st.success(f"Predicted Genre: **{genre[0]}**")
