import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model and the vectorizer
model_path = "data/models/logistic_regression.pkl"
vectorizer_path = "data/models/vectorizer.pkl"
model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# Streamlit app
st.title("AI or Non-AI Predictor")
st.caption("Enter a text and click on Predict to check if it's AI-related or not.")
user_input = st.text_input("Enter a word, bigram, or trigram:")

if st.button("Predict"):
    # Transform the input text using the loaded vectorizer
    transformed_input = vectorizer.transform([user_input])
    
    # Make a prediction
    prediction = model.predict(transformed_input)
    prob = model.predict_proba(transformed_input)

    # Display the prediction
    if prediction[0] == 1:
        # Make two columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p style='color: green; font-weight: bold;'>The text is related to AI.</p>", unsafe_allow_html=True)
        with col2:
            st.metric("Certainty", f"{prob[0][1].round(2)}%")
       
    else:
        # Make two columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p style='color: red; font-weight: bold;'>The text is not related to AI.</p>", unsafe_allow_html=True)
        with col2:
            st.metric("Certainty", f"{prob[0][0].round(2)}%")

