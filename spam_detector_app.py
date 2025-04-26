import csv
from datetime import datetime


import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk


nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

st.markdown(
    """
    <style>
    body {
        background-color: #121212;  /* Black background */
        color: white;  /* White text */
        font-family: 'Arial', sans-serif;
    }

    .title {
        font-size: 50px;  /* Increased title font size */
        font-weight: bold;
        color: #4CAF50;  /* Green color */
        text-align: center;
    }

    .description {
        font-size: 18px;
        color: #b0b0b0;  /* Light gray */
        text-align: center;
        margin-bottom: 30px;
    }

    .input-section {
        padding: 30px;
        border-radius: 15px;
        width: 60%;
        margin: 0 auto;
    }

    .input-section textarea {
        width: 100%;
        height: 150px;
        padding: 15px;
        font-size: 16px;
        border: 2px solid #555;
        border-radius: 8px;
        margin-bottom: 20px;
        background-color: #333;  /* Dark gray text area */
        color: white;
    }

    .input-section button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        width: 100%;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    .input-section button:hover {
        background-color: #45a049;
    }

    .result-section {
        margin-top: 20px;
        font-size: 30px;  /* Increased font size for the result */
        text-align: center;
    }

    .footer {
        font-size: 14px;
        color: #b0b0b0;  /* Light gray footer text */
        text-align: center;
        margin-top: 40px;
    }

    </style>
    """, unsafe_allow_html=True
)

st.markdown('<p class="title">📩 Text Spam Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Enter your text below to check if it\'s spam or not.</p>', unsafe_allow_html=True)

st.markdown('<div class="input-section">', unsafe_allow_html=True)
user_input = st.text_area("Your message", placeholder="Type your message here...")
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Check", help="Click to check if the message is spam or ham"):
    clean_text = preprocess(user_input)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)

    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    if prediction[0] == 1:
        st.error("🚨 This is a SPAM text!", icon="🚨")
    else:
        st.success("This is a legit text.", icon="✅")
    st.markdown('</div>', unsafe_allow_html=True)



