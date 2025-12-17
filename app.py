import streamlit as st
import librosa
import numpy as np
import joblib

model = joblib.load("emotion_model.pkl")

st.title("Speech Emotion Recognition")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    audio, sr = librosa.load(uploaded_file, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)

    pred = model.predict(features)[0]
    st.write("Predicted Emotion:", pred)
