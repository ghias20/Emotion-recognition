import streamlit as st
import os
from functions import predict_emotion  # change filename

# App title
st.set_page_config(page_title="Speech Emotion Recognition")
st.title("🎤 Speech Emotion Recognition")

st.write("Upload a WAV audio file to predict the emotion")

# Upload audio
uploaded_file = st.file_uploader(
    "",
    type=["wav"]
)

# Create uploads folder if not exists
os.makedirs("uploads", exist_ok=True)

if uploaded_file is not None:
    file_path = os.path.join("uploads", uploaded_file.name)

    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Play audio
    st.audio(file_path, format="audio/wav")

    # Predict button
    if st.button("Predict Emotion"):
        emotion = predict_emotion(file_path)
        st.success(f"🎯 Predicted Emotion: **{emotion.upper()}**")