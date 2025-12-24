# 🎙️ Speech Emotion Recognition Web App

A machine learning–based web application that predicts **human emotions from speech audio** using **MFCC features** and an **SVM (Support Vector Machine)** model.  
The application is built with **Python**, **Librosa**, **Scikit-learn**, and **Streamlit**.

---

## 🚀 Live Demo
Upload a `.wav` audio file and instantly get the predicted **emotion**.

---

## 🧠 Emotions Supported
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fear
- Disgust
- Surprise

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- Librosa (audio processing)
- NumPy & Pandas
- Scikit-learn (SVM)
- MFCC feature extraction

### Web Framework
- Streamlit

---

## 📂 Project Structure

Emotion-recognition/

├── app.py # Streamlit frontend

├── function.py # Backend ML logic

├── audio/ # Dataset audio files

├── uploads/ # User uploaded audio files

├── requirements.txt # Project dependencies

├── README.md # Project documentation

└── speech_emotion_model.pkl (optional)


---

## ⚙️ How It Works

1. User uploads a `.wav` audio file
2. Audio is converted to **MFCC features**
3. Trained **SVM model** predicts emotion
4. Emotion result is displayed on the web UI

---

## 🖥️ How to Run the Project

```bash
git clone https://github.com/ghias20/Emotion-recognition.git
cd speech-emotion-recognition

###Install Dependencies
```bash
pip install -r requirements.txt

### Run the Streamlit app
```bash
streamlit run app.py

 

### Dataset:https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio


### testing Audio:https://www.kaggle.com/datasets/pavanelisetty/sample-audio-files-for-speech-recognition
