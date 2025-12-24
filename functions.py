import os
import librosa
import numpy as np
import pandas as pd
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}


emotion_to_num = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fear": 5,
    "disgust": 6,
    "surprise": 7
}

def extract_mfcc(path):
    audio, sr = librosa.load(path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)


def audio_to_dataset():
    rows = []

    folder = "audio/" 

    files = glob.glob(f"{folder}/**/*.wav", recursive=True)
    print("Total wav files found:", len(files))

    for file_path in files:
        file = os.path.basename(file_path)

        parts = file.replace("_", "-").split('-')

        emotion_code = parts[2]
        intensity = parts[3]
        statement = parts[4]
        repetition = parts[5]
        actor_id = parts[6].split('.')[0]

        gender = "female" if int(actor_id) % 2 == 0 else "male"

        mfcc_features = extract_mfcc(file_path)

        row = {
            "emotion": emotion_map[emotion_code],
            "gender": gender,
            "actor_id": int(actor_id),
            "intensity": intensity,
            "statement": statement,
            "repetition": repetition
        }

        for i, val in enumerate(mfcc_features):
            row[f"mfcc_{i+1}"] = val

        rows.append(row)
    return rows


# data1=audio_to_dataset()
# data=pd.DataFrame(data=data1)
# print(data.shape)


def train_and_test_model(df):
    df["emotion_num"] = df["emotion"].map(emotion_to_num)
    # Split dataset
    X = df.filter(like="mfcc").values
    y = df["emotion_num"].values

    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create pipeline: Scale → SVM
    svm_model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=10, gamma='scale')
    )

    # Train
    svm_model.fit(X_train, y_train)

    # Predict
    y_pred_svm = svm_model.predict(X_test)

    joblib.dump(svm_model, "svm_emotion_model.pkl")
    print("Model saved")

    
data=pd.read_csv("dataset.csv")
print(train_and_test_model(data))

# load model ONCE
svm_model = joblib.load("svm_emotion_model.pkl")

def predict_emotion(audio_path):
    audio, sr = librosa.load(audio_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)

    pred = svm_model.predict(features)[0]

    for emo, num in emotion_to_num.items():
        if num == pred:
            return emo
