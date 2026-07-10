"""
Talk2Mind - Speech Emotion Recognition Module
===============================================

In production this module would extract MFCCs/pitch/energy/tempo from
raw audio (via librosa) and feed them into a BiLSTM/Transformer classifier
trained on a speech-emotion dataset (e.g. RAVDESS, CREMA-D, IEMOCAP).

This module trains on the FEATURES already present in speech_features.csv
(speech_prob_*, pitch/energy/rate/pause/jitter/shimmer, speech_embed_*)
which stand in for what the acoustic-feature-extraction + BiLSTM pipeline
would output. Swap in real audio extraction via `extract_features_from_audio()`
and the rest of the pipeline is unchanged.
"""

import numpy as np
import pandas as pd

try:
    pd.set_option("future.infer_string", False)
except Exception:
    pass
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

SPEECH_EMOTIONS = ["calm", "stressed", "sad", "anxious", "neutral", "energetic"]


class SpeechEmotionModel:
    def __init__(self):
        self.clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
        self.feature_cols = None

    @staticmethod
    def _feature_columns(df):
        cols = [c for c in df.columns if c.startswith("speech_embed_")]
        cols += ["pitch_mean_hz", "pitch_variance", "energy_mean", "speech_rate_wpm",
                  "pause_ratio", "jitter", "shimmer"]
        return cols

    def fit(self, df: pd.DataFrame):
        self.feature_cols = self._feature_columns(df)
        X = df[self.feature_cols].to_numpy(dtype=float)
        y = df["dominant_speech_emotion"].astype(str).to_numpy(dtype=object)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.clf.fit(X_train, y_train)
        preds = self.clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"[SpeechEmotionModel] Test Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds, zero_division=0))
        return acc

    def predict(self, row: pd.Series):
        X = row[self.feature_cols].to_numpy(dtype=float).reshape(1, -1)
        probs = self.clf.predict_proba(X)[0]
        classes = self.clf.classes_
        dominant = classes[int(np.argmax(probs))]
        return dict(zip(classes, probs)), dominant

    def save(self, path):
        joblib.dump({"clf": self.clf, "feature_cols": self.feature_cols}, path)

    def load(self, path):
        obj = joblib.load(path)
        self.clf = obj["clf"]
        self.feature_cols = obj["feature_cols"]
        return self


def extract_features_from_audio(audio_path: str) -> dict:
    """
    PLACEHOLDER for real deployment.
    Real implementation would use `librosa` to load the waveform and compute:
      - MFCCs (e.g. librosa.feature.mfcc)
      - pitch (librosa.pyin), energy (RMS), tempo/speech-rate
      - jitter/shimmer (via a library like parselmouth/Praat)
    then feed the feature vector into a trained BiLSTM/Transformer classifier.
    Returns a dict matching the speech_features.csv schema (minus user_id).
    """
    raise NotImplementedError(
        "Real audio-based extraction requires librosa/parselmouth + a trained "
        "checkpoint. Use the synthetic speech_features.csv for training/demo."
    )


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "..", "data", "speech_features.csv")
    df = pd.read_csv(data_path)

    model = SpeechEmotionModel()
    model.fit(df)

    out_path = os.path.join(here, "..", "data", "speech_emotion_model.joblib")
    model.save(out_path)
    print(f"Saved trained speech emotion model -> {out_path}")
