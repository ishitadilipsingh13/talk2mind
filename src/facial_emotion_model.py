"""
Talk2Mind - Facial Emotion Recognition Module
==============================================

In production this module would wrap a CNN backbone (ResNet / EfficientNet)
fine-tuned on a facial-expression dataset (e.g. FER2013, AffectNet) to
classify emotions frame-by-frame from webcam video and pool them into a
session-level embedding + emotion distribution.

Since no real video data/GPU training is available here, this module is
built to train on the FEATURES already present in facial_features.csv
(face_prob_*, eye_contact_ratio, blink_rate_per_min, facial_muscle_tension,
smile_frequency, head_movement_variance, visual_embed_*) which stand in for
what a CNN backbone would output after inference. This keeps the module's
interface identical to what you'd swap in a real CNN for later:

    model = FacialEmotionModel()
    model.fit(X_train, y_train)
    probs, dominant_emotion = model.predict(features_row)

Swap-in path for a real model: replace `extract_features_from_video()`
with actual OpenCV/MediaPipe face detection + a torchvision/keras CNN
forward pass producing the same feature schema, and the rest of the
pipeline (fusion, scoring) does not need to change.
"""

import numpy as np
import pandas as pd

# Guard against newer pandas versions (2.2+) that default string/object
# columns to a PyArrow-backed dtype. That backend is not fully compatible
# with scikit-learn's array indexing, causing a
# "TypeError: only integer scalar arrays can be converted to a scalar index"
# on train_test_split. Forcing plain-object string storage avoids it.
try:
    pd.set_option("future.infer_string", False)
except Exception:
    pass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

FACIAL_EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]


class FacialEmotionModel:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.feature_cols = None

    @staticmethod
    def _feature_columns(df):
        cols = [c for c in df.columns if c.startswith("visual_embed_")]
        cols += ["eye_contact_ratio", "blink_rate_per_min", "facial_muscle_tension",
                 "smile_frequency", "head_movement_variance"]
        return cols

    def fit(self, df: pd.DataFrame):
        """df must contain the facial feature columns + dominant_facial_emotion label."""
        self.feature_cols = self._feature_columns(df)
        X = df[self.feature_cols].to_numpy(dtype=float)
        y = df["dominant_facial_emotion"].astype(str).to_numpy(dtype=object)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.clf.fit(X_train, y_train)
        preds = self.clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"[FacialEmotionModel] Test Accuracy: {acc:.3f}")
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


def extract_features_from_video(video_path: str) -> dict:
    """
    PLACEHOLDER for real deployment.
    Real implementation would:
      1. Use OpenCV/MediaPipe to detect the face in each frame.
      2. Run a pretrained CNN (ResNet/EfficientNet) fine-tuned for FER
         to get per-frame emotion probabilities.
      3. Aggregate (mean-pool) over the session + compute engineered
         signals: blink rate, eye contact ratio, facial muscle tension
         (e.g. via facial action units / landmarks), smile frequency,
         head movement variance.
    Returns a dict matching the facial_features.csv schema (minus user_id).
    """
    raise NotImplementedError(
        "Real video-based extraction requires OpenCV + a trained CNN checkpoint. "
        "Use the synthetic facial_features.csv for training/demo purposes."
    )


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "..", "data", "facial_features.csv")
    df = pd.read_csv(data_path)

    model = FacialEmotionModel()
    model.fit(df)

    out_path = os.path.join(here, "..", "data", "facial_emotion_model.joblib")
    model.save(out_path)
    print(f"Saved trained facial emotion model -> {out_path}")
