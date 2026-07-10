"""
Talk2Mind - Multimodal Feature Fusion Module
==============================================

Combines facial-emotion embeddings, speech-emotion embeddings, and
questionnaire responses into a single fused feature vector, which is
then passed to the Mental Well-Being Scoring Engine.

Two fusion strategies are provided:
  1. `concat_fusion`   - simple concatenation of normalized features (fast, interpretable)
  2. `weighted_fusion`  - weighted sum of per-modality "distress proxies" (used by
                          the scoring engine / matches the architecture diagram's
                          "visual_emotion_embeddings + speech_emotion_embeddings" box)

In a deeper implementation, `concat_fusion` output could instead be fed
into a small trainable fusion MLP (see `train_fusion_mlp`) that learns
optimal modality weights instead of hand-set ones.
"""

import numpy as np
import pandas as pd

try:
    pd.set_option("future.infer_string", False)
except Exception:
    pass
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os


QUESTIONNAIRE_COLS = None  # set dynamically
FACIAL_COLS = None
SPEECH_COLS = None


def get_modality_columns(df: pd.DataFrame):
    quest_cols = [c for c in df.columns if c.startswith("phq9_") or c.startswith("gad7_")]
    facial_cols = [c for c in df.columns if c.startswith("face_") or c.startswith("visual_embed_")
                   or c in ["eye_contact_ratio", "blink_rate_per_min", "facial_muscle_tension",
                             "smile_frequency", "head_movement_variance"]]
    speech_cols = [c for c in df.columns if c.startswith("speech_") and c != "speech_rate_wpm" or
                   c in ["pitch_mean_hz", "pitch_variance", "energy_mean", "speech_rate_wpm",
                         "pause_ratio", "jitter", "shimmer"]]
    # de-duplicate while preserving order
    speech_cols = list(dict.fromkeys(speech_cols))
    return quest_cols, facial_cols, speech_cols


def concat_fusion(df: pd.DataFrame):
    """Concatenate & standardize numeric features from all 3 modalities."""
    quest_cols, facial_cols, speech_cols = get_modality_columns(df)
    numeric_facial = [c for c in facial_cols if pd.api.types.is_numeric_dtype(df[c])]
    numeric_speech = [c for c in speech_cols if pd.api.types.is_numeric_dtype(df[c])]

    feature_cols = quest_cols + numeric_facial + numeric_speech
    X = df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, feature_cols, scaler


def train_fusion_mlp(df: pd.DataFrame, target_col="mental_wellbeing_score", save_path=None):
    """
    Trains a small MLP regressor that learns to predict the Mental
    Well-Being Score directly from the fused feature vector
    (this is the "Feature Fusion Layer" learning optimal modality weights).
    """
    X, feature_cols, scaler = concat_fusion(df)
    y = df[target_col].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)

    preds = mlp.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"[FusionMLP] MAE: {mae:.2f}, R2: {r2:.3f}")

    if save_path:
        joblib.dump({"model": mlp, "scaler": scaler, "feature_cols": feature_cols}, save_path)
        print(f"Saved fusion model -> {save_path}")

    return mlp, scaler, feature_cols, {"mae": mae, "r2": r2}


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "..", "data", "talk2mind_dataset.csv")
    df = pd.read_csv(data_path)

    out_path = os.path.join(here, "..", "data", "fusion_mlp_model.joblib")
    train_fusion_mlp(df, save_path=out_path)
