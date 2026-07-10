"""
Talk2Mind - Mental Well-Being Scoring Engine
==============================================

Turns fused multimodal signals into:
  - A 0-100 "Mental Well-Being Score" (100 = best)
  - A risk category: Good / Mild Concern / Moderate Concern / High Concern

Two modes:
  1. `rule_based_score()`   - transparent, explainable weighted-sum scoring
                               (good default; matches the concept-note formula)
  2. `ModelBasedScorer`     - wraps the trained fusion MLP (feature_fusion.py)
                               for a learned score, with rule-based as fallback
"""

import numpy as np
import pandas as pd
import joblib
import os


def rule_based_score(phq9_total: int, gad7_total: int,
                      face_probs: dict, speech_probs: dict,
                      facial_muscle_tension: float = 0.3,
                      pause_ratio: float = 0.15,
                      weights=(0.5, 0.25, 0.25)):
    """
    weights = (questionnaire_weight, facial_weight, speech_weight), must sum to 1.
    face_probs / speech_probs: dict of emotion -> probability (0-1).
    """
    w_quest, w_face, w_speech = weights
    assert abs(sum(weights) - 1.0) < 1e-6, "weights must sum to 1"

    phq_norm = np.clip(phq9_total / 27.0, 0, 1)
    gad_norm = np.clip(gad7_total / 21.0, 0, 1)
    quest_distress = 0.5 * phq_norm + 0.5 * gad_norm

    face_distress = np.clip(
        face_probs.get("sad", 0) + face_probs.get("fear", 0) + face_probs.get("angry", 0)
        - face_probs.get("happy", 0) + facial_muscle_tension * 0.5,
        0, 1
    )

    speech_distress = np.clip(
        speech_probs.get("stressed", 0) + speech_probs.get("anxious", 0) + speech_probs.get("sad", 0)
        - speech_probs.get("calm", 0) + pause_ratio * 0.5,
        0, 1
    )

    combined = w_quest * quest_distress + w_face * face_distress + w_speech * speech_distress
    combined = np.clip(combined, 0, 1)
    score = round((1 - combined) * 100, 1)
    return score, categorize_score(score)


def categorize_score(score: float) -> str:
    if score >= 75:
        return "Good"
    elif score >= 55:
        return "Mild Concern"
    elif score >= 35:
        return "Moderate Concern"
    return "High Concern"


class ModelBasedScorer:
    """Loads the trained fusion MLP model saved by feature_fusion.py"""

    def __init__(self, model_path):
        obj = joblib.load(model_path)
        self.model = obj["model"]
        self.scaler = obj["scaler"]
        self.feature_cols = obj["feature_cols"]

    def score(self, row: pd.Series):
        X = row[self.feature_cols].to_numpy(dtype=float).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        pred = float(np.clip(self.model.predict(X_scaled)[0], 0, 100))
        return round(pred, 1), categorize_score(pred)


if __name__ == "__main__":
    # Quick smoke test using rule_based_score
    demo_face_probs = {"neutral": 0.3, "happy": 0.1, "sad": 0.3, "angry": 0.1,
                        "fear": 0.1, "surprise": 0.05, "disgust": 0.05}
    demo_speech_probs = {"calm": 0.1, "stressed": 0.35, "sad": 0.2, "anxious": 0.2,
                          "neutral": 0.1, "energetic": 0.05}
    score, category = rule_based_score(
        phq9_total=14, gad7_total=12,
        face_probs=demo_face_probs, speech_probs=demo_speech_probs,
        facial_muscle_tension=0.6, pause_ratio=0.3
    )
    print(f"Demo Mental Well-Being Score: {score} -> {category}")
