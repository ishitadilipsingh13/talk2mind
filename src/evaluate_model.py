"""
Talk2Mind - Model Evaluation and Performance Analysis
=======================================================

Runs evaluation for all trained components and produces a summary report:
  - Facial Emotion Recognition Model: accuracy, per-class report
  - Speech Emotion Recognition Model: accuracy, per-class report
  - Fusion MLP (Mental Well-Being Score regressor): MAE, RMSE, R2
  - Risk-category classification derived from predicted score: accuracy, confusion matrix

Outputs a text report to data/evaluation_report.txt and prints to console.
"""

import os
import numpy as np
import pandas as pd

try:
    pd.set_option("future.infer_string", False)
except Exception:
    pass
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, mean_absolute_error,
    mean_squared_error, r2_score, confusion_matrix
)

from facial_emotion_model import FacialEmotionModel
from speech_emotion_model import SpeechEmotionModel
from feature_fusion import concat_fusion
from scoring_engine import categorize_score
from sklearn.neural_network import MLPRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")


def evaluate_facial_model(report_lines):
    df = pd.read_csv(os.path.join(DATA_DIR, "facial_features.csv"))
    model = FacialEmotionModel()
    acc = model.fit(df)
    report_lines.append(f"Facial Emotion Model Accuracy: {acc:.3f}")


def evaluate_speech_model(report_lines):
    df = pd.read_csv(os.path.join(DATA_DIR, "speech_features.csv"))
    model = SpeechEmotionModel()
    acc = model.fit(df)
    report_lines.append(f"Speech Emotion Model Accuracy: {acc:.3f}")


def evaluate_fusion_and_scoring(report_lines):
    df = pd.read_csv(os.path.join(DATA_DIR, "talk2mind_dataset.csv"))
    X, feature_cols, scaler = concat_fusion(df)
    y = df["mental_wellbeing_score"].to_numpy(dtype=float)
    y_cat = df["risk_category"].astype(str).to_numpy(dtype=object)

    X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
        X, y, y_cat, test_size=0.2, random_state=42
    )

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    preds = mlp.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    report_lines.append(f"\nFusion MLP - Mental Well-Being Score Regression:")
    report_lines.append(f"  MAE:  {mae:.2f}")
    report_lines.append(f"  RMSE: {rmse:.2f}")
    report_lines.append(f"  R2:   {r2:.3f}")

    pred_cat = pd.Series(preds).apply(categorize_score).astype(str).to_numpy(dtype=object)
    cat_acc = accuracy_score(cat_test, pred_cat)
    report_lines.append(f"\nDerived Risk Category Accuracy: {cat_acc:.3f}")
    report_lines.append("\nConfusion Matrix (rows=actual, cols=predicted):")
    labels = ["Good", "Mild Concern", "Moderate Concern", "High Concern"]
    cm = confusion_matrix(cat_test, pred_cat, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    report_lines.append(cm_df.to_string())

    report_lines.append("\nClassification Report (risk category):")
    report_lines.append(classification_report(cat_test, pred_cat, labels=labels, zero_division=0))


def main():
    report_lines = ["=" * 60, "Talk2Mind - Model Evaluation Report", "=" * 60, ""]

    print("\n--- Evaluating Facial Emotion Model ---")
    evaluate_facial_model(report_lines)

    print("\n--- Evaluating Speech Emotion Model ---")
    evaluate_speech_model(report_lines)

    print("\n--- Evaluating Fusion + Scoring ---")
    evaluate_fusion_and_scoring(report_lines)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    out_path = os.path.join(DATA_DIR, "evaluation_report.txt")
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"\nSaved evaluation report -> {out_path}")


if __name__ == "__main__":
    main()
