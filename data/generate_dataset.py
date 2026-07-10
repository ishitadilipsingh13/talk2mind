"""
Talk2Mind - Synthetic Multimodal Dataset Generator
====================================================

Generates a realistic SYNTHETIC dataset for the Talk2Mind project since
collecting real facial video + speech audio + questionnaire data from
human subjects requires ethics approval and informed consent.

This script creates:
    1. talk2mind_dataset.csv       -> master fused dataset (one row per user session)
    2. facial_features.csv         -> simulated facial-expression embeddings
    3. speech_features.csv         -> simulated speech/audio embeddings
    4. questionnaire_responses.csv -> simulated PHQ-9 + GAD-7 style responses

The synthetic generation process is *label-consistent*: a hidden
"true stress/anxiety level" per user drives correlated facial, speech and
questionnaire signals, the same way real depressed/anxious individuals
tend to show correlated behavioral cues. This makes the data usable for
training/demoing the downstream fusion + scoring models.

Usage:
    python generate_dataset.py --n_users 2000 --seed 42
"""

import argparse
import numpy as np
import pandas as pd
import os

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

FACIAL_EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]
SPEECH_EMOTIONS = ["calm", "stressed", "sad", "anxious", "neutral", "energetic"]

# PHQ-9 (depression) style questions (0-3 scale each)
PHQ9_QUESTIONS = [
    "little_interest_or_pleasure",
    "feeling_down_or_hopeless",
    "sleep_trouble",
    "low_energy_fatigue",
    "appetite_changes",
    "feeling_bad_about_self",
    "concentration_trouble",
    "moving_speaking_slowly_or_restless",
    "thoughts_of_self_harm",
]

# GAD-7 (anxiety) style questions (0-3 scale each)
GAD7_QUESTIONS = [
    "feeling_nervous_anxious",
    "cant_stop_worrying",
    "worrying_too_much",
    "trouble_relaxing",
    "restlessness",
    "easily_annoyed_irritable",
    "feeling_afraid",
]


def make_rng(seed):
    return np.random.default_rng(seed)


def generate_users(n_users, rng):
    """Generate base demographic + a hidden latent distress score per user."""
    user_ids = [f"U{i:05d}" for i in range(1, n_users + 1)]
    age = rng.integers(16, 60, size=n_users)
    gender = rng.choice(["male", "female", "other"], size=n_users, p=[0.47, 0.47, 0.06])
    occupation = rng.choice(
        ["student", "employee", "unemployed", "self_employed"],
        size=n_users, p=[0.4, 0.42, 0.1, 0.08]
    )

    # Hidden ground-truth latent distress in [0, 1]. 0 = thriving, 1 = severe distress.
    # Modeled as a mixture: most people low-moderate, smaller tail of high distress.
    latent_distress = np.clip(
        rng.beta(a=2.0, b=5.0, size=n_users) + rng.normal(0, 0.03, n_users), 0, 1
    )

    df = pd.DataFrame({
        "user_id": user_ids,
        "age": age,
        "gender": gender,
        "occupation": occupation,
        "_latent_distress": latent_distress,  # hidden variable, kept for generating other tables
    })
    return df


def generate_questionnaire(users_df, rng):
    """Simulate PHQ-9 + GAD-7 responses correlated with latent distress."""
    n = len(users_df)
    rows = []
    for i in range(n):
        distress = users_df["_latent_distress"].iloc[i]
        row = {"user_id": users_df["user_id"].iloc[i]}

        # Each question score 0-3, mean scales with distress, with individual noise
        for q in PHQ9_QUESTIONS:
            if q == "thoughts_of_self_harm":
                # Keep this rare/low even at high distress (synthetic, non-clinical)
                p = np.clip(distress * 0.35, 0, 0.9)
            else:
                p = np.clip(0.15 + distress * 0.75, 0, 0.95)
            score = rng.binomial(3, p)
            row[f"phq9_{q}"] = int(score)

        for q in GAD7_QUESTIONS:
            p = np.clip(0.15 + distress * 0.75, 0, 0.95)
            score = rng.binomial(3, p)
            row[f"gad7_{q}"] = int(score)

        rows.append(row)

    qdf = pd.DataFrame(rows)
    phq_cols = [f"phq9_{q}" for q in PHQ9_QUESTIONS]
    gad_cols = [f"gad7_{q}" for q in GAD7_QUESTIONS]
    qdf["phq9_total"] = qdf[phq_cols].sum(axis=1)          # 0-27
    qdf["gad7_total"] = qdf[gad_cols].sum(axis=1)           # 0-21
    return qdf


def generate_facial_features(users_df, rng):
    """
    Simulate facial-expression-derived features as if extracted by a
    CNN (ResNet/EfficientNet) backbone: emotion probability distribution
    + engineered micro-expression signals.
    """
    n = len(users_df)
    rows = []
    for i in range(n):
        distress = users_df["_latent_distress"].iloc[i]
        row = {"user_id": users_df["user_id"].iloc[i]}

        # Emotion probabilities: higher distress -> more sad/fear/angry, less happy
        base = np.array([
            0.35,  # neutral
            0.30,  # happy
            0.08,  # sad
            0.05,  # angry
            0.05,  # fear
            0.10,  # surprise
            0.07,  # disgust
        ])
        shift = np.array([0.0, -0.28, 0.20, 0.08, 0.10, -0.05, 0.05]) * distress
        probs = np.clip(base + shift + rng.normal(0, 0.03, 7), 0.001, None)
        probs = probs / probs.sum()
        for emo, p in zip(FACIAL_EMOTIONS, probs):
            row[f"face_prob_{emo}"] = round(float(p), 4)

        row["dominant_facial_emotion"] = FACIAL_EMOTIONS[int(np.argmax(probs))]

        # Micro-expression / behavioral engineered features
        row["eye_contact_ratio"] = round(float(np.clip(rng.normal(0.75 - 0.35 * distress, 0.08), 0, 1)), 3)
        row["blink_rate_per_min"] = round(float(np.clip(rng.normal(15 + 10 * distress, 4), 4, 45)), 1)
        row["facial_muscle_tension"] = round(float(np.clip(rng.normal(0.3 + 0.5 * distress, 0.1), 0, 1)), 3)
        row["smile_frequency"] = round(float(np.clip(rng.normal(0.5 - 0.4 * distress, 0.12), 0, 1)), 3)
        row["head_movement_variance"] = round(float(np.clip(rng.normal(0.4 + 0.3 * distress, 0.15), 0, 1)), 3)

        # 16-dim simulated visual embedding (as if pooled from ResNet/EfficientNet)
        emb = rng.normal(distress, 1.0, size=16)
        for j, v in enumerate(emb):
            row[f"visual_embed_{j}"] = round(float(v), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def generate_speech_features(users_df, rng):
    """
    Simulate speech-emotion-derived features as if extracted via
    MFCCs / pitch / energy / tempo + a BiLSTM/Transformer embedding.
    """
    n = len(users_df)
    rows = []
    for i in range(n):
        distress = users_df["_latent_distress"].iloc[i]
        row = {"user_id": users_df["user_id"].iloc[i]}

        base = np.array([0.30, 0.10, 0.10, 0.10, 0.30, 0.10])  # calm,stressed,sad,anxious,neutral,energetic
        shift = np.array([-0.20, 0.22, 0.15, 0.18, -0.05, -0.15]) * distress
        probs = np.clip(base + shift + rng.normal(0, 0.03, 6), 0.001, None)
        probs = probs / probs.sum()
        for emo, p in zip(SPEECH_EMOTIONS, probs):
            row[f"speech_prob_{emo}"] = round(float(p), 4)
        row["dominant_speech_emotion"] = SPEECH_EMOTIONS[int(np.argmax(probs))]

        # Classic acoustic features
        row["pitch_mean_hz"] = round(float(np.clip(rng.normal(180 + 25 * distress, 20), 80, 320)), 1)
        row["pitch_variance"] = round(float(np.clip(rng.normal(30 + 15 * distress, 8), 5, 100)), 2)
        row["energy_mean"] = round(float(np.clip(rng.normal(0.55 - 0.2 * distress, 0.1), 0.05, 1)), 3)
        row["speech_rate_wpm"] = round(float(np.clip(rng.normal(150 - 25 * distress, 15), 60, 220)), 1)
        row["pause_ratio"] = round(float(np.clip(rng.normal(0.15 + 0.25 * distress, 0.07), 0, 0.8)), 3)
        row["jitter"] = round(float(np.clip(rng.normal(0.01 + 0.015 * distress, 0.005), 0.001, 0.08)), 4)
        row["shimmer"] = round(float(np.clip(rng.normal(0.03 + 0.03 * distress, 0.01), 0.005, 0.15)), 4)

        # 16-dim simulated MFCC-derived embedding
        emb = rng.normal(distress * -1, 1.0, size=16)
        for j, v in enumerate(emb):
            row[f"speech_embed_{j}"] = round(float(v), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_wellbeing_score(qdf, fdf, sdf, rng):
    """
    Compute a 0-100 Mental Well-Being Score (100 = best) by fusing
    questionnaire, facial and speech signals, plus a risk category label.
    This mirrors the "Feature Fusion and Mental Health Scoring" step,
    used here to create a plausible TARGET label for model training.
    """
    phq_norm = qdf["phq9_total"] / 27.0
    gad_norm = qdf["gad7_total"] / 21.0

    face_distress_proxy = (
        fdf["face_prob_sad"] + fdf["face_prob_fear"] + fdf["face_prob_angry"]
        - fdf["face_prob_happy"] + fdf["facial_muscle_tension"] * 0.5
    )
    face_norm = np.clip((face_distress_proxy - face_distress_proxy.min())
                         / (face_distress_proxy.max() - face_distress_proxy.min() + 1e-9), 0, 1)

    speech_distress_proxy = (
        sdf["speech_prob_stressed"] + sdf["speech_prob_anxious"] + sdf["speech_prob_sad"]
        - sdf["speech_prob_calm"] + sdf["pause_ratio"] * 0.5
    )
    speech_norm = np.clip((speech_distress_proxy - speech_distress_proxy.min())
                           / (speech_distress_proxy.max() - speech_distress_proxy.min() + 1e-9), 0, 1)

    # Weighted fusion (weights are configurable - matches "Behavioral Feature Fusion")
    w_quest, w_face, w_speech = 0.5, 0.25, 0.25
    combined_distress = (
        w_quest * (0.5 * phq_norm + 0.5 * gad_norm)
        + w_face * face_norm
        + w_speech * speech_norm
    )
    combined_distress = np.clip(combined_distress + rng.normal(0, 0.02, len(combined_distress)), 0, 1)

    wellbeing_score = np.round((1 - combined_distress) * 100, 1)

    def categorize(score):
        if score >= 75:
            return "Good"
        elif score >= 55:
            return "Mild Concern"
        elif score >= 35:
            return "Moderate Concern"
        else:
            return "High Concern"

    risk_category = wellbeing_score.apply(categorize)
    return wellbeing_score, risk_category


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Talk2Mind dataset")
    parser.add_argument("--n_users", type=int, default=2000, help="Number of synthetic user sessions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    rng = make_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Generating {args.n_users} synthetic user sessions (seed={args.seed})...")

    users_df = generate_users(args.n_users, rng)
    qdf = generate_questionnaire(users_df, rng)
    fdf = generate_facial_features(users_df, rng)
    sdf = generate_speech_features(users_df, rng)

    wellbeing_score, risk_category = compute_wellbeing_score(qdf, fdf, sdf, rng)

    # Save individual modality tables
    qdf.to_csv(os.path.join(args.out_dir, "questionnaire_responses.csv"), index=False)
    fdf.to_csv(os.path.join(args.out_dir, "facial_features.csv"), index=False)
    sdf.to_csv(os.path.join(args.out_dir, "speech_features.csv"), index=False)

    # Build fused master dataset
    meta = users_df.drop(columns=["_latent_distress"])
    master = meta.merge(qdf, on="user_id").merge(fdf, on="user_id").merge(sdf, on="user_id")
    master["mental_wellbeing_score"] = wellbeing_score
    master["risk_category"] = risk_category

    master.to_csv(os.path.join(args.out_dir, "talk2mind_dataset.csv"), index=False)

    print("Done. Files written to:", args.out_dir)
    print(" - questionnaire_responses.csv:", qdf.shape)
    print(" - facial_features.csv:", fdf.shape)
    print(" - speech_features.csv:", sdf.shape)
    print(" - talk2mind_dataset.csv (fused):", master.shape)
    print("\nRisk category distribution:")
    print(master["risk_category"].value_counts())


if __name__ == "__main__":
    main()
