"""
Talk2Mind - Personalized Recommendation Engine
=================================================

Given a Mental Well-Being Score, risk category, and the dominant
signals driving it (e.g. high anxiety, low sleep, high facial tension),
produces a short list of tailored, non-clinical self-care recommendations.

IMPORTANT: These are general wellness suggestions, not medical advice.
When crisis indicators are present, the engine ALWAYS prioritizes
surfacing professional/crisis resources over generic tips.
"""

import random

GENERAL_TIPS = {
    "Good": [
        "Keep up your current routine — it's working well for you.",
        "Consider journaling one thing you're grateful for each day.",
        "Share what's working for you with a friend who might be struggling.",
    ],
    "Mild Concern": [
        "Try a 10-minute mindfulness or breathing exercise today.",
        "Take a short walk outside — light activity can lift mood and energy.",
        "Set a consistent sleep and wake time this week.",
        "Reach out to a friend or family member you haven't spoken to in a while.",
    ],
    "Moderate Concern": [
        "Consider talking to a counselor or therapist about how you've been feeling.",
        "Try structured relaxation techniques (e.g. progressive muscle relaxation).",
        "Limit caffeine and screen time before bed to improve sleep quality.",
        "Break large tasks into smaller, manageable steps to reduce overwhelm.",
    ],
    "High Concern": [
        "We strongly encourage you to speak with a mental health professional soon.",
        "Consider reaching out to a trusted person today and letting them know how you're feeling.",
        "If you ever feel unsafe or in crisis, please contact a crisis helpline immediately.",
    ],
}

SIGNAL_SPECIFIC_TIPS = {
    "sleep_trouble": "Your responses suggest sleep difficulty — try a consistent wind-down routine and avoid screens 30 min before bed.",
    "low_energy_fatigue": "Low energy levels detected — short bursts of physical activity (even 5-10 min) can help.",
    "concentration_trouble": "Trouble focusing? Try the Pomodoro technique: 25 min focused work, 5 min break.",
    "feeling_nervous_anxious": "Elevated anxiety detected — box breathing (4s in, 4s hold, 4s out, 4s hold) can help calm your nervous system.",
    "high_facial_tension": "We noticed elevated facial muscle tension — a few minutes of jaw/shoulder relaxation stretches may help.",
    "high_speech_pauses": "Your speech pattern showed longer pauses, often linked with stress — try speaking with someone you trust about what's on your mind.",
}

CRISIS_MESSAGE = (
    "Your responses indicate you may be experiencing thoughts of self-harm. "
    "Please know that support is available and you don't have to face this alone. "
    "Consider reaching out to a mental health professional or a crisis helpline right now."
)


def generate_recommendations(risk_category: str,
                              questionnaire_flags: list = None,
                              facial_muscle_tension: float = None,
                              speech_pause_ratio: float = None,
                              crisis_flag: bool = False,
                              n_tips: int = 3,
                              seed: int = None):
    """
    risk_category: one of Good / Mild Concern / Moderate Concern / High Concern
    questionnaire_flags: list of PHQ9/GAD7 question keys where the user scored high (>=2)
    """
    rng = random.Random(seed)
    recs = []

    if crisis_flag:
        recs.append(CRISIS_MESSAGE)

    pool = list(GENERAL_TIPS.get(risk_category, GENERAL_TIPS["Mild Concern"]))
    rng.shuffle(pool)
    recs.extend(pool[:n_tips])

    if questionnaire_flags:
        for flag in questionnaire_flags:
            if flag in SIGNAL_SPECIFIC_TIPS:
                recs.append(SIGNAL_SPECIFIC_TIPS[flag])

    if facial_muscle_tension is not None and facial_muscle_tension > 0.6:
        recs.append(SIGNAL_SPECIFIC_TIPS["high_facial_tension"])

    if speech_pause_ratio is not None and speech_pause_ratio > 0.35:
        recs.append(SIGNAL_SPECIFIC_TIPS["high_speech_pauses"])

    # de-duplicate, preserve order
    seen = set()
    unique_recs = []
    for r in recs:
        if r not in seen:
            unique_recs.append(r)
            seen.add(r)

    return unique_recs


if __name__ == "__main__":
    recs = generate_recommendations(
        risk_category="Moderate Concern",
        questionnaire_flags=["sleep_trouble", "feeling_nervous_anxious"],
        facial_muscle_tension=0.7,
        speech_pause_ratio=0.4,
        crisis_flag=False,
    )
    print("Sample Recommendations:")
    for r in recs:
        print(" -", r)
