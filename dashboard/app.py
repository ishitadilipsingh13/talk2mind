"""
Talk2Mind - Streamlit Dashboard
=================================

An interactive dashboard that lets a user:
  1. Fill out the mental health questionnaire (PHQ-9 + GAD-7 style)
  2. Simulate facial & speech emotion inputs (sliders stand in for
     live webcam/mic capture, which needs local hardware access)
  3. See their fused Mental Well-Being Score + risk category
  4. Get personalized recommendations
  5. Track score history across sessions (in-session only)

Run with:
    streamlit run app.py
"""

import os
import sys
import datetime
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from questionnaire import PHQ9_QUESTIONS, GAD7_QUESTIONS, score_questionnaire, CRISIS_QUESTION_KEY
from scoring_engine import rule_based_score, categorize_score
from recommendation_engine import generate_recommendations

st.set_page_config(page_title="Talk2Mind", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown("""
<style>
:root {
    --t2m-teal: #1b6e6b;
    --t2m-teal-dark: #12504d;
    --t2m-cream: #f7f5f0;
    --t2m-ink: #23302f;
    --t2m-accent: #e08e45;
}
.main { background-color: var(--t2m-cream); }
h1, h2, h3 { color: var(--t2m-teal-dark); font-family: 'Georgia', serif; }
.t2m-card {
    background: white;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border-left: 5px solid var(--t2m-teal);
    margin-bottom: 1rem;
}
.t2m-score {
    font-size: 3.2rem;
    font-weight: 700;
    color: var(--t2m-teal-dark);
}
.t2m-badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    color: white;
}
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🧠 Talk2Mind")
st.caption("A Multimodal AI-Based Mental Well-Being Assessment and Support System")

with st.expander("ℹ️ About & Disclaimer", expanded=False):
    st.write(
        "Talk2Mind is a self-awareness tool for educational purposes. It is **not** a "
        "diagnostic or clinical instrument and does not replace professional mental "
        "health care. If you are in crisis or thinking about self-harm, please contact "
        "a crisis helpline or mental health professional immediately."
    )

tab_assess, tab_history = st.tabs(["📝 New Assessment", "📈 Progress History"])

BADGE_COLORS = {
    "Good": "#2f9e44",
    "Mild Concern": "#f08c00",
    "Moderate Concern": "#e8590c",
    "High Concern": "#c92a2a",
}

with tab_assess:
    col_quest, col_signals = st.columns([1.3, 1])

    with col_quest:
        st.subheader("Step 1 — Questionnaire")
        st.write("Over the last 2 weeks, how often have you been bothered by the following?")
        options = {0: "Not at all", 1: "Several days", 2: "More than half the days", 3: "Nearly every day"}

        responses = {}
        st.markdown("**Depression-related items**")
        for key, text in PHQ9_QUESTIONS.items():
            responses[key] = st.select_slider(
                text, options=[0, 1, 2, 3], value=0,
                format_func=lambda x: options[x], key=f"phq_{key}"
            )

        st.markdown("**Anxiety-related items**")
        for key, text in GAD7_QUESTIONS.items():
            responses[key] = st.select_slider(
                text, options=[0, 1, 2, 3], value=0,
                format_func=lambda x: options[x], key=f"gad_{key}"
            )

    with col_signals:
        st.subheader("Step 2 — Facial & Speech Signals")
        st.caption(
            "In the full system these come from live webcam + microphone analysis "
            "during your guided conversation. Adjust the sliders here to simulate them."
        )

        st.markdown("**Simulated facial signals**")
        face_sad = st.slider("Sad expression intensity", 0.0, 1.0, 0.15, key="face_sad")
        face_fear = st.slider("Fear/tension expression intensity", 0.0, 1.0, 0.10, key="face_fear")
        face_angry = st.slider("Angry expression intensity", 0.0, 1.0, 0.05, key="face_angry")
        face_happy = st.slider("Happy expression intensity", 0.0, 1.0, 0.30, key="face_happy")
        facial_tension = st.slider("Facial muscle tension", 0.0, 1.0, 0.30, key="facial_tension")

        st.markdown("**Simulated speech signals**")
        speech_stressed = st.slider("Stressed tone intensity", 0.0, 1.0, 0.15, key="speech_stressed")
        speech_anxious = st.slider("Anxious tone intensity", 0.0, 1.0, 0.15, key="speech_anxious")
        speech_sad = st.slider("Sad tone intensity", 0.0, 1.0, 0.10, key="speech_sad")
        speech_calm = st.slider("Calm tone intensity", 0.0, 1.0, 0.35, key="speech_calm")
        pause_ratio = st.slider("Pause ratio while speaking", 0.0, 0.8, 0.15, key="pause_ratio")

    st.divider()

    if st.button("🔍 Generate My Mental Well-Being Score", type="primary", use_container_width=True):
        qresult = score_questionnaire(responses)

        face_probs = {"sad": face_sad, "fear": face_fear, "angry": face_angry, "happy": face_happy}
        speech_probs = {"stressed": speech_stressed, "anxious": speech_anxious,
                         "sad": speech_sad, "calm": speech_calm}

        score, category = rule_based_score(
            phq9_total=qresult.phq9_total,
            gad7_total=qresult.gad7_total,
            face_probs=face_probs,
            speech_probs=speech_probs,
            facial_muscle_tension=facial_tension,
            pause_ratio=pause_ratio,
        )

        high_flags = [k for k, v in responses.items() if v >= 2]
        recs = generate_recommendations(
            risk_category=category,
            questionnaire_flags=high_flags,
            facial_muscle_tension=facial_tension,
            speech_pause_ratio=pause_ratio,
            crisis_flag=qresult.crisis_flag,
        )

        st.session_state.history.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "score": score,
            "category": category,
            "phq9_total": qresult.phq9_total,
            "gad7_total": qresult.gad7_total,
        })

        st.markdown("### Your Results")
        r1, r2 = st.columns([1, 2])
        with r1:
            color = BADGE_COLORS.get(category, "#495057")
            st.markdown(f"""
            <div class="t2m-card">
                <div class="t2m-score">{score}</div>
                <span class="t2m-badge" style="background:{color}">{category}</span>
                <p style="margin-top:0.8rem;color:#555;">PHQ-9: {qresult.phq9_total}/27 ({qresult.depression_severity})
                <br>GAD-7: {qresult.gad7_total}/21 ({qresult.anxiety_severity})</p>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown("#### Personalized Recommendations")
            for rec in recs:
                st.markdown(f"- {rec}")

        if qresult.crisis_flag:
            st.error(
                "⚠️ Your responses indicate you may be having thoughts of self-harm. "
                "Please reach out to a mental health professional or a crisis helpline "
                "right away — support is available and you don't have to go through this alone."
            )

with tab_history:
    st.subheader("Mental Well-Being Score Over Time")
    if not st.session_state.history:
        st.info("No assessments yet this session. Complete an assessment in the first tab to see your trend.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.line_chart(hist_df.set_index("timestamp")["score"])
        st.dataframe(hist_df, use_container_width=True)
