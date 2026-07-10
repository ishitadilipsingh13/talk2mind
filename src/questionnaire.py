"""
Talk2Mind - Mental Health Questionnaire Module
================================================

Implements a lightweight, non-clinical self-report questionnaire inspired
by widely-used validated screening tools:
  - PHQ-9 style items (depression-related indicators)
  - GAD-7 style items (anxiety-related indicators)

IMPORTANT: This is a self-awareness / educational tool, NOT a diagnostic
instrument. It does not replace PHQ-9/GAD-7 administered by a clinician,
and Talk2Mind should always display a disclaimer + point users toward
professional help when scores are high.

Each question is answered on a 0-3 scale:
    0 = Not at all
    1 = Several days
    2 = More than half the days
    3 = Nearly every day
"""

from dataclasses import dataclass, field
from typing import Dict, List

PHQ9_QUESTIONS = {
    "little_interest_or_pleasure": "Little interest or pleasure in doing things",
    "feeling_down_or_hopeless": "Feeling down, depressed, or hopeless",
    "sleep_trouble": "Trouble falling/staying asleep, or sleeping too much",
    "low_energy_fatigue": "Feeling tired or having little energy",
    "appetite_changes": "Poor appetite or overeating",
    "feeling_bad_about_self": "Feeling bad about yourself, or that you are a failure",
    "concentration_trouble": "Trouble concentrating on things",
    "moving_speaking_slowly_or_restless": "Moving/speaking slowly, or being fidgety/restless",
    "thoughts_of_self_harm": "Thoughts that you would be better off not being here",
}

GAD7_QUESTIONS = {
    "feeling_nervous_anxious": "Feeling nervous, anxious, or on edge",
    "cant_stop_worrying": "Not being able to stop or control worrying",
    "worrying_too_much": "Worrying too much about different things",
    "trouble_relaxing": "Trouble relaxing",
    "restlessness": "Being so restless that it's hard to sit still",
    "easily_annoyed_irritable": "Becoming easily annoyed or irritable",
    "feeling_afraid": "Feeling afraid as if something awful might happen",
}

CRISIS_QUESTION_KEY = "thoughts_of_self_harm"


@dataclass
class QuestionnaireResult:
    responses: Dict[str, int]
    phq9_total: int
    gad7_total: int
    depression_severity: str
    anxiety_severity: str
    crisis_flag: bool


def _severity_from_phq9(total: int) -> str:
    if total <= 4:
        return "Minimal"
    elif total <= 9:
        return "Mild"
    elif total <= 14:
        return "Moderate"
    elif total <= 19:
        return "Moderately Severe"
    return "Severe"


def _severity_from_gad7(total: int) -> str:
    if total <= 4:
        return "Minimal"
    elif total <= 9:
        return "Mild"
    elif total <= 14:
        return "Moderate"
    return "Severe"


def score_questionnaire(responses: Dict[str, int]) -> QuestionnaireResult:
    """
    responses: dict mapping question_key -> int score (0-3) for all
    PHQ9_QUESTIONS and GAD7_QUESTIONS keys.
    """
    missing = [k for k in list(PHQ9_QUESTIONS) + list(GAD7_QUESTIONS) if k not in responses]
    if missing:
        raise ValueError(f"Missing responses for: {missing}")

    phq9_total = sum(responses[k] for k in PHQ9_QUESTIONS)
    gad7_total = sum(responses[k] for k in GAD7_QUESTIONS)

    crisis_flag = responses.get(CRISIS_QUESTION_KEY, 0) >= 1

    return QuestionnaireResult(
        responses=responses,
        phq9_total=phq9_total,
        gad7_total=gad7_total,
        depression_severity=_severity_from_phq9(phq9_total),
        anxiety_severity=_severity_from_gad7(gad7_total),
        crisis_flag=crisis_flag,
    )


def print_questionnaire_cli():
    """Simple CLI demo to answer the questionnaire interactively."""
    print("=== Talk2Mind Self-Assessment Questionnaire ===")
    print("Answer each item: 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day\n")
    responses = {}
    print("--- Depression-related items ---")
    for key, text in PHQ9_QUESTIONS.items():
        while True:
            try:
                val = int(input(f"{text}: "))
                if val in (0, 1, 2, 3):
                    responses[key] = val
                    break
            except ValueError:
                pass
            print("Please enter a number 0-3.")

    print("\n--- Anxiety-related items ---")
    for key, text in GAD7_QUESTIONS.items():
        while True:
            try:
                val = int(input(f"{text}: "))
                if val in (0, 1, 2, 3):
                    responses[key] = val
                    break
            except ValueError:
                pass
            print("Please enter a number 0-3.")

    result = score_questionnaire(responses)
    print("\n=== Result ===")
    print(f"PHQ-9 total: {result.phq9_total}/27 -> {result.depression_severity}")
    print(f"GAD-7 total: {result.gad7_total}/21 -> {result.anxiety_severity}")
    if result.crisis_flag:
        print("\n⚠️  Your response indicates you may be having thoughts of self-harm.")
        print("Please reach out to a mental health professional or a crisis helpline "
              "right away. You deserve support, and you don't have to go through this alone.")
    return result


if __name__ == "__main__":
    print_questionnaire_cli()
