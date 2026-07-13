"""Pitch scoring utilities for the Pitch Perfect Audio Engine."""

from __future__ import annotations


def score_frequency_match(detected_hz: float | None, expected_hz: float | None) -> float:
    """Score the quality of a detected frequency match.

    This function is a placeholder for the scorer component.
    """
    return 0.0


class Scorer:

    def calculate_pitch_score(
        self,
        cent_error: float,
    ) -> int:

        error = abs(cent_error)

        if error <= 5:
            return 100

        if error <= 10:
            return 90

        if error <= 20:
            return 80

        if error <= 35:
            return 65

        if error <= 50:
            return 50

        return 25

    def determine_rating(
        self,
        score: int,
    ) -> str:

        if score == 100:
            return "Perfect"

        if score == 90:
            return "Excellent"

        if score == 80:
            return "Good"

        if score == 65:
            return "Fair"

        if score == 50:
            return "Needs Practice"

        return "Retry"

    def generate_feedback(
        self,
        cent_error: float,
        rating: str,
    ) -> str:

        if rating == "Perfect":
            return "Excellent! You matched the note perfectly."

        direction = (
            "lower your pitch slightly"
            if cent_error > 0
            else "raise your pitch slightly"
        )

        if rating == "Excellent":
            return (
                f"Very close. Try to {direction}."
            )

        if rating == "Good":
            return (
                f"Good attempt. Try to {direction}."
            )

        if rating == "Fair":
            return (
                f"You're getting there. Try to {direction}."
            )

        if rating == "Needs Practice":
            return (
                f"Keep practicing this note. Try to {direction}."
            )

        return (
            f"Sing the note again and try to {direction}."
        )

    def score_note(
        self,
        cent_error: float,
    ) -> dict:

        score = self.calculate_pitch_score(
            cent_error,
        )

        rating = self.determine_rating(
            score,
        )

        feedback = self.generate_feedback(
            cent_error,
            rating,
        )

        return {
            "pitch_score": score,
            "rating": rating,
            "feedback": feedback,
            "cent_error": cent_error,
        }