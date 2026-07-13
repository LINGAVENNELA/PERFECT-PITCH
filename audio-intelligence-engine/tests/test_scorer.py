import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_engine.scorer import Scorer


@pytest.fixture
def scorer() -> Scorer:
    return Scorer()


def test_perfect_score(scorer: Scorer) -> None:

    result = scorer.score_note(0.0)

    assert result["pitch_score"] == 100
    assert result["rating"] == "Perfect"


def test_excellent_score(scorer: Scorer) -> None:

    result = scorer.score_note(8.0)

    assert result["pitch_score"] == 90
    assert result["rating"] == "Excellent"


def test_good_score(scorer: Scorer) -> None:

    result = scorer.score_note(18.0)

    assert result["pitch_score"] == 80
    assert result["rating"] == "Good"


def test_fair_score(scorer: Scorer) -> None:

    result = scorer.score_note(30.0)

    assert result["pitch_score"] == 65
    assert result["rating"] == "Fair"


def test_needs_practice_score(scorer: Scorer) -> None:

    result = scorer.score_note(45.0)

    assert result["pitch_score"] == 50
    assert result["rating"] == "Needs Practice"


def test_retry_score(scorer: Scorer) -> None:

    result = scorer.score_note(70.0)

    assert result["pitch_score"] == 25
    assert result["rating"] == "Retry"


def test_negative_cent_error(scorer: Scorer) -> None:

    result = scorer.score_note(-8.0)

    assert result["pitch_score"] == 90
    assert "raise your pitch" in result["feedback"]


def test_positive_cent_error(scorer: Scorer) -> None:

    result = scorer.score_note(8.0)

    assert result["pitch_score"] == 90
    assert "lower your pitch" in result["feedback"]


def test_perfect_feedback(scorer: Scorer) -> None:

    result = scorer.score_note(0.0)

    assert "matched the note perfectly" in result["feedback"]


def test_cent_error_is_preserved(scorer: Scorer) -> None:

    result = scorer.score_note(-13.2)

    assert result["cent_error"] == pytest.approx(
        -13.2,
        abs=1e-6,
    )