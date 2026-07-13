import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_engine.pitch_detector.yin import YinPitchDetector


SAMPLE_RATE = 22050
FRAME_SIZE = 2048
TOLERANCE = 2.0


def generate_sine_wave(
    frequency: float,
    duration: float = FRAME_SIZE / SAMPLE_RATE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:

    t = np.arange(
        int(sample_rate * duration)
    ) / sample_rate

    return np.sin(
        2 * np.pi * frequency * t
    ).astype(np.float32)
def test_detects_a3() -> None:

    detector = YinPitchDetector()

    audio = generate_sine_wave(220.0)

    detected = detector.detect(
        audio,
        SAMPLE_RATE,
    )

    assert detected is not None

    assert abs(detected - 220.0) < TOLERANCE


def test_detects_a4() -> None:

    detector = YinPitchDetector()

    audio = generate_sine_wave(440.0)

    detected = detector.detect(
        audio,
        SAMPLE_RATE,
    )

    assert detected is not None

    assert abs(detected - 440.0) < TOLERANCE


def test_detects_c4() -> None:

    detector = YinPitchDetector()

    audio = generate_sine_wave(261.63)

    detected = detector.detect(
        audio,
        SAMPLE_RATE,
    )

    assert detected is not None

    assert abs(detected - 261.63) < TOLERANCE


def test_detects_c5() -> None:

    detector = YinPitchDetector()

    audio = generate_sine_wave(523.25)

    detected = detector.detect(
        audio,
        SAMPLE_RATE,
    )

    assert detected is not None

    assert abs(detected - 523.25) < TOLERANCE

def test_returns_none_for_silence() -> None:

    detector = YinPitchDetector()

    audio = np.zeros(
        FRAME_SIZE,
        dtype=np.float32,
    )

    detected = detector.detect(
        audio,
        SAMPLE_RATE,
    )

    assert detected is None


def test_rejects_empty_audio() -> None:

    detector = YinPitchDetector()

    with pytest.raises(ValueError):

        detector.detect(
            np.array(
                [],
                dtype=np.float32,
            ),
            SAMPLE_RATE,
        )


def test_rejects_stereo_audio() -> None:

    detector = YinPitchDetector()

    audio = np.zeros(
        (
            FRAME_SIZE,
            2,
        ),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):

        detector.detect(
            audio,
            SAMPLE_RATE,
        )


def test_rejects_frame_that_is_too_short() -> None:

    detector = YinPitchDetector()

    audio = np.ones(
        32,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):

        detector.detect(
            audio,
            SAMPLE_RATE,
        )


def test_same_signal_produces_same_result() -> None:

    detector = YinPitchDetector()

    audio = generate_sine_wave(
        440.0,
    )

    result_1 = detector.detect(
        audio,
        SAMPLE_RATE,
    )

    result_2 = detector.detect(
        audio,
        SAMPLE_RATE,
    )

    assert result_1 is not None

    assert result_2 is not None

    assert result_1 == pytest.approx(
        result_2,
        abs=1e-6,
    )


def test_noise_does_not_produce_valid_pitch() -> None:

    detector = YinPitchDetector()

    rng = np.random.default_rng(
        seed=42,
    )

    noise = rng.normal(
        0.0,
        1.0,
        FRAME_SIZE,
    ).astype(np.float32)

    detected = detector.detect(
        noise,
        SAMPLE_RATE,
    )

    assert (
        detected is None
        or detector.min_frequency
        <= detected
        <= detector.max_frequency
    )