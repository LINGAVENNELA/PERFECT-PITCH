import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_engine.main import AudioEngine


def test_pipeline_detects_note_from_synthetic_signal():
    engine = AudioEngine()
    sample_rate = 16000
    duration = 0.25
    t = np.arange(int(sample_rate * duration)) / sample_rate
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    result = engine.process_audio(audio, sample_rate=sample_rate, expected_frequency=440.0)

    assert result["frequency_hz"] is not None
    assert result["note"] == "A4"
    assert result["score"] >= 0.0
