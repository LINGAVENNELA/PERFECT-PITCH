"""Placeholder YIN-style pitch detector for the audio engine."""

from __future__ import annotations

import numpy as np

from .detector import PitchDetector


class YinPitchDetector(PitchDetector):
    """Placeholder YIN pitch detector."""

    def detect(self, audio: np.ndarray, sample_rate: int) -> float | None:
        data = np.asarray(audio, dtype=np.float32)
        if data.size == 0:
            return None

        return None
