"""Base pitch detector interface for the audio engine."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PitchDetector(ABC):
    """Abstract base class for pitch detector implementations."""

    @abstractmethod
    def detect(self, audio: np.ndarray, sample_rate: int) -> float | None:
        """Return the estimated frequency in hertz, or None when unavailable."""
        raise NotImplementedError
