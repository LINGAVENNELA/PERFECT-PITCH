"""Pitch detection implementations for the audio engine.

Provides multiple pitch detection algorithms:
  - YinPitchDetector: Autocorrelation-based YIN algorithm
  - PyinPitchDetector: FFT-based fallback detector
  - PitchDetector: Abstract base class for custom detectors
"""

from .detector import PitchDetector
from .yin import YinPitchDetector
from .pyin import PyinPitchDetector

__all__ = ["PitchDetector", "YinPitchDetector", "PyinPitchDetector"]
