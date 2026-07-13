"""Audio engine pipeline package for recording, preprocessing, pitch detection, and note mapping.

The audio_engine package provides an integrated pipeline for:
  - Recording audio from input devices
  - Preprocessing raw audio (normalization, silence trimming)
  - Detecting pitch using YIN or PYIN algorithms
  - Mapping detected frequencies to musical notes
  - Scoring pitch accuracy against expected frequencies

Key Classes:
  - AudioEngine: Main coordinator for the full pipeline
  - AudioRecorder: Record audio with optional file persistence
  - AudioPreprocessor: Normalize and trim audio before analysis
  - YinPitchDetector: YIN-style autocorrelation-based pitch detector
  - PyinPitchDetector: FFT-based fallback pitch detector

Example:
    from audio_engine import AudioEngine

    engine = AudioEngine()
    result = engine.process_audio(audio_samples, sample_rate=16000)
    print(result["note"])  # e.g., "A4"
"""

try:
    from .main import AudioEngine
except ImportError:  # pragma: no cover
    AudioEngine = None

from .recorder import AudioRecorder
from .preprocessor import AudioPreprocessor
from .pitch_detector.detector import PitchDetector
from .pitch_detector.yin import YinPitchDetector
from .pitch_detector.pyin import PyinPitchDetector
from .note_mapper import frequency_to_note_name
from .scorer import score_frequency_match

__all__ = [
    "AudioEngine",
    "AudioRecorder",
    "AudioPreprocessor",
    "PitchDetector",
    "YinPitchDetector",
    "PyinPitchDetector",
    "frequency_to_note_name",
    "score_frequency_match",
]
