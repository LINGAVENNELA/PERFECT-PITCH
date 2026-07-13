"""Audio Intelligence Engine package."""

from .main import AudioEngine
from .note_mapper import NoteMapper
from .preprocessor import AudioPreprocessor
from .recorder import AudioRecorder
from .scorer import Scorer

from .pitch_detector.detector import PitchDetector
from .pitch_detector.yin import YinPitchDetector
from .pitch_detector.pyin import PyinPitchDetector

__all__ = [
    "AudioEngine",
    "AudioRecorder",
    "AudioPreprocessor",
    "PitchDetector",
    "YinPitchDetector",
    "PyinPitchDetector",
    "NoteMapper",
    "Scorer",
]