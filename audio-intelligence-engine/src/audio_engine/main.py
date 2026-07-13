"""Main entry point for the Audio Intelligence Engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .note_mapper import NoteMapper
from .pitch_detector.pyin import PyinPitchDetector
from .pitch_detector.yin import YinPitchDetector
from .preprocessor import AudioPreprocessor
from .recorder import AudioRecorder
from .scorer import Scorer


class AudioEngine:
    """Coordinate audio capture, preprocessing, pitch detection, note mapping and scoring."""

    def __init__(self) -> None:
        self.recorder = AudioRecorder()

        self.primary_detector = YinPitchDetector()
        self.fallback_detector = PyinPitchDetector()

        self.note_mapper = NoteMapper()
        self.scorer = Scorer()

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        expected_frequency: float | None = None,
    ) -> dict:

        preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
        )

        processed = preprocessor.preprocess_for_detection(
            audio,
        )

        frequency = self.primary_detector.detect(
            processed,
            sample_rate,
        )

        if frequency is None:
            frequency = self.fallback_detector.detect(
                processed,
                sample_rate,
            )

        if frequency is None:
            return {
                "raw_audio": audio,
                "processed_audio": processed,
                "frequency_hz": None,
                "note": None,
                "score": None,
            }

        note = self.note_mapper.map_frequency(
            frequency,
        )

        score = self.scorer.score_note(
            note["cent_error"],
        )

        return {
            "raw_audio": audio,
            "processed_audio": processed,
            "frequency_hz": frequency,
            "note": note,
            "score": score,
        }


def main() -> int:

    project_root = Path(__file__).resolve().parents[2]

    sample_path = (
        project_root
        / "data"
        / "raw"
        / "a4_test.wav"
    )

    print("Audio Engine Initialized")

    recorder = AudioRecorder()

    audio, sample_rate = recorder.load_file(
        sample_path,
    )

    duration = len(audio) / sample_rate

    print(f"Sample Rate: {sample_rate}")
    print(f"Duration: {duration:.3f} s")
    print(f"Samples: {len(audio)}")

    engine = AudioEngine()

    result = engine.process_audio(
        audio,
        sample_rate,
    )

    print()
    print("Detection Result")
    print("----------------")

    if result["note"] is None:
        print("No pitch detected.")
    else:
        print(f"Frequency : {result['frequency_hz']:.2f} Hz")
        print(f"Note      : {result['note']['note_name']}")
        print(f"Score     : {result['score']['pitch_score']}")
        print(f"Rating    : {result['score']['rating']}")
        print(f"Feedback  : {result['score']['feedback']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())