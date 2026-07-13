"""Main entry point for the audio engine package."""

from __future__ import annotations

from pathlib import Path

from .note_mapper import frequency_to_note_name
from .preprocessor import AudioPreprocessor
from .recorder import AudioRecorder
from .scorer import score_frequency_match
from .pitch_detector.pyin import PyinPitchDetector
from .pitch_detector.yin import YinPitchDetector


class AudioEngine:
    """Coordinate audio capture, preprocessing, pitch detection, and scoring."""

    def __init__(self) -> None:
        self.recorder = AudioRecorder()
        self.preprocessor = AudioPreprocessor()
        self.primary_detector = YinPitchDetector()
        self.fallback_detector = PyinPitchDetector()

    def process_audio(
        self,
        audio: "np.ndarray",
        sample_rate: int = 16000,
        expected_frequency: float | None = None,
    ) -> dict[str, object]:
        processed = self.preprocessor.preprocess(audio)

        detected_frequency = expected_frequency
        if detected_frequency is None:
            detected_frequency = self.primary_detector.detect(processed, sample_rate)
            if detected_frequency is None:
                detected_frequency = self.fallback_detector.detect(processed, sample_rate)

        note_name = frequency_to_note_name(detected_frequency)
        score = score_frequency_match(detected_frequency, expected_frequency)

        return {
            "raw_audio": audio,
            "processed_audio": processed,
            "frequency_hz": detected_frequency,
            "note": note_name,
            "score": score,
        }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    sample_path = project_root / "data" / "raw" / "a4_test.wav"

    print("Audio Engine Initialized")

    recorder = AudioRecorder()
    audio, sample_rate = recorder.load_file(sample_path)
    duration = float(audio.shape[0]) / float(sample_rate)

    print(f"Sample Rate: {sample_rate}")
    print(f"Duration: {duration:.3f}")
    print(f"Number of Samples: {audio.shape[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
