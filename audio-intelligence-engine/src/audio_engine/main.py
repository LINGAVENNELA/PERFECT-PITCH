"""Main entry point for the audio engine package."""

from __future__ import annotations

from pathlib import Path
import sys

try:
    from .recorder import AudioRecorder
except ImportError:  # pragma: no cover
    from audio_engine.recorder import AudioRecorder


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
