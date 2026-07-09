"""Audio playback utilities."""

from __future__ import annotations

from pathlib import Path

import sounddevice as sd
import soundfile as sf


class AudioPlayer:
    """Simple WAV playback helper for debugging recordings."""

    def play(self, file_path: str) -> None:
        path = Path(file_path)
        audio, sample_rate = sf.read(path)
        sd.play(audio, sample_rate)
        sd.wait()
