"""Audio recorder module for the Pitch Perfect audio engine."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    """Audio input module for WAV files and live microphone recording."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, dtype: str = "float32") -> None:
        """Create a new AudioRecorder.

        Args:
            sample_rate: Default sample rate for live recordings.
            channels: Channel count for live microphone recording.
            dtype: NumPy dtype for returned audio arrays.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype

    def load_file(self, path: str | Path) -> tuple[np.ndarray, int]:
        """Load a WAV file and return mono audio plus its sample rate.

        Stereo WAV files are converted to mono by averaging all channels.

        Args:
            path: Path to a WAV file.

        Returns:
            A tuple containing a 1D float32 NumPy array and its sample rate.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If the WAV file cannot be read.
        """
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"WAV file not found: {source}")

        try:
            audio, sample_rate = sf.read(source, dtype="float32")
        except Exception as exc:
            raise RuntimeError(f"Unable to load WAV file: {source}") from exc

        audio_data = np.asarray(audio, dtype=np.float32)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        return audio_data.reshape(-1).astype(np.float32), int(sample_rate)

    def record(
        self,
        duration: float,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        device: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        """Record live audio from a microphone.

        Args:
            duration: Recording duration in seconds.
            sample_rate: Optional override of the default sample rate.
            channels: Optional override of the channel count.
            device: Optional sounddevice device index.

        Returns:
            A tuple containing a 1D float32 NumPy array and the sample rate.

        Raises:
            RuntimeError: If no microphone input device is available.
            ValueError: If duration is not positive.
        """
        if duration <= 0:
            raise ValueError("duration must be a positive number")

        rate = sample_rate or self.sample_rate
        ch = channels or self.channels

        if not self._has_input_device():
            raise RuntimeError(
                "No input audio device available. Live microphone recording is unavailable in this environment."
            )

        try:
            frames = int(rate * duration)
            audio = sd.rec(frames, samplerate=rate, channels=ch, dtype=self.dtype, device=device)
            sd.wait()
        except Exception as exc:
            raise RuntimeError(
                "Unable to record from the microphone. Live recording is unavailable in this environment."
            ) from exc

        audio_data = np.asarray(audio, dtype=np.float32)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        return audio_data.reshape(-1), rate

    def _has_input_device(self) -> bool:
        try:
            devices = sd.query_devices()
        except Exception:
            return False

        if isinstance(devices, dict):
            devices = [devices]

        return any(
            isinstance(device, dict) and device.get("max_input_channels", 0) > 0
            for device in devices
        )


def _run_demo() -> None:
    parser = argparse.ArgumentParser(description="Demo for AudioRecorder.")
    parser.add_argument("--wav", default="example.wav", help="Path to a WAV file to load.")
    parser.add_argument("--duration", type=float, default=1.0, help="Seconds to record from the microphone.")
    args = parser.parse_args()

    recorder = AudioRecorder()

    print("=== WAV file load demo ===")
    try:
        audio, sample_rate = recorder.load_file(args.wav)
        print(f"Loaded {audio.shape[0]} samples at {sample_rate} Hz from '{args.wav}'.")
    except FileNotFoundError as exc:
        print(exc)
    except RuntimeError as exc:
        print(exc)

    print("\n=== Microphone record demo ===")
    try:
        audio, sample_rate = recorder.record(duration=args.duration)
        print(f"Recorded {audio.shape[0]} samples at {sample_rate} Hz.")
    except RuntimeError as exc:
        print(exc)


if __name__ == "__main__":
    _run_demo()
