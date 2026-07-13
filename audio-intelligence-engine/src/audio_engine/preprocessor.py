"""
Audio preprocessing module for the Pitch Perfect Audio Intelligence Engine.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class AudioPreprocessor:
    """
    Preprocesses raw audio for pitch detection.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        frame_size: int = 2048,
        hop_length: int = 512,
        silence_threshold: float = 0.01,
    ) -> None:

        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.silence_threshold = silence_threshold
        self.window = np.hanning(frame_size).astype(np.float32)

    def preprocess(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        signal = self.validate_audio(audio)

        signal = self.to_mono(signal)

        signal = self.remove_dc_offset(signal)

        signal = self.normalize(signal)

        signal = self.trim_silence(signal)

        frames = self.frame_audio(signal)

        frames = self.apply_hann_window(frames)

        return frames.astype(np.float32)

    def preprocess_for_detection(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Preprocess audio for pitch detection (returns 1D signal, no framing)."""

        signal = self.validate_audio(audio)

        signal = self.to_mono(signal)

        signal = self.remove_dc_offset(signal)

        signal = self.normalize(signal)

        signal = self.trim_silence(signal)

        return signal.astype(np.float32)

    def validate_audio(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        if audio is None:
            raise ValueError("Audio cannot be None.")

        signal = np.asarray(audio, dtype=np.float32)

        if signal.size == 0:
            raise ValueError("Audio array is empty.")

        if signal.ndim > 2:
            raise ValueError(
                "Audio must be mono or stereo."
            )

        return signal

    def to_mono(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        if audio.ndim == 1:
            return audio

        return np.mean(audio, axis=1)

    def remove_dc_offset(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        mean = np.mean(audio)

        return audio - mean

    def normalize(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        peak = np.max(np.abs(audio))

        if peak <= 0.0:
            return audio

        return audio / peak

    def trim_silence(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        mask = np.abs(audio) >= self.silence_threshold

        if not np.any(mask):
            return audio

        start = np.argmax(mask)

        end = len(mask) - np.argmax(mask[::-1])

        return audio[start:end]

    def frame_audio(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        if audio.size == 0:
            raise ValueError("Cannot frame empty audio.")

        if audio.size < self.frame_size:
            padding = self.frame_size - audio.size

            audio = np.pad(
                audio,
                (0, padding),
                mode="constant",
            )

        total_samples = audio.shape[0]

        number_of_frames = (
            1
            + (total_samples - self.frame_size)
            // self.hop_length
        )

        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(
                number_of_frames,
                self.frame_size,
            ),
            strides=(
                audio.strides[0] * self.hop_length,
                audio.strides[0],
            ),
            writeable=False,
        ).copy()

        return frames.astype(np.float32)

    def apply_hann_window(
        self,
        frames: np.ndarray,
    ) -> np.ndarray:

        if frames.ndim != 2:
            raise ValueError(
                "Frames must be a 2-dimensional array."
            )

        return frames * self.window

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        frame = self.remove_dc_offset(frame)

        frame = self.normalize(frame)

        return frame * self.window

    def process_stream(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        frames = self.frame_audio(audio)

        processed_frames = np.empty_like(frames)

        for index in range(frames.shape[0]):

            processed_frames[index] = self.process_frame(
                frames[index]
            )

        return processed_frames

    def __call__(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        return self.preprocess(audio)
    def preprocess_frames(
        self,
        frames: np.ndarray,
    ) -> np.ndarray:

        if frames.ndim != 2:
            raise ValueError(
                "Frames must be a 2-dimensional array."
            )

        processed = np.empty_like(frames)

        for index in range(frames.shape[0]):

            frame = frames[index]

            frame = self.remove_dc_offset(frame)

            frame = self.normalize(frame)

            processed[index] = frame * self.window

        return processed

    def get_frame_duration(self) -> float:

        return self.frame_size / self.sample_rate

    def get_hop_duration(self) -> float:

        return self.hop_length / self.sample_rate

    def get_frame_count(
        self,
        audio: np.ndarray,
    ) -> int:

        audio = self.validate_audio(audio)

        audio = self.to_mono(audio)

        if audio.size < self.frame_size:
            return 1

        return (
            1
            + (audio.size - self.frame_size)
            // self.hop_length
        )

    def reset(self) -> None:

        self.window = np.hanning(
            self.frame_size
        ).astype(np.float32)

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"sample_rate={self.sample_rate}, "
            f"frame_size={self.frame_size}, "
            f"hop_length={self.hop_length}, "
            f"silence_threshold={self.silence_threshold})"
        )