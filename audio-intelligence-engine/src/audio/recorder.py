"""Backward compatibility shim for the legacy audio package."""

from __future__ import annotations

from audio_engine.recorder import AudioRecorder

__all__ = ["AudioRecorder"]
