"""Note mapping scaffolding for the audio engine."""

from __future__ import annotations


def frequency_to_note_name(frequency_hz: float | None) -> str | None:
    """Convert a detected frequency to a musical note name.

    This is a minimal placeholder implementation that supports a small set of
    stable reference frequencies for development and testing.
    """
    if frequency_hz is None:
        return None

    reference_notes = {
        440.0: "A4",
        261.63: "C4",
        523.25: "C5",
        220.0: "A3",
    }

    for reference_frequency, note_name in reference_notes.items():
        if abs(frequency_hz - reference_frequency) <= 1.0:
            return note_name

    return None
