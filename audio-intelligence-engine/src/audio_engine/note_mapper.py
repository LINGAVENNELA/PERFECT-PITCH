"""Music note mapping utilities for the Pitch Perfect Audio Engine."""

from __future__ import annotations

import math


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


class NoteMapper:

    NOTE_NAMES = (
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    )

    A4_FREQUENCY = 440.0
    A4_MIDI = 69

    def frequency_to_midi(
        self,
        frequency: float,
    ) -> int:

        if frequency <= 0.0:
            raise ValueError(
                "Frequency must be positive."
            )

        midi = (
            12
            * math.log2(
                frequency / self.A4_FREQUENCY
            )
        ) + self.A4_MIDI

        return int(round(midi))

    def midi_to_frequency(
        self,
        midi: int,
    ) -> float:

        return self.A4_FREQUENCY * (
            2 ** (
                (midi - self.A4_MIDI)
                / 12
            )
        )
    def midi_to_note_name(
        self,
        midi: int,
    ) -> tuple[str, int, str]:

        note = self.NOTE_NAMES[
            midi % 12
        ]

        octave = (
            midi // 12
        ) - 1

        note_name = (
            f"{note}{octave}"
        )

        return (
            note,
            octave,
            note_name,
        )

    def calculate_cent_error(
        self,
        frequency: float,
        reference_frequency: float,
    ) -> float:

        return (
            1200
            * math.log2(
                frequency
                / reference_frequency
            )
        )

    def map_frequency(
        self,
        frequency: float,
        tolerance_cents: float = 10.0,
    ) -> dict:

        midi = self.frequency_to_midi(
            frequency,
        )

        reference_frequency = (
            self.midi_to_frequency(
                midi,
            )
        )

        note, octave, note_name = (
            self.midi_to_note_name(
                midi,
            )
        )

        cent_error = (
            self.calculate_cent_error(
                frequency,
                reference_frequency,
            )
        )

        return {
            "frequency": frequency,
            "midi": midi,
            "note": note,
            "octave": octave,
            "note_name": note_name,
            "reference_frequency": reference_frequency,
            "cent_error": cent_error,
            "is_in_tune": abs(
                cent_error
            ) <= tolerance_cents,
        }