import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_engine.note_mapper import NoteMapper


@pytest.fixture
def mapper() -> NoteMapper:
    return NoteMapper()


def test_frequency_to_midi_a4(mapper: NoteMapper) -> None:
    assert mapper.frequency_to_midi(440.0) == 69


def test_frequency_to_midi_c4(mapper: NoteMapper) -> None:
    assert mapper.frequency_to_midi(261.63) == 60


def test_frequency_to_midi_c5(mapper: NoteMapper) -> None:
    assert mapper.frequency_to_midi(523.25) == 72


def test_invalid_frequency(mapper: NoteMapper) -> None:
    with pytest.raises(ValueError):
        mapper.frequency_to_midi(0.0)


def test_maps_a4(mapper: NoteMapper) -> None:
    result = mapper.map_frequency(440.0)

    assert result["note"] == "A"
    assert result["octave"] == 4
    assert result["note_name"] == "A4"
    assert result["midi"] == 69
    assert result["is_in_tune"] is True


def test_maps_c4(mapper: NoteMapper) -> None:
    result = mapper.map_frequency(261.63)

    assert result["note_name"] == "C4"
    assert result["midi"] == 60


def test_maps_c5(mapper: NoteMapper) -> None:
    result = mapper.map_frequency(523.25)

    assert result["note_name"] == "C5"
    assert result["midi"] == 72


def test_reference_frequency(mapper: NoteMapper) -> None:
    result = mapper.map_frequency(440.0)

    assert result["reference_frequency"] == pytest.approx(
        440.0,
        abs=1e-6,
    )


def test_zero_cent_error(mapper: NoteMapper) -> None:
    result = mapper.map_frequency(440.0)

    assert result["cent_error"] == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_sharp_note(mapper: NoteMapper) -> None:
    result = mapper.map_frequency(445.0)

    assert result["cent_error"] > 0
    assert result["is_in_tune"] is False


def test_flat_note(mapper: NoteMapper) -> None:
    result = mapper.map_frequency(435.0)

    assert result["cent_error"] < 0
    assert result["is_in_tune"] is False