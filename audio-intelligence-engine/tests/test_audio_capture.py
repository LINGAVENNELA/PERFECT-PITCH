import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_engine.recorder import AudioRecorder


class AudioRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = AudioRecorder()

    def test_loads_mono_wav_file(self) -> None:
        mono_audio = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mono.wav"
            sf.write(str(path), mono_audio, samplerate=16000)

            audio, sample_rate = self.recorder.load_file(path)

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.shape[0], mono_audio.shape[0])
        self.assertEqual(audio.dtype, np.float32)
        np.testing.assert_allclose(
            audio,
            mono_audio,
            atol=1e-4,
        )

    def test_loads_stereo_wav_file_and_converts_to_mono(self) -> None:
        left = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
        right = np.linspace(0.5, -0.5, 1600, dtype=np.float32)
        stereo_audio = np.stack([left, right], axis=1)
        expected_mono = np.mean(stereo_audio, axis=1).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stereo.wav"
            sf.write(str(path), stereo_audio, samplerate=16000)

            audio, sample_rate = self.recorder.load_file(path)

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.shape[0], expected_mono.shape[0])
        np.testing.assert_allclose(
            audio,
            expected_mono,
            atol=1e-4,
        )

    def test_missing_wav_file_raises_file_not_found_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.recorder.load_file("does_not_exist.wav")

    @patch("audio_engine.recorder.sd.query_devices", return_value=[])
    def test_record_raises_when_microphone_is_unavailable(self, _) -> None:
        with self.assertRaises(RuntimeError) as context:
            self.recorder.record(duration=0.1)

        self.assertIn("no input audio device available", str(context.exception).lower())


if __name__ == "__main__":
    unittest.main()
