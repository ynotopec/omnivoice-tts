import asyncio
import base64
import io
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from app.server import SAMPLE_RATE, SpeechRequest, _decode_audio, create_speech, encode_audio, reference_input


def wav_bytes(source, sample_rate=8000):
    encoded = io.BytesIO()
    sf.write(encoded, source, sample_rate, format="WAV")
    return encoded.getvalue()


class AudioPipelineTests(unittest.TestCase):
    def test_decode_audio_uses_content_instead_of_extension(self):
        source = np.linspace(-0.5, 0.5, 8000, dtype=np.float32)

        decoded, sample_rate = _decode_audio(wav_bytes(source))

        self.assertEqual(sample_rate, 8000)
        np.testing.assert_allclose(decoded, source, atol=4e-5)

    def test_reference_input_uses_supported_waveform_tuple(self):
        source = np.sin(2 * np.pi * 200 * np.arange(8000) / 8000).astype(np.float32)

        waveform, sample_rate = reference_input(wav_bytes(source))

        self.assertIsInstance(waveform, torch.Tensor)
        self.assertEqual(waveform.ndim, 1)
        self.assertEqual(sample_rate, 8000)

    def test_silent_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "silent"):
            reference_input(wav_bytes(np.zeros(8000, dtype=np.float32)))

    def test_encode_wav_converts_tensor_like_audio_to_pcm16(self):
        waveform = np.array([0.0, np.nan, 2.0, -2.0], dtype=np.float64)

        buffer, content_type = encode_audio(waveform, "wav")
        decoded, sample_rate = sf.read(buffer)

        self.assertEqual(content_type, "audio/wav")
        self.assertEqual(sample_rate, SAMPLE_RATE)
        self.assertTrue(np.isfinite(decoded).all())
        self.assertLessEqual(np.max(decoded), 1.0)

    def test_speech_uses_omnivoice_native_reference_options(self):
        class FakeModel:
            sampling_rate = SAMPLE_RATE

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [np.zeros(100, dtype=np.float32)]

        model = FakeModel()
        reference = wav_bytes(np.full(8000, 0.1, dtype=np.float32))
        request = SpeechRequest(
            input="Olá mundo",
            voice="clone",
            language="Portuguese",
            speed=1.25,
            denoise=True,
            reference_audio=base64.b64encode(reference).decode(),
            reference_text="Olá.",
            response_format="wav",
        )

        with patch("app.server.load_model", return_value=model):
            asyncio.run(create_speech(request))

        self.assertEqual(model.kwargs["language"], "Portuguese")
        self.assertEqual(model.kwargs["speed"], 1.25)
        self.assertTrue(model.kwargs["denoise"])
        self.assertTrue(model.kwargs["preprocess_prompt"])
        self.assertIsInstance(model.kwargs["ref_audio"], tuple)
        self.assertEqual(model.kwargs["ref_text"], "Olá.")


if __name__ == "__main__":
    unittest.main()
