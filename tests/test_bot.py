import math
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from meeting360.media import (
    MediaConversionError,
    MediaValidationError,
    convert_to_speech_mp3,
    split_for_telegram,
    validate_media_size,
)
from meeting360.transcription import (
    AssemblyAIError,
    format_diarized_transcript,
    format_timestamp,
)


class TimestampTests(unittest.TestCase):
    def test_formats_minutes_and_hours(self):
        self.assertEqual(format_timestamp(65_000), "01:05")
        self.assertEqual(format_timestamp(3_665_000), "01:01:05")


class TranscriptTests(unittest.TestCase):
    def test_formats_speakers_and_timestamps(self):
        result = format_diarized_transcript(
            {
                "audio_duration": 12.5,
                "utterances": [
                    {"speaker": "A", "start": 0, "text": "Добрый день."},
                    {"speaker": "B", "start": 5_500, "text": "Начнём."},
                ],
            }
        )
        self.assertEqual(result.speaker_count, 2)
        self.assertEqual(result.duration_ms, 12_500)
        self.assertIn("[00:05] Спикер B: Начнём.", result.transcript)

    def test_falls_back_to_plain_text(self):
        result = format_diarized_transcript({"text": "Один голос", "utterances": []})
        self.assertEqual(result.speaker_count, 0)
        self.assertEqual(result.transcript, "[00:00] Спикер ?: Один голос")

    def test_rejects_empty_transcript(self):
        with self.assertRaises(AssemblyAIError):
            format_diarized_transcript({})


class ValidationTests(unittest.TestCase):
    def test_file_size_limit(self):
        validate_media_size(20 * 1024 * 1024, 20)
        with self.assertRaises(MediaValidationError):
            validate_media_size(20 * 1024 * 1024 + 1, 20)

    def test_telegram_split_preserves_content(self):
        text = "первая строка\nвторая строка\nтретья строка"
        chunks = split_for_telegram(text, chunk_size=20)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("\n".join(chunks), text)


class MediaConversionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_converts_wav_to_speech_mp3(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "sample.wav"
            output = Path(tmp_dir) / "speech.mp3"
            sample_rate = 8_000

            with wave.open(str(source), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                frames = b"".join(
                    struct.pack(
                        "<h", int(8_000 * math.sin(2 * math.pi * 440 * i / sample_rate))
                    )
                    for i in range(sample_rate)
                )
                wav_file.writeframes(frames)

            result = convert_to_speech_mp3(source, output)

            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_rejects_missing_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(MediaConversionError):
                convert_to_speech_mp3(
                    Path(tmp_dir) / "missing.mp4", Path(tmp_dir) / "speech.mp3"
                )


if __name__ == "__main__":
    unittest.main()
