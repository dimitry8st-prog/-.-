import math
import os
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from telegram.error import TelegramError

from meeting360.audit import analyze_with_openai
from meeting360.config import Settings
from meeting360.media import (
    MediaConversionError,
    MediaValidationError,
    convert_to_speech_mp3,
    split_for_telegram,
    validate_media_size,
)
from meeting360.transcription import (
    AssemblyAIError,
    create_assemblyai_transcript,
    format_diarized_transcript,
    format_timestamp,
)
from meeting360.telegram_app import _update_progress


class SettingsTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "TELEGRAM_BOT_TOKEN": "telegram-test",
            "ASSEMBLYAI_API_KEY": "assembly-test",
            "OPENAI_API_KEY": "openai-test",
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
        },
        clear=True,
    )
    def test_uses_current_assemblyai_models_by_default(self):
        settings = Settings.from_env()
        self.assertEqual(
            settings.assemblyai_speech_models,
            ("universal-3-5-pro", "universal-2"),
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_reports_missing_required_settings(self):
        with self.assertRaisesRegex(ValueError, "TELEGRAM_BOT_TOKEN"):
            Settings.from_env()


class AssemblyAIRequestTests(unittest.TestCase):
    @patch("meeting360.transcription.requests.post")
    def test_submits_diarization_and_current_models(self, post: Mock):
        response = post.return_value
        response.json.return_value = {"id": "transcript-id"}

        result = create_assemblyai_transcript(
            "https://example.test/audio.mp3",
            "test-key",
            ("universal-3-5-pro", "universal-2"),
        )

        self.assertEqual(result, "transcript-id")
        payload = post.call_args.kwargs["json"]
        self.assertTrue(payload["speaker_labels"])
        self.assertTrue(payload["language_detection"])
        self.assertEqual(
            payload["speech_models"], ["universal-3-5-pro", "universal-2"]
        )


class OpenAITests(unittest.TestCase):
    @patch("meeting360.audit.OpenAI")
    def test_rejects_empty_message_content(self, openai: Mock):
        client = openai.return_value
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )
        settings = Settings(
            telegram_bot_token="telegram-test",
            assemblyai_api_key="assembly-test",
            openai_api_key="openai-test",
            database_url="postgresql://user:pass@localhost/db",
        )

        with self.assertRaisesRegex(RuntimeError, "empty analysis"):
            analyze_with_openai("тест", "инструкция", settings)


class TelegramProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_progress_edit_failure_does_not_cancel_processing(self):
        progress_message = SimpleNamespace(
            edit_text=AsyncMock(side_effect=TelegramError("temporary failure"))
        )

        await _update_progress(progress_message, "Обрабатываю...")

        progress_message.edit_text.assert_awaited_once_with("Обрабатываю...")


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
