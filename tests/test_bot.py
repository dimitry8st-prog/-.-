import unittest

from bot import (
    AssemblyAIError,
    MediaValidationError,
    format_diarized_transcript,
    format_timestamp,
    split_for_telegram,
    validate_media_size,
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


if __name__ == "__main__":
    unittest.main()
