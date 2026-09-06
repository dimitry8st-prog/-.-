import io
import mimetypes
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from telegram import Message


MESSAGE_CHUNK_SIZE = 3900


class MediaValidationError(ValueError):
    """Raised when Telegram media does not satisfy configured limits."""


class MediaConversionError(RuntimeError):
    """Raised when FFmpeg cannot produce an audio track."""


@dataclass(frozen=True)
class MediaFileData:
    file_id: str
    file_unique_id: str
    filename: str
    file_size: Optional[int]
    kind: Literal["video", "audio"]


def _safe_filename(original_name: str, fallback: str, mime_type: str) -> str:
    if original_name:
        safe_name = re.sub(r"[^\w.\-]", "_", original_name)
        if Path(safe_name).suffix:
            return safe_name
        extension = mimetypes.guess_extension(mime_type) or Path(fallback).suffix
        return f"{safe_name}{extension}"
    return fallback


def extract_media_file_data(message: Message) -> Optional[MediaFileData]:
    if message.video:
        mime = message.video.mime_type or "video/mp4"
        extension = mimetypes.guess_extension(mime) or ".mp4"
        return MediaFileData(
            file_id=message.video.file_id,
            file_unique_id=message.video.file_unique_id,
            filename=f"{message.video.file_unique_id}{extension}",
            file_size=message.video.file_size,
            kind="video",
        )

    if message.document and (message.document.mime_type or "").startswith("video/"):
        mime = message.document.mime_type or "video/mp4"
        original_name = message.document.file_name or ""
        fallback_name = f"{message.document.file_unique_id}.mp4"
        return MediaFileData(
            file_id=message.document.file_id,
            file_unique_id=message.document.file_unique_id,
            filename=_safe_filename(original_name, fallback_name, mime),
            file_size=message.document.file_size,
            kind="video",
        )

    if message.audio:
        mime = message.audio.mime_type or "audio/mpeg"
        original_name = message.audio.file_name or ""
        fallback_name = f"{message.audio.file_unique_id}.mp3"
        return MediaFileData(
            file_id=message.audio.file_id,
            file_unique_id=message.audio.file_unique_id,
            filename=_safe_filename(original_name, fallback_name, mime),
            file_size=message.audio.file_size,
            kind="audio",
        )

    if message.document and (message.document.mime_type or "").startswith("audio/"):
        mime = message.document.mime_type or "audio/mpeg"
        original_name = message.document.file_name or ""
        fallback_name = f"{message.document.file_unique_id}.mp3"
        return MediaFileData(
            file_id=message.document.file_id,
            file_unique_id=message.document.file_unique_id,
            filename=_safe_filename(original_name, fallback_name, mime),
            file_size=message.document.file_size,
            kind="audio",
        )

    return None


def validate_media_size(file_size: Optional[int], max_file_size_mb: int) -> None:
    if file_size is None:
        return
    limit_bytes = max_file_size_mb * 1024 * 1024
    if file_size > limit_bytes:
        raise MediaValidationError(
            f"Размер файла превышает лимит {max_file_size_mb} МБ. "
            "Сожмите видео или отправьте более короткий фрагмент."
        )


def convert_to_speech_mp3(
    source_path: Path, output_path: Path, timeout_seconds: int = 900
) -> Path:
    """Extract and normalize the first audio track for speech recognition."""
    if not source_path.is_file():
        raise MediaConversionError("Исходный медиафайл не найден.")

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise MediaConversionError("FFmpeg не установлен в системе.") from error
    except subprocess.TimeoutExpired as error:
        raise MediaConversionError("Преобразование файла превысило лимит времени.") from error

    if result.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        details = (result.stderr or "неизвестная ошибка FFmpeg").strip().splitlines()
        last_line = details[-1][:300] if details else "неизвестная ошибка FFmpeg"
        raise MediaConversionError(
            f"Не удалось извлечь звуковую дорожку: {last_line}"
        )
    return output_path


def build_transcript_document(transcript: str, filename: str) -> io.BytesIO:
    document = io.BytesIO(transcript.encode("utf-8"))
    safe_stem = re.sub(r"[^\w.\-]", "_", Path(filename).stem) or "meeting"
    document.name = f"{safe_stem}_transcript.txt"
    document.seek(0)
    return document


def split_for_telegram(text: str, chunk_size: int = MESSAGE_CHUNK_SIZE) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, chunk_size)
        if split_at == -1:
            split_at = chunk_size
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return chunks
