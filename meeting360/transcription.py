import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"


class AssemblyAIError(RuntimeError):
    """Raised when AssemblyAI returns an error state."""


@dataclass(frozen=True)
class TranscriptionResult:
    transcript: str
    speaker_count: int
    duration_ms: Optional[int]


def upload_to_assemblyai(file_path: Path, api_key: str) -> str:
    headers = {"authorization": api_key}
    with file_path.open("rb") as audio_file:
        response = requests.post(
            ASSEMBLYAI_UPLOAD_URL,
            headers=headers,
            data=audio_file,
            timeout=1200,
        )
    response.raise_for_status()

    upload_url = response.json().get("upload_url")
    if not upload_url:
        raise AssemblyAIError("AssemblyAI did not return upload_url.")
    return upload_url


def create_assemblyai_transcript(
    upload_url: str, api_key: str, speech_models: tuple[str, ...]
) -> str:
    headers = {"authorization": api_key, "content-type": "application/json"}
    payload = {
        "audio_url": upload_url,
        "speech_models": list(speech_models),
        "language_detection": True,
        "speaker_labels": True,
    }
    response = requests.post(
        ASSEMBLYAI_TRANSCRIPT_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    transcript_id = response.json().get("id")
    if not transcript_id:
        raise AssemblyAIError("AssemblyAI did not return transcript id.")
    return transcript_id


def format_timestamp(milliseconds: int) -> str:
    total_seconds = max(0, milliseconds // 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_diarized_transcript(payload: dict) -> TranscriptionResult:
    utterances = payload.get("utterances") or []
    lines: list[str] = []
    speakers: set[str] = set()

    for utterance in utterances:
        text = str(utterance.get("text") or "").strip()
        if not text:
            continue
        speaker = str(utterance.get("speaker") or "?")
        start_ms = int(utterance.get("start") or 0)
        speakers.add(speaker)
        lines.append(f"[{format_timestamp(start_ms)}] Спикер {speaker}: {text}")

    if not lines:
        plain_text = str(payload.get("text") or "").strip()
        if not plain_text:
            raise AssemblyAIError("AssemblyAI returned empty transcript.")
        lines.append(f"[00:00] Спикер ?: {plain_text}")

    duration_value = payload.get("audio_duration")
    duration_ms = int(float(duration_value) * 1000) if duration_value is not None else None
    return TranscriptionResult(
        transcript="\n".join(lines),
        speaker_count=len(speakers),
        duration_ms=duration_ms,
    )


def poll_assemblyai_transcript(
    transcript_id: str, api_key: str, timeout_seconds: int = 3600
) -> TranscriptionResult:
    headers = {"authorization": api_key}
    deadline = time.monotonic() + timeout_seconds
    endpoint = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}"

    while time.monotonic() < deadline:
        response = requests.get(endpoint, headers=headers, timeout=60)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")

        if status == "completed":
            return format_diarized_transcript(payload)
        if status == "error":
            raise AssemblyAIError(payload.get("error") or "Unknown AssemblyAI error.")
        time.sleep(3)

    raise TimeoutError("Timed out while waiting for AssemblyAI transcription.")


def transcribe_audio(
    file_path: Path, api_key: str, speech_models: tuple[str, ...]
) -> TranscriptionResult:
    upload_url = upload_to_assemblyai(file_path=file_path, api_key=api_key)
    transcript_id = create_assemblyai_transcript(
        upload_url=upload_url,
        api_key=api_key,
        speech_models=speech_models,
    )
    return poll_assemblyai_transcript(transcript_id=transcript_id, api_key=api_key)
