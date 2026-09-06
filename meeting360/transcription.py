import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"


class AssemblyAIError(RuntimeError):
    """Raised when AssemblyAI returns an error state."""


@dataclass(frozen=True)
class TranscriptionResult:
    transcript: str
    speaker_count: int
    duration_ms: Optional[int]


def _polling_session() -> requests.Session:
    """Retry safe GET requests used while waiting for a transcript."""
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def upload_to_assemblyai(file_path: Path, api_key: str) -> str:
    headers = {"authorization": api_key}
    with file_path.open("rb") as audio_file:
        try:
            response = requests.post(
                ASSEMBLYAI_UPLOAD_URL,
                headers=headers,
                data=audio_file,
                timeout=1200,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            raise AssemblyAIError("Не удалось загрузить аудио в AssemblyAI.") from error

    if not isinstance(payload, dict):
        raise AssemblyAIError("AssemblyAI returned an invalid upload response.")
    upload_url = payload.get("upload_url")
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
    try:
        response = requests.post(
            ASSEMBLYAI_TRANSCRIPT_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        response_payload = response.json()
    except requests.RequestException as error:
        raise AssemblyAIError("AssemblyAI не принял запрос на транскрибацию.") from error

    if not isinstance(response_payload, dict):
        raise AssemblyAIError("AssemblyAI returned an invalid transcript response.")
    transcript_id = response_payload.get("id")
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

    with _polling_session() as session:
        while time.monotonic() < deadline:
            try:
                response = session.get(endpoint, headers=headers, timeout=60)
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as error:
                raise AssemblyAIError(
                    "Потеряно соединение с AssemblyAI во время транскрибации."
                ) from error
            if not isinstance(payload, dict):
                raise AssemblyAIError("AssemblyAI returned an invalid status response.")
            status = payload.get("status")

            if status == "completed":
                return format_diarized_transcript(payload)
            if status == "error":
                raise AssemblyAIError(payload.get("error") or "Unknown AssemblyAI error.")
            if status not in {"queued", "processing"}:
                raise AssemblyAIError(f"AssemblyAI returned unexpected status: {status!r}.")
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
