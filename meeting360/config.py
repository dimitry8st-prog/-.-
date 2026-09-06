import os
from dataclasses import dataclass


DEFAULT_MAX_FILE_SIZE_MB = 20


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    assemblyai_api_key: str
    openai_api_key: str
    database_url: str
    openai_model: str = "gpt-4.1-mini"
    assemblyai_speech_models: tuple[str, ...] = ("universal-3-pro", "universal-2")
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB

    @classmethod
    def from_env(cls) -> "Settings":
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        database_url = os.getenv("DATABASE_URL", "").strip()
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"

        speech_models_raw = os.getenv(
            "ASSEMBLYAI_SPEECH_MODELS", "universal-3-pro,universal-2"
        )
        assemblyai_speech_models = tuple(
            item.strip() for item in speech_models_raw.split(",") if item.strip()
        )
        if not assemblyai_speech_models:
            raise ValueError("ASSEMBLYAI_SPEECH_MODELS must contain at least one model")

        try:
            max_file_size_mb = int(
                os.getenv("MAX_FILE_SIZE_MB", str(DEFAULT_MAX_FILE_SIZE_MB))
            )
        except ValueError as error:
            raise ValueError("MAX_FILE_SIZE_MB must be an integer") from error
        if max_file_size_mb <= 0:
            raise ValueError("MAX_FILE_SIZE_MB must be greater than zero")

        required = {
            "TELEGRAM_BOT_TOKEN": telegram_bot_token,
            "ASSEMBLYAI_API_KEY": assemblyai_api_key,
            "OPENAI_API_KEY": openai_api_key,
            "DATABASE_URL": database_url,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        return cls(
            telegram_bot_token=telegram_bot_token,
            assemblyai_api_key=assemblyai_api_key,
            openai_api_key=openai_api_key,
            database_url=database_url,
            openai_model=openai_model,
            assemblyai_speech_models=assemblyai_speech_models,
            max_file_size_mb=max_file_size_mb,
        )
