import logging
import time
from typing import Optional

import psycopg2


DB_CONNECTION_RETRIES = 30
DB_CONNECTION_RETRY_DELAY_SECONDS = 2


def init_database(database_url: str) -> None:
    with psycopg2.connect(database_url, connect_timeout=10) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS video_audits (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    chat_id BIGINT NOT NULL,
                    user_id BIGINT NULL,
                    username TEXT NULL,
                    file_id TEXT NOT NULL,
                    file_unique_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    transcript TEXT NULL,
                    analysis TEXT NULL,
                    speaker_count INTEGER NULL,
                    duration_ms BIGINT NULL,
                    prompt_version TEXT NOT NULL DEFAULT 'meeting360-v1',
                    status TEXT NOT NULL,
                    error_message TEXT NULL
                );
                """
            )
            cursor.execute(
                """
                ALTER TABLE video_audits
                    ADD COLUMN IF NOT EXISTS speaker_count INTEGER NULL,
                    ADD COLUMN IF NOT EXISTS duration_ms BIGINT NULL,
                    ADD COLUMN IF NOT EXISTS prompt_version TEXT NOT NULL DEFAULT 'meeting360-v1';
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_video_audits_chat_id
                ON video_audits (chat_id);
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_video_audits_created_at
                ON video_audits (created_at DESC);
                """
            )
        connection.commit()


def init_database_with_retry(database_url: str) -> None:
    for attempt in range(1, DB_CONNECTION_RETRIES + 1):
        try:
            init_database(database_url)
            logging.info("Postgres is ready. Database schema initialized.")
            return
        except Exception as error:
            logging.warning(
                "Postgres is not ready yet (attempt %s/%s): %s",
                attempt,
                DB_CONNECTION_RETRIES,
                error,
            )
            time.sleep(DB_CONNECTION_RETRY_DELAY_SECONDS)

    raise RuntimeError("Could not initialize Postgres schema after multiple attempts.")


def save_audit_result(
    database_url: str,
    chat_id: int,
    user_id: Optional[int],
    username: Optional[str],
    file_id: str,
    file_unique_id: str,
    filename: str,
    transcript: Optional[str],
    analysis: Optional[str],
    speaker_count: Optional[int],
    duration_ms: Optional[int],
    status: str,
    error_message: Optional[str],
) -> None:
    with psycopg2.connect(database_url, connect_timeout=10) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO video_audits (
                    chat_id, user_id, username, file_id, file_unique_id, filename,
                    transcript, analysis, speaker_count, duration_ms, status, error_message
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    chat_id,
                    user_id,
                    username,
                    file_id,
                    file_unique_id,
                    filename,
                    transcript,
                    analysis,
                    speaker_count,
                    duration_ms,
                    status,
                    error_message,
                ),
            )
        connection.commit()
