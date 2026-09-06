import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from meeting360.audit import analyze_with_openai
from meeting360.format_telegram import format_audit_for_telegram
from meeting360.config import Settings
from meeting360.database import save_audit_result
from meeting360.media import (
    MediaConversionError,
    MediaValidationError,
    build_transcript_document,
    convert_to_speech_mp3,
    extract_media_file_data,
    split_for_telegram,
    validate_media_size,
)
from meeting360.transcription import transcribe_audio


async def _update_progress(progress_message, text: str) -> None:
    """A failed status edit must not cancel media processing."""
    try:
        await progress_message.edit_text(text)
    except TelegramError as error:
        logging.warning("Could not update progress message: %s", error)


async def _delete_progress(progress_message) -> None:
    try:
        await progress_message.delete()
    except TelegramError as error:
        logging.warning("Could not delete progress message: %s", error)


async def start_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(
        "Встреча 360 принимает видео или аудио встречи.\n"
        "Я извлеку звуковую дорожку, разделю реплики участников, подготовлю "
        "транскрипцию с тайм-кодами и оценю встречу по 11 критериям."
    )


async def help_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(
        "Как использовать:\n"
        "1) Отправьте видео или аудиофайл.\n"
        "2) Дождитесь подготовки аудио, транскрибации и анализа.\n"
        "3) Получите транскрипт, итоговую оценку и рекомендации.\n\n"
        "Совет: используйте запись с разборчивой речью и минимальным фоновым шумом."
    )


async def _save_result(
    settings: Settings,
    message,
    file_data,
    transcript: Optional[str],
    analysis: Optional[str],
    speaker_count: Optional[int],
    duration_ms: Optional[int],
    status: str,
    error_message: Optional[str],
) -> None:
    user_id = message.from_user.id if message.from_user else None
    username = message.from_user.username if message.from_user else None
    try:
        await asyncio.to_thread(
            save_audit_result,
            settings.database_url,
            message.chat_id,
            user_id,
            username,
            file_data.file_id,
            file_data.file_unique_id,
            file_data.filename,
            transcript,
            analysis,
            speaker_count,
            duration_ms,
            status,
            error_message,
        )
    except Exception as db_error:
        logging.exception("Failed to save audit status '%s': %s", status, db_error)


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    message = update.message
    file_data = extract_media_file_data(message)
    if not file_data:
        await message.reply_text("Не вижу подходящий файл. Отправьте видео или аудио.")
        return

    settings: Settings = context.application.bot_data["settings"]
    audit_prompt: str = context.application.bot_data["audit_prompt"]
    transcript: Optional[str] = None
    analysis: Optional[str] = None
    speaker_count: Optional[int] = None
    duration_ms: Optional[int] = None

    progress_message = await message.reply_text("Файл получен. Проверяю...")
    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)

    try:
        validate_media_size(file_data.file_size, settings.max_file_size_mb)
        with tempfile.TemporaryDirectory(prefix="meeting360_") as tmp_dir:
            source_path = Path(tmp_dir) / file_data.filename
            audio_path = Path(tmp_dir) / "speech.mp3"

            await _update_progress(progress_message, "Скачиваю файл из Telegram...")
            telegram_file = await context.bot.get_file(file_data.file_id)
            await telegram_file.download_to_drive(custom_path=str(source_path))
            validate_media_size(source_path.stat().st_size, settings.max_file_size_mb)

            await _update_progress(
                progress_message, "Извлекаю и подготавливаю звуковую дорожку..."
            )
            await asyncio.to_thread(convert_to_speech_mp3, source_path, audio_path)

            await _update_progress(
                progress_message, "Разделяю говорящих и создаю транскрипцию..."
            )
            transcription = await asyncio.to_thread(
                transcribe_audio,
                audio_path,
                settings.assemblyai_api_key,
                settings.assemblyai_speech_models,
            )
            transcript = transcription.transcript
            speaker_count = transcription.speaker_count
            duration_ms = transcription.duration_ms

        await _update_progress(
            progress_message, "Транскрипция готова. Анализирую встречу..."
        )
        analysis = format_audit_for_telegram(
            await asyncio.to_thread(
                analyze_with_openai, transcript, audit_prompt, settings
            )
        )

        await _update_progress(progress_message, "Анализ готов. Отправляю результат...")
        await message.reply_document(
            document=build_transcript_document(transcript, file_data.filename),
            caption=(
                "Транскрипция готова. "
                f"Распознано участников: {speaker_count or 'не определено'}."
            ),
        )
        for chunk in split_for_telegram(analysis):
            await message.reply_text(chunk)

        await _save_result(
            settings,
            message,
            file_data,
            transcript,
            analysis,
            speaker_count,
            duration_ms,
            "success",
            None,
        )
        await _delete_progress(progress_message)

    except MediaValidationError as error:
        await _update_progress(progress_message, str(error))
    except MediaConversionError as error:
        logging.warning("Media conversion failed for %s: %s", file_data.file_unique_id, error)
        await _save_result(
            settings,
            message,
            file_data,
            transcript,
            analysis,
            speaker_count,
            duration_ms,
            "failed",
            str(error),
        )
        await _update_progress(
            progress_message,
            "Не удалось извлечь речь из файла. Проверьте, что запись содержит "
            "звуковую дорожку и не повреждена.",
        )
    except Exception as error:
        logging.exception("Failed to process media file: %s", error)
        await _save_result(
            settings,
            message,
            file_data,
            transcript,
            analysis,
            speaker_count,
            duration_ms,
            "failed",
            str(error),
        )
        await _update_progress(
            progress_message,
            "Не получилось обработать файл. Попробуйте ещё раз. "
            "Если ошибка повторится, администратору нужно проверить журнал контейнера.",
        )


async def handle_unsupported_message(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text("Отправьте видео или аудиофайл для анализа.")


def build_application(settings: Settings, audit_prompt: str) -> Application:
    app = Application.builder().token(settings.telegram_bot_token).build()
    app.bot_data["settings"] = settings
    app.bot_data["audit_prompt"] = audit_prompt

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(
        MessageHandler(
            filters.VIDEO | filters.Document.VIDEO | filters.AUDIO | filters.Document.AUDIO,
            handle_media,
        )
    )
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_unsupported_message))
    return app
