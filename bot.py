import logging
from pathlib import Path

from dotenv import load_dotenv

from meeting360.audit import load_audit_prompt
from meeting360.config import Settings
from meeting360.database import init_database_with_retry
from meeting360.telegram_app import build_application


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    settings = Settings.from_env()
    init_database_with_retry(settings.database_url)
    audit_prompt = load_audit_prompt(Path(__file__).with_name("prompt-audit.md"))

    application = build_application(settings, audit_prompt)
    logging.info("Встреча 360 запущена.")
    application.run_polling(allowed_updates=None)


if __name__ == "__main__":
    main()
