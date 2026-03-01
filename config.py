from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")
CODEX_CMD = os.getenv("CODEX_CMD", "codex").strip() or "codex"


def resolve_default_workdir() -> str:
    configured = (os.getenv("CODEX_WORKDIR") or "").strip()
    launch_dir = os.path.abspath(os.getcwd())
    if not configured:
        return launch_dir

    candidate = os.path.abspath(os.path.expanduser(configured))
    if os.path.isdir(candidate):
        return candidate

    LOGGER.warning(
        "CODEX_WORKDIR is not a valid directory (%s); using launch directory (%s)",
        candidate,
        launch_dir,
    )
    return launch_dir


DEFAULT_WORKDIR = resolve_default_workdir()

VOICE_TRANSCRIPTION = os.getenv("VOICE_TRANSCRIPTION", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")

MAX_TELEGRAM_CHUNK = 4096
PROGRESS_MAX_ACTIONS = 5
PROGRESS_COMMAND_WIDTH = 200
PROGRESS_EDIT_MIN_INTERVAL_S = 1.0
PROGRESS_TICK_S = 3.0
OPENAI_AUDIO_MAX_BYTES = 25 * 1024 * 1024
RUN_COMMAND_TIMEOUT_S = 30.0
RUN_OUTPUT_MAX_CHARS = 8_000
RESULT_MAX_CHANGED_FILES = 8
SESSIONS_LIST_LIMIT = 20
SESSIONS_FETCH_LIMIT = 50
SESSIONS_BUTTON_LIMIT = 6

STATE_DIR = os.path.join(os.path.expanduser("~"), ".djinn")
STATE_PATH = os.path.join(STATE_DIR, "state.json")
PROJECTS_PATH = os.path.join(STATE_DIR, "projects.json")

SYSTEM_HINT = (
    "You are Djinn, a Telegram bridge running through Codex App Server. "
    "Tools are enabled; use them directly for commands and edits. "
    "Answer directly and concisely. "
    "You have filesystem access and can run commands when needed."
)
