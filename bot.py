from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shlex
import textwrap
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from codex import (
    CodexApprovalRequest,
    CodexAppServerClient,
    CodexProgressEvent,
    CodexThreadSummary,
    CodexTurnResult,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)
# Avoid leaking bot token in HTTP URL logs from lower-level clients.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")

CODEX_CMD = os.getenv("CODEX_CMD", "codex")
CODEX_APPROVAL_POLICY = os.getenv("CODEX_APPROVAL_POLICY", "on-request")


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
OPENAI_TRANSCRIPTION_MODEL = os.getenv(
    "OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"
)

MAX_TELEGRAM_CHUNK = 4096
# Keep the progress panel compact so frequent edits remain readable in Telegram.
PROGRESS_MAX_ACTIONS = 5
# Long shell commands are truncated to avoid edit failures and noisy UI updates.
PROGRESS_COMMAND_WIDTH = 200
PROGRESS_EDIT_MIN_INTERVAL_S = 1.0
PROGRESS_TICK_S = 3.0
OPENAI_AUDIO_MAX_BYTES = 25 * 1024 * 1024
# Keep direct `/run` usage responsive on mobile and avoid hanging forever.
RUN_COMMAND_TIMEOUT_S = 30.0
# Cap shell output so a single command cannot flood the Telegram chat.
RUN_OUTPUT_MAX_CHARS = 8_000
# Keep final "changed files" recap short enough for quick mobile scanning.
RESULT_MAX_CHANGED_FILES = 8
# Number of session rows shown in `/sessions` results.
SESSIONS_LIST_LIMIT = 20
# Fetch more than we display so newest-per-workdir dedupe still has enough candidates.
SESSIONS_FETCH_LIMIT = 50
# Keep inline session picker compact to avoid oversized callback keyboards.
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

APPROVAL_ACCEPT = "accept"
APPROVAL_DECLINE = "decline"


@dataclass
class PendingApproval:
    request: CodexApprovalRequest
    future: asyncio.Future[str]
    message_id: int | None
    chat_id: int


@dataclass
class QueuedTurn:
    user_text: str
    chat_id: int
    reply_to_message_id: int | None


@dataclass
class ProgressState:
    started_at: float
    actions: dict[str, str] = field(default_factory=dict)
    changed_files: dict[str, str] = field(default_factory=dict)


@dataclass
class ProjectState:
    path: str
    thread_id: str | None = None
    pin: str | None = None


@dataclass
class BotState:
    workdir: str = DEFAULT_WORKDIR
    thread_id: str | None = None
    project_map: dict[str, ProjectState] = field(default_factory=dict)
    active_project: str | None = None
    pin: str | None = None
    run_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_shell_proc: asyncio.subprocess.Process | None = None
    shell_cancel_requested: bool = False
    queued_turn: QueuedTurn | None = None
    last_session_ids: list[str] = field(default_factory=list)
    last_turn_result: str | None = None
    codex: CodexAppServerClient | None = None
    pending_approvals: dict[str, PendingApproval] = field(default_factory=dict)


# Generic persistence

def _load_json_dict(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        LOGGER.warning("failed to parse JSON at %s: %s", path, exc)
        return {}
    except OSError as exc:
        LOGGER.warning("failed to read %s: %s", path, exc)
        return {}

    if not isinstance(payload, dict):
        LOGGER.warning("invalid JSON payload at %s: expected object", path)
        return {}
    return payload


def _save_json_dict(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        os.replace(tmp_path, path)
    except OSError as exc:
        LOGGER.warning("failed to save %s: %s", path, exc)


def load_runtime_state() -> dict[str, str]:
    payload = _load_json_dict(STATE_PATH)
    state: dict[str, str] = {}

    workdir = payload.get("workdir")
    if isinstance(workdir, str) and workdir and os.path.isdir(workdir):
        state["workdir"] = workdir

    thread_id = payload.get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        state["thread_id"] = thread_id

    pin = payload.get("pin")
    if isinstance(pin, str) and pin:
        state["pin"] = pin

    active_project = payload.get("active_project")
    if isinstance(active_project, str) and active_project:
        state["active_project"] = active_project

    return state


def save_runtime_state(state: BotState) -> None:
    payload: dict[str, Any] = {
        "workdir": state.workdir,
    }
    if state.thread_id:
        payload["thread_id"] = state.thread_id
    if state.pin:
        payload["pin"] = state.pin
    if state.active_project:
        payload["active_project"] = state.active_project
    _save_json_dict(STATE_PATH, payload)


def _coerce_project_state(raw: Any) -> ProjectState | None:
    if isinstance(raw, str):
        path = raw.strip()
        if not path:
            return None
        return ProjectState(path=path)

    if not isinstance(raw, dict):
        return None

    path = raw.get("path")
    if not isinstance(path, str) or not path:
        return None

    thread_id = raw.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        thread_id = None

    pin = raw.get("pin")
    if not isinstance(pin, str) or not pin:
        pin = None

    return ProjectState(path=path, thread_id=thread_id, pin=pin)


def load_projects() -> dict[str, ProjectState]:
    payload = _load_json_dict(PROJECTS_PATH)
    cleaned: dict[str, ProjectState] = {}
    migrated = False
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            continue
        project = _coerce_project_state(value)
        if project is None:
            continue
        cleaned[key] = project
        if isinstance(value, str):
            migrated = True

    if migrated:
        save_projects(cleaned)

    return cleaned


def save_projects(projects: dict[str, ProjectState]) -> None:
    payload: dict[str, dict[str, Any]] = {}
    for name, project in projects.items():
        if not isinstance(name, str) or not name:
            continue
        payload[name] = {
            "path": project.path,
            "thread_id": project.thread_id,
            "pin": project.pin,
        }
    _save_json_dict(PROJECTS_PATH, payload)


def sync_active_project_state(state: BotState) -> None:
    if not state.active_project:
        return
    project = state.project_map.get(state.active_project)
    if project is None:
        return
    project.path = state.workdir
    project.thread_id = state.thread_id
    project.pin = state.pin


def persist_state(state: BotState) -> None:
    sync_active_project_state(state)
    save_runtime_state(state)
    save_projects(state.project_map)


def restore_project_state(state: BotState, name: str) -> bool:
    project = state.project_map.get(name)
    if project is None:
        return False
    if not os.path.isdir(project.path):
        return False

    state.active_project = name
    state.workdir = project.path
    state.thread_id = project.thread_id
    state.pin = project.pin
    return True


# Formatting

def format_elapsed(elapsed_s: float) -> str:
    total = max(0, int(elapsed_s))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def shorten(text: str, width: int | None) -> str:
    if width is None:
        return text
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    return textwrap.shorten(text, width=width, placeholder="...")


def sanitize_code(text: str) -> str:
    return text.replace("`", "'")


def _format_change_summary(changes: list[dict[str, str]]) -> str:
    if not changes:
        return "files changed"
    return f"files changed ({len(changes)})"


def _normalize_file_changes(changes: Any) -> list[dict[str, str]]:
    if not isinstance(changes, list):
        return []

    normalized: list[dict[str, str]] = []
    for change in changes:
        if not isinstance(change, dict):
            continue
        path = change.get("path")
        kind = change.get("kind")
        if not isinstance(path, str) or not path:
            continue
        entry = {"path": path}
        if isinstance(kind, str) and kind:
            entry["kind"] = kind
        normalized.append(entry)
    return normalized


def format_progress_line(event: CodexProgressEvent) -> str:
    if event.phase == "completed":
        if event.ok is None or event.ok is True:
            status = "OK"
        else:
            status = "ERR"
    elif event.phase == "delta":
        status = "UPD"
    else:
        status = "RUN"

    if event.kind == "command":
        title = shorten(event.title or "", PROGRESS_COMMAND_WIDTH)
        return f"{status} cmd: `{sanitize_code(title)}`"

    if event.kind == "tool":
        title = shorten(event.title or "", PROGRESS_COMMAND_WIDTH)
        return f"{status} tool: `{sanitize_code(title)}`"

    if event.kind == "file_change":
        normalized = _normalize_file_changes(event.detail.get("changes"))
        if normalized:
            return f"{status} {_format_change_summary(normalized)}"
        return f"{status} files changed"

    title = shorten(event.title or event.kind, PROGRESS_COMMAND_WIDTH)
    return f"{status} {event.kind}: `{sanitize_code(title)}`"


def note_progress_event(progress: ProgressState, event: CodexProgressEvent) -> bool:
    if event.kind not in {"command", "tool", "file_change"}:
        return False

    if event.kind == "file_change":
        for change in _normalize_file_changes(event.detail.get("changes")):
            progress.changed_files[change["path"]] = change.get("kind", "")

    new_line = format_progress_line(event)
    old_line = progress.actions.get(event.item_id)
    if old_line == new_line:
        return False
    progress.actions[event.item_id] = new_line
    return True


def format_changed_files_summary(
    progress: ProgressState,
    *,
    max_files: int = RESULT_MAX_CHANGED_FILES,
) -> str | None:
    if not progress.changed_files:
        return None

    items = list(progress.changed_files.items())
    lines = ["Changed files:"]
    for path, kind in items[:max_files]:
        suffix = f" ({kind})" if kind else ""
        lines.append(f"- `{sanitize_code(path)}`{suffix}")

    hidden = len(items) - max_files
    if hidden > 0:
        lines.append(f"- ... and {hidden} more")

    return "\n".join(lines)


def truncate_output(text: str, *, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    return f"{clipped}\n... ({label} truncated to {max_chars} chars)"


def _short_thread_id(thread_id: str) -> str:
    if len(thread_id) <= 16:
        return thread_id
    return f"{thread_id[:8]}...{thread_id[-6:]}"


def _session_age_label(updated_at: int | None) -> str:
    if updated_at is None:
        return "unknown"
    now = int(time.time())
    delta = max(0, now - updated_at)
    return format_elapsed(delta)


def latest_sessions_by_workdir(
    sessions: Sequence[CodexThreadSummary],
    *,
    max_sessions: int,
) -> list[CodexThreadSummary]:
    deduped: list[CodexThreadSummary] = []
    seen_workdirs: set[str] = set()

    for session in sessions:
        key = session.cwd or ""
        if key in seen_workdirs:
            continue
        seen_workdirs.add(key)
        deduped.append(session)
        if len(deduped) >= max_sessions:
            break

    return deduped


def format_sessions_text(
    sessions: Sequence[CodexThreadSummary],
    *,
    scope_all: bool,
    workdir: str,
    current_thread_id: str | None,
    next_cursor: str | None = None,
    latest_per_workdir: bool = False,
) -> str:
    heading = "Recent sessions (all workdirs):" if scope_all else f"Recent sessions ({workdir}):"
    lines = [heading]
    if latest_per_workdir:
        lines.append("Showing newest session per workdir.")
    if not sessions:
        lines.append("(none found)")
    else:
        for index, session in enumerate(sessions, start=1):
            current = " (current)" if current_thread_id == session.thread_id else ""
            source = session.source or "unknown"
            cwd = session.cwd or "unknown"
            lines.append(
                f"{index}. `{sanitize_code(_short_thread_id(session.thread_id))}`"
                f" [{source}] {_session_age_label(session.updated_at)}"
                f"\n   `{sanitize_code(cwd)}`{current}"
            )

    lines.append("")
    lines.append("Use `/sessions use <n>` or `/sessions use <thread_id>` to switch.")
    if not scope_all:
        lines.append("Use `/sessions` to browse newest sessions across workdirs.")
    else:
        lines.append("Use `/sessions here` to list more sessions for the current workdir.")
    if next_cursor and not latest_per_workdir:
        lines.append("More sessions exist; run `/sessions here` again to continue browsing.")
    return "\n".join(lines)


def resolve_session_selection(state: BotState, token: str) -> tuple[str | None, str | None]:
    target = token.strip()
    if not target:
        return None, "Usage: /sessions use <n|thread_id>"

    if target.isdigit():
        index = int(target)
        if index < 1 or index > len(state.last_session_ids):
            return None, f"No session #{index}. Run /sessions first."
        return state.last_session_ids[index - 1], None

    return target, None


def _sessions_keyboard(sessions: Sequence[CodexThreadSummary]) -> InlineKeyboardMarkup | None:
    if not sessions:
        return None

    buttons: list[InlineKeyboardButton] = []
    for index, session in enumerate(sessions[:SESSIONS_BUTTON_LIMIT], start=1):
        buttons.append(
            InlineKeyboardButton(
                f"Use {index}",
                callback_data=f"sessions:{session.thread_id}",
            )
        )

    if not buttons:
        return None

    rows: list[list[InlineKeyboardButton]] = []
    for i in range(0, len(buttons), 3):
        rows.append(buttons[i : i + 3])
    return InlineKeyboardMarkup(rows)


def build_help_text() -> str:
    lines = [
        "Djinn commands:",
        "/start - verify bot is online",
        "/help - show this help",
        "/cancel - stop the active run/turn",
        "/last - resend the latest turn result",
        "/approve - approve most recent pending action",
        "/deny - deny most recent pending action",
        "/status - show active project, workdir, thread, pin",
        "/proj - list project contexts",
        "/proj <name> - switch to saved project context",
        "/proj <name> <path> - create/update and switch project context",
        "/proj rm <name> - remove a project context",
        "/sessions - list newest session per workdir",
        "/sessions here - list recent sessions for this workdir",
        "/sessions use <n|thread_id> - switch current session",
        "/pin <text> - set pinned context",
        "/pin - show current pin",
        "/unpin - clear pin",
        "/run <command> - run a shell command in current workdir",
        "/cd <path> - change workdir",
        "Advanced:",
        "/reset - clear current thread",
    ]
    return "\n".join(lines)


async def terminate_process(proc: asyncio.subprocess.Process, *, grace_s: float = 3.0) -> None:
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()


def latest_pending_approval(state: BotState) -> PendingApproval | None:
    if not state.pending_approvals:
        return None
    latest_key = next(reversed(state.pending_approvals))
    return state.pending_approvals.get(latest_key)


def render_progress(progress: ProgressState, *, label: str, pin: str | None = None) -> str:
    lines: list[str] = [f"{label} {format_elapsed(time.monotonic() - progress.started_at)}"]
    if pin:
        lines.append(f"pin: {pin}")

    body = list(progress.actions.values())
    if PROGRESS_MAX_ACTIONS > 0:
        body = body[-PROGRESS_MAX_ACTIONS:]

    if not body:
        return "\n".join(lines)

    return "\n\n".join(["\n".join(lines), "\n".join(body)])


# Telegram messaging

async def _send_once(
    application: Application,
    *,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None = None,
    disable_notification: bool = False,
    reply_markup: InlineKeyboardMarkup | None = None,
    markdown: bool = True,
):
    kwargs: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "reply_to_message_id": reply_to_message_id,
        "disable_notification": disable_notification,
    }
    if reply_markup is not None:
        kwargs["reply_markup"] = reply_markup
    if markdown:
        kwargs["parse_mode"] = ParseMode.MARKDOWN
    return await application.bot.send_message(**kwargs)


async def send_message(
    application: Application,
    text: str,
    *,
    chat_id: int,
    reply_to_message_id: int | None = None,
    disable_notification: bool = False,
    reply_markup: InlineKeyboardMarkup | None = None,
    prefer_markdown: bool = True,
) -> int | None:
    if not text:
        return None

    body = text.strip()
    if not body:
        return None

    if len(body) <= MAX_TELEGRAM_CHUNK:
        try:
            message = await _send_once(
                application,
                chat_id=chat_id,
                text=body,
                reply_to_message_id=reply_to_message_id,
                disable_notification=disable_notification,
                reply_markup=reply_markup,
                markdown=prefer_markdown,
            )
            return getattr(message, "message_id", None)
        except BadRequest:
            message = await _send_once(
                application,
                chat_id=chat_id,
                text=body,
                reply_to_message_id=reply_to_message_id,
                disable_notification=disable_notification,
                reply_markup=reply_markup,
                markdown=False,
            )
            return getattr(message, "message_id", None)

    message_id: int | None = None
    first = True
    for i in range(0, len(body), MAX_TELEGRAM_CHUNK):
        chunk = body[i : i + MAX_TELEGRAM_CHUNK]
        sent = await _send_once(
            application,
            chat_id=chat_id,
            text=chunk,
            reply_to_message_id=reply_to_message_id if first else None,
            disable_notification=disable_notification,
            reply_markup=reply_markup if first else None,
            markdown=False,
        )
        if message_id is None:
            message_id = getattr(sent, "message_id", None)
        first = False
    return message_id


async def edit_message(
    application: Application,
    *,
    chat_id: int,
    message_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
    prefer_markdown: bool = True,
) -> None:
    body = text.strip()
    if not body:
        return

    if len(body) > MAX_TELEGRAM_CHUNK:
        body = body[: MAX_TELEGRAM_CHUNK - 3] + "..."

    kwargs: dict[str, Any] = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": body,
    }
    if reply_markup is not None:
        kwargs["reply_markup"] = reply_markup

    try:
        if prefer_markdown:
            await application.bot.edit_message_text(parse_mode=ParseMode.MARKDOWN, **kwargs)
            return
        await application.bot.edit_message_text(**kwargs)
    except BadRequest:
        try:
            await application.bot.edit_message_text(**kwargs)
        except BadRequest:
            LOGGER.debug("telegram message %s could not be edited", message_id)


async def delete_message(application: Application, *, chat_id: int, message_id: int) -> None:
    try:
        await application.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except BadRequest:
        LOGGER.debug("telegram message %s could not be deleted", message_id)
    except Exception as exc:
        LOGGER.warning("failed to delete telegram message %s: %s", message_id, exc)


# Voice transcription

def _normalize_voice_filename(file_path: str | None) -> str:
    name = os.path.basename(file_path or "")
    if not name:
        return "voice.ogg"
    if name.endswith(".oga"):
        return f"{name[:-4]}.ogg"
    return name


def _transcribe_audio_sync(
    *,
    audio_bytes: bytes,
    filename: str,
    model: str,
    api_key: str,
) -> str:
    client = OpenAI(api_key=api_key)
    payload = io.BytesIO(audio_bytes)
    payload.name = filename
    response = client.audio.transcriptions.create(model=model, file=payload)

    text = getattr(response, "text", None)
    if not isinstance(text, str) and isinstance(response, dict):
        text = response.get("text")

    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("transcription returned empty text")
    return text.strip()


async def transcribe_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str | None:
    if update.message is None or update.message.voice is None or update.effective_chat is None:
        return None

    chat_id = int(update.effective_chat.id)
    reply_to = update.message.message_id

    if not VOICE_TRANSCRIPTION:
        await send_message(
            context.application,
            "Voice transcription is disabled.",
            chat_id=chat_id,
            reply_to_message_id=reply_to,
        )
        return None

    if not OPENAI_API_KEY:
        await send_message(
            context.application,
            "Voice transcription requires OPENAI_API_KEY.",
            chat_id=chat_id,
            reply_to_message_id=reply_to,
        )
        return None

    voice = update.message.voice
    if voice.file_size is not None and voice.file_size > OPENAI_AUDIO_MAX_BYTES:
        await send_message(
            context.application,
            "Voice message is too large to transcribe.",
            chat_id=chat_id,
            reply_to_message_id=reply_to,
        )
        return None

    try:
        file = await context.bot.get_file(voice.file_id)
        audio_bytes = await file.download_as_bytearray()
    except Exception as exc:
        LOGGER.warning("failed to download voice message: %s", exc)
        await send_message(
            context.application,
            "Failed to download voice message.",
            chat_id=chat_id,
            reply_to_message_id=reply_to,
        )
        return None

    filename = _normalize_voice_filename(getattr(file, "file_path", None))

    try:
        return await asyncio.to_thread(
            _transcribe_audio_sync,
            audio_bytes=bytes(audio_bytes),
            filename=filename,
            model=OPENAI_TRANSCRIPTION_MODEL,
            api_key=OPENAI_API_KEY,
        )
    except Exception as exc:
        LOGGER.warning("voice transcription failed: %s", exc)
        await send_message(
            context.application,
            f"Transcription failed: {exc}",
            chat_id=chat_id,
            reply_to_message_id=reply_to,
        )
        return None


# Auth + path helpers

def is_authorized(update: Update) -> bool:
    if update.effective_chat is None:
        LOGGER.warning("dropping update without effective_chat")
        return False

    chat_id = str(update.effective_chat.id)
    allowed_chat_id = str(TELEGRAM_CHAT_ID)
    if chat_id != allowed_chat_id:
        LOGGER.warning("unauthorized chat id %s (expected %s)", chat_id, allowed_chat_id)
        return False

    if TELEGRAM_USER_ID:
        if update.effective_user is None:
            LOGGER.warning(
                "unauthorized update from chat %s without effective_user; TELEGRAM_USER_ID is set",
                chat_id,
            )
            return False
        user_id = str(update.effective_user.id)
        allowed_user_id = str(TELEGRAM_USER_ID)
        if user_id != allowed_user_id:
            LOGGER.warning(
                "unauthorized user id %s in chat %s (expected %s)",
                user_id,
                chat_id,
                allowed_user_id,
            )
            return False

    return True


def resolve_workdir(target: str, base: str) -> str | None:
    target_path = os.path.expanduser(target)
    if not os.path.isabs(target_path):
        target_path = os.path.abspath(os.path.join(base, target_path))
    return target_path


def join_command_args(args: Sequence[object] | None) -> str:
    if not args:
        return ""
    return " ".join(str(part) for part in args).strip()


def build_prompt(user_text: str, pin: str | None = None) -> str:
    if not pin:
        return user_text
    return f"Pinned context: {pin}\n\nUser: {user_text}"


# Codex helpers

def _approval_key(request: CodexApprovalRequest) -> str:
    return str(request.request_id)


def format_approval_text(request: CodexApprovalRequest) -> str:
    lines = ["Djinn needs approval."]
    if request.kind == "command":
        lines.append("Type: command execution")
        if request.command:
            lines.append(f"Command: `{sanitize_code(request.command)}`")
    elif request.kind == "file_change":
        lines.append("Type: file change")

    if request.cwd:
        lines.append(f"CWD: `{sanitize_code(request.cwd)}`")
    if request.reason:
        lines.append(f"Reason: {request.reason}")

    lines.append("Allow this action?")
    return "\n".join(lines)


def _approval_keyboard(key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Allow", callback_data=f"approval:{key}:a"),
                InlineKeyboardButton("Deny", callback_data=f"approval:{key}:d"),
            ]
        ]
    )


async def get_codex_client(state: BotState) -> CodexAppServerClient:
    if state.codex is None:
        state.codex = CodexAppServerClient(
            codex_cmd=CODEX_CMD,
            client_name="djinn-telegram",
            client_version="0.1.0",
        )
    await state.codex.start()
    return state.codex


async def request_approval(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    state: BotState,
    request: CodexApprovalRequest,
    chat_id: int,
    reply_to_message_id: int | None,
) -> str:
    key = _approval_key(request)
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    text = format_approval_text(request)
    message_id = await send_message(
        context.application,
        text,
        chat_id=chat_id,
        reply_to_message_id=reply_to_message_id,
        reply_markup=_approval_keyboard(key),
        disable_notification=True,
    )

    state.pending_approvals[key] = PendingApproval(
        request=request,
        future=future,
        message_id=message_id,
        chat_id=chat_id,
    )

    try:
        decision = await future
    finally:
        state.pending_approvals.pop(key, None)

    if message_id is not None:
        status = "approved" if decision == APPROVAL_ACCEPT else "denied"
        await edit_message(
            context.application,
            chat_id=chat_id,
            message_id=message_id,
            text=f"{text}\n\nDecision: {status}",
            reply_markup=None,
            prefer_markdown=False,
        )

    return decision


async def progress_loop(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    state: BotState,
    chat_id: int,
    progress_message_id: int,
    progress: ProgressState,
    queue: asyncio.Queue[CodexProgressEvent],
) -> None:
    last_rendered: str | None = None
    last_edit_at = 0.0
    label = "working"

    async def maybe_render(force: bool = False) -> None:
        nonlocal last_rendered, last_edit_at
        now = time.monotonic()
        if not force and now - last_edit_at < PROGRESS_EDIT_MIN_INTERVAL_S:
            return
        rendered = render_progress(progress, label=label, pin=state.pin)
        if rendered == last_rendered:
            return
        await edit_message(
            context.application,
            chat_id=chat_id,
            message_id=progress_message_id,
            text=rendered,
        )
        last_rendered = rendered
        last_edit_at = now

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=PROGRESS_TICK_S)
            except TimeoutError:
                event = None

            if event is not None and note_progress_event(progress, event):
                label = "working"

            await maybe_render()
    except asyncio.CancelledError:
        await maybe_render(force=True)


# Command handlers

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return
    await update.message.reply_text("Connected to Djinn bridge.")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return
    await update.message.reply_text(build_help_text())


async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return

    state: BotState = context.application.bot_data["state"]
    cancelled: list[str] = []

    proc = state.active_shell_proc
    if proc is not None and proc.returncode is None:
        state.shell_cancel_requested = True
        try:
            await terminate_process(proc, grace_s=1.0)
        except Exception as exc:
            LOGGER.warning("failed to terminate active shell command: %s", exc)
        else:
            cancelled.append("shell command")

    client = state.codex
    if client is not None:
        try:
            if await client.interrupt_active_turn():
                cancelled.append("agent turn")
        except Exception as exc:
            LOGGER.warning("failed to interrupt active turn: %s", exc)

    if cancelled:
        await update.message.reply_text(
            f"Cancellation requested for {' and '.join(cancelled)}."
        )
        return

    if state.run_lock.locked():
        await update.message.reply_text(
            "No cancellable task found. The active run may have just finished."
        )
        return

    await update.message.reply_text("No active run to cancel.")


async def _resolve_latest_approval(
    *,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    decision: str,
) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return

    state: BotState = context.application.bot_data["state"]
    pending = latest_pending_approval(state)
    if pending is None:
        await update.message.reply_text("No pending approvals.")
        return

    if pending.future.done():
        await update.message.reply_text("Latest approval request is no longer active.")
        return

    pending.future.set_result(decision)
    if decision == APPROVAL_ACCEPT:
        await update.message.reply_text("Approved latest request.")
    else:
        await update.message.reply_text("Denied latest request.")


async def approve_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _resolve_latest_approval(update=update, context=context, decision=APPROVAL_ACCEPT)


async def deny_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _resolve_latest_approval(update=update, context=context, decision=APPROVAL_DECLINE)


async def last_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None or update.effective_chat is None:
        return

    state: BotState = context.application.bot_data["state"]
    if not state.last_turn_result:
        await update.message.reply_text("No turn result yet.")
        return

    await send_message(
        context.application,
        state.last_turn_result,
        chat_id=int(update.effective_chat.id),
        reply_to_message_id=update.message.message_id,
    )


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return
    state: BotState = context.application.bot_data["state"]
    state.thread_id = None
    persist_state(state)
    await update.message.reply_text("Thread reset. Next message starts a new Djinn thread.")


async def pwd_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return
    state: BotState = context.application.bot_data["state"]
    await update.message.reply_text(state.workdir)


async def cd_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return

    state: BotState = context.application.bot_data["state"]
    target = join_command_args(context.args)
    if not target:
        await update.message.reply_text("Usage: /cd <path>")
        return

    target_path = resolve_workdir(target, state.workdir)
    if target_path and os.path.isdir(target_path):
        state.workdir = target_path
        persist_state(state)
        await update.message.reply_text(state.workdir)
        return

    await update.message.reply_text(f"No such directory: {target_path or target}")


async def thread_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return

    state: BotState = context.application.bot_data["state"]
    target = join_command_args(context.args)
    if not target:
        await update.message.reply_text(f"thread: {state.thread_id or 'none'}")
        return

    state.thread_id = target
    persist_state(state)
    await update.message.reply_text(f"thread set: {state.thread_id}")


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return

    state: BotState = context.application.bot_data["state"]
    lines = [
        "Status:",
        f"- project: {state.active_project or 'none'}",
        f"- workdir: {state.workdir}",
        f"- thread: {state.thread_id or 'none'}",
        f"- pin: {state.pin or '(empty)'}",
    ]
    lines.append(f"- approvals_pending: {len(state.pending_approvals)}")

    await update.message.reply_text("\n".join(lines))


async def sessions_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None or update.effective_chat is None:
        return

    state: BotState = context.application.bot_data["state"]
    args = context.args

    if args and args[0] == "use":
        target = join_command_args(args[1:])
        thread_id, error_text = resolve_session_selection(state, target)
        if error_text:
            await update.message.reply_text(error_text)
            return
        assert thread_id is not None
        state.thread_id = thread_id
        persist_state(state)
        await update.message.reply_text(f"Session set: {state.thread_id}")
        return

    scope_all = True
    if args:
        if args[0] == "here":
            scope_all = False
        elif args[0] != "all":
            await update.message.reply_text(
                "Usage: /sessions | /sessions here | /sessions use <n|thread_id>"
            )
            return
        if len(args) > 1:
            await update.message.reply_text(
                "Usage: /sessions | /sessions here | /sessions use <n|thread_id>"
            )
            return

    try:
        client = await get_codex_client(state)
        if scope_all:
            listing = await client.list_threads(
                limit=SESSIONS_FETCH_LIMIT,
                cwd=None,
                sort_key="updated_at",
            )
            sessions = latest_sessions_by_workdir(
                listing.threads,
                max_sessions=SESSIONS_LIST_LIMIT,
            )
            next_cursor = listing.next_cursor
            latest_per_workdir = True
        else:
            listing = await client.list_threads(
                limit=SESSIONS_LIST_LIMIT,
                cwd=state.workdir,
                sort_key="updated_at",
            )
            sessions = list(listing.threads)
            next_cursor = listing.next_cursor
            latest_per_workdir = False
    except Exception as exc:
        LOGGER.warning("failed to list sessions: %s", exc)
        await update.message.reply_text(f"Failed to list sessions: {exc}")
        return

    state.last_session_ids = [session.thread_id for session in sessions]
    await send_message(
        context.application,
        format_sessions_text(
            sessions,
            scope_all=scope_all,
            workdir=state.workdir,
            current_thread_id=state.thread_id,
            next_cursor=next_cursor,
            latest_per_workdir=latest_per_workdir,
        ),
        chat_id=int(update.effective_chat.id),
        reply_to_message_id=update.message.message_id,
        reply_markup=_sessions_keyboard(sessions),
    )


async def proj_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return

    state: BotState = context.application.bot_data["state"]
    args = context.args

    if not args:
        if not state.project_map:
            await update.message.reply_text("projects: (empty)")
            return
        items: list[str] = []
        for name, project in sorted(state.project_map.items()):
            label = f"{name} (active)" if name == state.active_project else name
            items.append(f"{label} -> {project.path}")
        await update.message.reply_text("projects:\n" + "\n".join(items))
        return

    if args[0] == "rm":
        if len(args) < 2:
            await update.message.reply_text("Usage: /proj rm <name>")
            return
        name = args[1]
        if name in state.project_map:
            state.project_map.pop(name, None)
            removed_active = state.active_project == name
            if removed_active:
                state.active_project = None
                state.thread_id = None
                state.pin = None
            persist_state(state)
            if removed_active:
                await update.message.reply_text(
                    f"removed: {name}\nactive project cleared; thread and pin reset."
                )
                return
            await update.message.reply_text(f"removed: {name}")
            return
        await update.message.reply_text(f"no such project: {name}")
        return

    name = args[0]
    if len(args) == 1:
        if name not in state.project_map:
            await update.message.reply_text(f"no such project: {name}")
            return
        sync_active_project_state(state)
        if not restore_project_state(state, name):
            project = state.project_map[name]
            LOGGER.warning(
                "project %s has missing directory during switch: %s",
                name,
                project.path,
            )
            persist_state(state)
            await update.message.reply_text(f"missing directory: {project.path}")
            return
        persist_state(state)
        await update.message.reply_text(state.workdir)
        return

    raw_path = " ".join(args[1:]).strip()
    if not raw_path:
        await update.message.reply_text("Usage: /proj <name> <path>")
        return

    path = resolve_workdir(raw_path, state.workdir)
    if not path or not os.path.isdir(path):
        await update.message.reply_text(f"No such directory: {path or raw_path}")
        return

    sync_active_project_state(state)
    existing = state.project_map.get(name)
    project = ProjectState(
        path=path,
        thread_id=existing.thread_id if existing else None,
        pin=existing.pin if existing else None,
    )
    state.project_map[name] = project
    state.active_project = name
    state.workdir = path
    state.thread_id = project.thread_id
    state.pin = project.pin
    persist_state(state)
    await update.message.reply_text(state.workdir)


async def pin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return

    state: BotState = context.application.bot_data["state"]
    text = join_command_args(context.args)
    if not text:
        await update.message.reply_text(f"pin: {state.pin or '(empty)'}")
        return

    state.pin = shorten(text, 120)
    persist_state(state)
    await update.message.reply_text(f"pin set: {state.pin}")


async def unpin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return

    state: BotState = context.application.bot_data["state"]
    state.pin = None
    persist_state(state)
    await update.message.reply_text("pin cleared.")


async def run_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None or update.effective_chat is None:
        return

    state: BotState = context.application.bot_data["state"]
    cmd_text = join_command_args(context.args)
    if not cmd_text:
        await update.message.reply_text("Usage: /run <command>")
        return

    if state.run_lock.locked():
        await send_message(
            context.application,
            "Djinn is busy. Please wait for the current run to finish.",
            chat_id=int(update.effective_chat.id),
            reply_to_message_id=update.message.message_id,
        )
        return

    try:
        argv = shlex.split(cmd_text)
    except ValueError as exc:
        await update.message.reply_text(f"Invalid command: {exc}")
        return

    if not argv:
        await update.message.reply_text("Usage: /run <command>")
        return

    async with state.run_lock:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=state.workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            await update.message.reply_text(f"Command not found: {argv[0]}")
            await drain_queued_turns(context, state)
            return
        except Exception as exc:
            LOGGER.warning("failed to start command %s: %s", argv, exc)
            await update.message.reply_text(f"Failed to start: {exc}")
            await drain_queued_turns(context, state)
            return

        state.active_shell_proc = proc
        state.shell_cancel_requested = False
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=RUN_COMMAND_TIMEOUT_S,
            )
        except TimeoutError:
            timed_out = True
            LOGGER.warning(
                "command timed out after %.1fs: %s",
                RUN_COMMAND_TIMEOUT_S,
                argv,
            )
            await terminate_process(proc)
            stdout = b""
            stderr = b""
        finally:
            state.active_shell_proc = None

        if timed_out:
            await send_message(
                context.application,
                f"$ {' '.join(argv)}\n(command timed out after {int(RUN_COMMAND_TIMEOUT_S)}s)",
                chat_id=int(update.effective_chat.id),
                reply_to_message_id=update.message.message_id,
            )
            await drain_queued_turns(context, state)
            return

        cancelled = state.shell_cancel_requested
        state.shell_cancel_requested = False

        out = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
        err = stderr.decode("utf-8", errors="replace").strip() if stderr else ""

        parts: list[str] = [f"$ {' '.join(argv)}"]
        if out:
            parts.append(truncate_output(out, max_chars=RUN_OUTPUT_MAX_CHARS, label="stdout"))
        if err:
            parts.append(truncate_output(err, max_chars=RUN_OUTPUT_MAX_CHARS, label="stderr"))
        if cancelled:
            parts.append("(cancelled by user)")
        parts.append(f"(exit {proc.returncode})")

        await send_message(
            context.application,
            "\n".join(parts),
            chat_id=int(update.effective_chat.id),
            reply_to_message_id=update.message.message_id,
        )
        await drain_queued_turns(context, state)


async def approval_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    if not is_authorized(update):
        await query.answer("Not authorized", show_alert=True)
        return

    data = query.data or ""
    parts = data.split(":", 2)
    if len(parts) != 3 or parts[0] != "approval":
        await query.answer("Invalid approval payload", show_alert=True)
        return

    key = parts[1]
    decision_token = parts[2]
    decision = APPROVAL_ACCEPT if decision_token == "a" else APPROVAL_DECLINE

    state: BotState = context.application.bot_data["state"]
    pending = state.pending_approvals.get(key)
    if pending is None:
        await query.answer("Approval request is no longer active", show_alert=True)
        return

    if not pending.future.done():
        pending.future.set_result(decision)

    await query.answer("Approved" if decision == APPROVAL_ACCEPT else "Denied")
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except BadRequest:
        LOGGER.debug("approval message already updated")


async def sessions_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    if not is_authorized(update):
        await query.answer("Not authorized", show_alert=True)
        return

    data = query.data or ""
    parts = data.split(":", 1)
    if len(parts) != 2 or parts[0] != "sessions":
        await query.answer("Invalid session payload", show_alert=True)
        return

    thread_id = parts[1].strip()
    if not thread_id:
        await query.answer("Missing thread id", show_alert=True)
        return

    state: BotState = context.application.bot_data["state"]
    state.thread_id = thread_id
    persist_state(state)

    await query.answer("Session selected")
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except BadRequest:
        LOGGER.debug("session message already updated")

    message = query.message
    if message is not None:
        chat = message.chat
        if chat is not None:
            await send_message(
                context.application,
                f"Session set: {thread_id}",
                chat_id=int(chat.id),
                reply_to_message_id=message.message_id,
                prefer_markdown=False,
            )


async def extract_user_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str | None:
    if update.message is None:
        return None

    if update.message.text is not None and update.message.text.strip():
        return update.message.text.strip()

    if update.message.voice is not None:
        return await transcribe_voice(update, context)

    return None


async def send_turn_result(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    reply_to_message_id: int | None,
    result: CodexTurnResult,
    changed_files_summary: str | None = None,
) -> str | None:
    pieces: list[str] = []
    if result.message:
        pieces.append(result.message)
    if result.error:
        pieces.append(result.error)
    if changed_files_summary:
        pieces.append(changed_files_summary)
    if not pieces and result.status != "completed":
        pieces.append(f"Turn ended with status: {result.status}")

    if not pieces:
        return None

    rendered = "\n\n".join(pieces)
    await send_message(
        context.application,
        rendered,
        chat_id=chat_id,
        reply_to_message_id=reply_to_message_id,
    )
    return rendered


async def run_turn_for_input(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    state: BotState,
    chat_id: int,
    reply_to_message_id: int | None,
    user_text: str,
) -> None:
    prompt = build_prompt(user_text, state.pin)

    try:
        client = await get_codex_client(state)
    except Exception as exc:
        LOGGER.exception("failed to start Codex app-server client: %s", exc)
        message = f"Failed to start Codex App Server: {exc}"
        state.last_turn_result = message
        await send_message(
            context.application,
            message,
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id,
        )
        return

    progress = ProgressState(started_at=time.monotonic())
    progress_text = render_progress(progress, label="working", pin=state.pin)
    progress_message_id = None

    try:
        progress_message_id = await send_message(
            context.application,
            progress_text,
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id,
            disable_notification=True,
        )
    except Exception as exc:
        LOGGER.warning("failed to send progress message: %s", exc)
        progress_message_id = None

    progress_queue: asyncio.Queue[CodexProgressEvent] | None = None
    progress_task: asyncio.Task[None] | None = None

    if progress_message_id is not None:
        progress_queue = asyncio.Queue(maxsize=200)
        progress_task = asyncio.create_task(
            progress_loop(
                context=context,
                state=state,
                chat_id=chat_id,
                progress_message_id=progress_message_id,
                progress=progress,
                queue=progress_queue,
            )
        )

    async def on_progress(event: CodexProgressEvent) -> None:
        if progress_queue is None:
            return
        try:
            progress_queue.put_nowait(event)
        except asyncio.QueueFull:
            LOGGER.warning("progress queue full; dropping event %s", event.item_id)

    async def on_approval(request: CodexApprovalRequest) -> str:
        return await request_approval(
            context=context,
            state=state,
            request=request,
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id,
        )

    try:
        result = await client.run_turn(
            prompt=prompt,
            thread_id=state.thread_id,
            cwd=state.workdir,
            approval_policy=CODEX_APPROVAL_POLICY,
            developer_instructions=SYSTEM_HINT,
            on_progress=on_progress if progress_queue is not None else None,
            on_approval=on_approval,
        )
    except Exception as exc:
        LOGGER.exception("turn failed: %s", exc)
        message = f"Djinn failed to complete the turn: {exc}"
        state.last_turn_result = message
        await send_message(
            context.application,
            message,
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id,
        )
        return
    finally:
        if progress_task is not None:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                LOGGER.warning("progress task failed: %s", exc)
        if progress_message_id is not None:
            await delete_message(
                context.application,
                chat_id=chat_id,
                message_id=progress_message_id,
            )

    state.thread_id = result.thread_id
    persist_state(state)
    rendered = await send_turn_result(
        context=context,
        chat_id=chat_id,
        reply_to_message_id=reply_to_message_id,
        result=result,
        changed_files_summary=format_changed_files_summary(progress),
    )
    state.last_turn_result = rendered


async def drain_queued_turns(context: ContextTypes.DEFAULT_TYPE, state: BotState) -> None:
    while state.queued_turn is not None:
        queued = state.queued_turn
        state.queued_turn = None
        await run_turn_for_input(
            context=context,
            state=state,
            chat_id=queued.chat_id,
            reply_to_message_id=queued.reply_to_message_id,
            user_text=queued.user_text,
        )


async def handle_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.effective_chat is None:
        return

    state: BotState = context.application.bot_data["state"]
    user_text = await extract_user_text(update, context)
    if not user_text:
        return

    await run_turn_for_input(
        context=context,
        state=state,
        chat_id=int(update.effective_chat.id),
        reply_to_message_id=update.message.message_id,
        user_text=user_text,
    )


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None or update.effective_chat is None:
        return
    if update.message.text is None and update.message.voice is None:
        return

    state: BotState = context.application.bot_data["state"]
    user_text = await extract_user_text(update, context)
    if not user_text:
        return

    turn = QueuedTurn(
        user_text=user_text,
        chat_id=int(update.effective_chat.id),
        reply_to_message_id=update.message.message_id,
    )

    if state.run_lock.locked():
        replaced = state.queued_turn is not None
        state.queued_turn = turn
        queue_message = (
            "Djinn is busy. Replaced the queued message with your latest one."
            if replaced
            else "Djinn is busy. Queued your message and will run it next."
        )
        await send_message(
            context.application,
            queue_message,
            chat_id=int(update.effective_chat.id),
            reply_to_message_id=update.message.message_id,
        )
        return

    async with state.run_lock:
        await run_turn_for_input(
            context=context,
            state=state,
            chat_id=turn.chat_id,
            reply_to_message_id=turn.reply_to_message_id,
            user_text=turn.user_text,
        )
        await drain_queued_turns(context, state)


async def startup_notify(application: Application) -> None:
    state = application.bot_data.get("state")
    if not isinstance(state, BotState):
        return

    try:
        chat_id = int(str(TELEGRAM_CHAT_ID))
    except (TypeError, ValueError):
        LOGGER.warning("startup notification skipped due to invalid TELEGRAM_CHAT_ID")
        return

    message = "\n".join(
        [
            "Djinn online.",
            f"project: {state.active_project or 'none'}",
            f"workdir: {state.workdir}",
        ]
    )
    try:
        await send_message(
            application,
            message,
            chat_id=chat_id,
            disable_notification=True,
            prefer_markdown=False,
        )
    except Exception as exc:
        LOGGER.warning("failed to send startup notification: %s", exc)


async def shutdown(application: Application) -> None:
    state = application.bot_data.get("state")
    if isinstance(state, BotState) and state.codex is not None:
        await state.codex.close()


# Entrypoint

def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")

    state = BotState()
    state.project_map = load_projects()
    runtime_state = load_runtime_state()

    # Runtime state is the fallback when no active project can be restored.
    saved_workdir = runtime_state.get("workdir")
    if saved_workdir and os.path.isdir(saved_workdir):
        state.workdir = saved_workdir

    saved_thread = runtime_state.get("thread_id")
    if saved_thread:
        state.thread_id = saved_thread

    saved_pin = runtime_state.get("pin")
    if saved_pin:
        state.pin = saved_pin

    saved_active_project = runtime_state.get("active_project")
    if saved_active_project:
        project = state.project_map.get(saved_active_project)
        if project is None:
            LOGGER.warning(
                "saved active project %s not found in project map",
                saved_active_project,
            )
        elif not os.path.isdir(project.path):
            LOGGER.warning(
                "saved active project %s has missing directory: %s",
                saved_active_project,
                project.path,
            )
        else:
            state.active_project = saved_active_project
            state.workdir = project.path
            state.thread_id = project.thread_id
            state.pin = project.pin

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(startup_notify)
        .post_shutdown(shutdown)
        .build()
    )
    application.bot_data["state"] = state

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("cancel", cancel_cmd))
    application.add_handler(CommandHandler("last", last_cmd))
    application.add_handler(CommandHandler("approve", approve_cmd))
    application.add_handler(CommandHandler("deny", deny_cmd))
    application.add_handler(CommandHandler("reset", reset_cmd))
    application.add_handler(CommandHandler("pwd", pwd_cmd))
    application.add_handler(CommandHandler("cd", cd_cmd))
    application.add_handler(CommandHandler("thread", thread_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("sessions", sessions_cmd))
    application.add_handler(CommandHandler("proj", proj_cmd))
    application.add_handler(CommandHandler("pin", pin_cmd))
    application.add_handler(CommandHandler("unpin", unpin_cmd))
    application.add_handler(CommandHandler("run", run_cmd))
    application.add_handler(CallbackQueryHandler(approval_callback, pattern=r"^approval:"))
    application.add_handler(CallbackQueryHandler(sessions_callback, pattern=r"^sessions:"))
    application.add_handler(
        MessageHandler((filters.TEXT | filters.VOICE) & ~filters.COMMAND, on_message)
    )

    application.run_polling()


if __name__ == "__main__":
    main()
