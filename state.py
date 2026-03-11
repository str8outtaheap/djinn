from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from codex import CodexAppServerClient
from config import DEFAULT_WORKDIR, STATE_PATH

LOGGER = logging.getLogger(__name__)


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
class BotState:
    workdir: str = DEFAULT_WORKDIR
    thread_id: str | None = None
    run_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_command_exec_id: str | None = None
    command_cancel_requested: bool = False
    queued_turn: QueuedTurn | None = None
    last_session_ids: list[str] = field(default_factory=list)
    codex: CodexAppServerClient | None = None


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

    return state


def save_runtime_state(state: BotState) -> None:
    payload: dict[str, Any] = {
        "workdir": state.workdir,
    }
    if state.thread_id:
        payload["thread_id"] = state.thread_id
    _save_json_dict(STATE_PATH, payload)


def persist_state(state: BotState) -> None:
    save_runtime_state(state)
