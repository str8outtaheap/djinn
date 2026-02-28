from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from codex import CodexAppServerClient
from config import DEFAULT_WORKDIR, PROJECTS_PATH, STATE_PATH

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
        path = raw
        thread_id = None
        pin = None
    elif isinstance(raw, dict):
        path = raw.get("path")
        thread_id = raw.get("thread_id")
        pin = raw.get("pin")
    else:
        return None

    if not isinstance(path, str) or not path:
        return None

    if not isinstance(thread_id, str) or not thread_id:
        thread_id = None

    if not isinstance(pin, str) or not pin:
        pin = None

    return ProjectState(path=path, thread_id=thread_id, pin=pin)


def load_projects() -> dict[str, ProjectState]:
    raw = _load_json_dict(PROJECTS_PATH)
    projects: dict[str, ProjectState] = {}
    migrated = False

    for name, payload in raw.items():
        if not isinstance(name, str) or not name:
            continue
        project = _coerce_project_state(payload)
        if project is None:
            continue
        if not os.path.isdir(project.path):
            LOGGER.warning("ignoring missing project path for %s: %s", name, project.path)
            continue
        projects[name] = project
        if not isinstance(payload, dict):
            migrated = True

    if migrated:
        save_projects(projects)

    return projects


def save_projects(projects: dict[str, ProjectState]) -> None:
    payload: dict[str, dict[str, Any]] = {}
    for name, project in projects.items():
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
