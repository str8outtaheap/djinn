from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

from telegram.constants import ParseMode
from telegram.error import BadRequest

import bot
from codex import (
    CodexProgressEvent,
    CodexThreadListResult,
    CodexThreadSummary,
)


def _update(chat_id: int | None, user_id: int | None):
    chat = SimpleNamespace(id=chat_id) if chat_id is not None else None
    user = SimpleNamespace(id=user_id) if user_id is not None else None
    return SimpleNamespace(effective_chat=chat, effective_user=user)


def _message(
    *,
    text: str | None,
    message_id: int = 1,
):
    replies: list[str] = []

    async def reply_text(value: str) -> None:
        replies.append(value)

    message = SimpleNamespace(
        text=text,
        voice=None,
        message_id=message_id,
        reply_text=reply_text,
    )
    return message, replies


def _update_with_message(
    *,
    chat_id: int,
    user_id: int,
    message: Any,
):
    return SimpleNamespace(
        message=message,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
    )


def _context_with_state(state: bot.BotState):
    app = SimpleNamespace(bot_data={"state": state})
    return SimpleNamespace(application=app, args=[])


def test_resolve_workdir_relative(tmp_path):
    base = str(tmp_path)
    resolved = bot.resolve_workdir("sub/dir", base)
    assert resolved == str((tmp_path / "sub" / "dir").resolve())


def test_resolve_workdir_absolute(tmp_path):
    absolute = str((tmp_path / "absolute").resolve())
    resolved = bot.resolve_workdir(absolute, str(tmp_path))
    assert resolved == absolute


def test_load_projects_migrates_legacy_format(tmp_path, monkeypatch):
    projects_path = tmp_path / "projects.json"
    projects_path.write_text('{"api":"/tmp/api"}', encoding="utf-8")
    monkeypatch.setattr(bot, "PROJECTS_PATH", str(projects_path))

    projects = bot.load_projects()

    assert "api" in projects
    assert projects["api"].path == "/tmp/api"
    assert projects["api"].thread_id is None
    assert projects["api"].pin is None

    payload = json.loads(projects_path.read_text(encoding="utf-8"))
    assert payload["api"] == {"path": "/tmp/api", "thread_id": None, "pin": None}


def test_restore_project_state_restores_workdir_thread_and_pin(tmp_path):
    project_dir = tmp_path / "api"
    project_dir.mkdir()
    state = bot.BotState(
        workdir=str(tmp_path),
        project_map={
            "api": bot.ProjectState(
                path=str(project_dir),
                thread_id="thread-123",
                pin="keep tests green",
            )
        },
    )

    ok = bot.restore_project_state(state, "api")

    assert ok is True
    assert state.active_project == "api"
    assert state.workdir == str(project_dir)
    assert state.thread_id == "thread-123"
    assert state.pin == "keep tests green"


def test_sync_active_project_state_updates_project_entry():
    state = bot.BotState(
        workdir="/tmp/new-path",
        thread_id="thread-999",
        pin="ctx",
        active_project="api",
        project_map={"api": bot.ProjectState(path="/tmp/old-path")},
    )

    bot.sync_active_project_state(state)

    project = state.project_map["api"]
    assert project.path == "/tmp/new-path"
    assert project.thread_id == "thread-999"
    assert project.pin == "ctx"


def test_project_switch_round_trip_preserves_each_context(tmp_path):
    project_a = tmp_path / "api"
    project_b = tmp_path / "web"
    project_a.mkdir()
    project_b.mkdir()
    project_a_subdir = project_a / "services"
    project_a_subdir.mkdir()

    state = bot.BotState(
        workdir=str(project_a),
        thread_id="thread-a0",
        pin="pin-a0",
        active_project="api",
        project_map={
            "api": bot.ProjectState(
                path=str(project_a),
                thread_id="thread-a0",
                pin="pin-a0",
            ),
            "web": bot.ProjectState(
                path=str(project_b),
                thread_id="thread-b0",
                pin="pin-b0",
            ),
        },
    )

    state.workdir = str(project_a_subdir)
    state.thread_id = "thread-a1"
    state.pin = "pin-a1"
    bot.sync_active_project_state(state)

    assert bot.restore_project_state(state, "web")
    state.thread_id = "thread-b1"
    state.pin = "pin-b1"
    bot.sync_active_project_state(state)

    assert bot.restore_project_state(state, "api")
    assert state.active_project == "api"
    assert state.workdir == str(project_a_subdir)
    assert state.thread_id == "thread-a1"
    assert state.pin == "pin-a1"

    assert state.project_map["web"].thread_id == "thread-b1"
    assert state.project_map["web"].pin == "pin-b1"


def test_persist_state_saves_active_project_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(bot, "PROJECTS_PATH", str(tmp_path / "projects.json"))
    project_dir = tmp_path / "api"
    project_dir.mkdir()

    state = bot.BotState(
        workdir=str(project_dir),
        thread_id="thread-abc",
        pin="pin",
        active_project="api",
        project_map={
            "api": bot.ProjectState(
                path=str(project_dir),
                thread_id="thread-abc",
                pin="pin",
            )
        },
    )
    bot.persist_state(state)

    loaded = bot.load_runtime_state()
    assert loaded["workdir"] == str(project_dir)
    assert loaded["thread_id"] == "thread-abc"
    assert loaded["pin"] == "pin"
    assert loaded["active_project"] == "api"


def test_build_prompt_without_pin_returns_user_text():
    text = bot.build_prompt("fix the test", pin=None)
    assert text == "fix the test"


def test_build_prompt_with_pin_includes_context():
    text = bot.build_prompt("fix the test", pin="project: api gateway")
    assert "Pinned context: project: api gateway" in text
    assert text.endswith("User: fix the test")


def test_build_help_text_mentions_cancel_and_proj():
    text = bot.build_help_text()
    assert "/cancel" in text
    assert "/proj <name>" in text
    assert "/sessions use <n|thread_id>" in text
    assert "/approve" not in text
    assert "/deny" not in text


def test_truncate_output_adds_notice_when_clipped():
    text = bot.truncate_output("abcdefghij", max_chars=5, label="stdout")
    assert text.startswith("abcde")
    assert "truncated to 5 chars" in text


def test_note_progress_event_tracks_changed_files():
    progress = bot.ProgressState(started_at=0.0)
    event = CodexProgressEvent(
        item_id="f1",
        kind="file_change",
        phase="completed",
        title="files",
        detail={"changes": [{"path": "src/app.py", "kind": "update"}]},
    )
    changed = bot.note_progress_event(progress, event)
    assert changed is True
    assert progress.changed_files == {"src/app.py": "update"}


def test_format_changed_files_summary_limits_output():
    progress = bot.ProgressState(
        started_at=0.0,
        changed_files={
            "a.py": "update",
            "b.py": "add",
            "c.py": "delete",
        },
    )
    summary = bot.format_changed_files_summary(progress, max_files=2)
    assert summary is not None
    assert "`a.py` (update)" in summary
    assert "`b.py` (add)" in summary
    assert "... and 1 more" in summary


def test_is_authorized_chat_only(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(bot, "TELEGRAM_USER_ID", None)

    assert bot.is_authorized(_update(chat_id=123, user_id=1))
    assert not bot.is_authorized(_update(chat_id=999, user_id=1))


def test_is_authorized_chat_and_user(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(bot, "TELEGRAM_USER_ID", "77")

    assert bot.is_authorized(_update(chat_id=123, user_id=77))
    assert not bot.is_authorized(_update(chat_id=123, user_id=88))


def test_format_progress_line_command_completed_ok():
    event = CodexProgressEvent(
        item_id="i1",
        kind="command",
        phase="completed",
        title="echo hello",
        ok=True,
    )
    line = bot.format_progress_line(event)
    assert line.startswith("OK cmd:")
    assert "echo hello" in line


def test_note_progress_event_ignores_agent_messages():
    progress = bot.ProgressState(started_at=0.0)
    event = CodexProgressEvent(
        item_id="m1",
        kind="agent_message",
        phase="delta",
        title="hello",
    )
    changed = bot.note_progress_event(progress, event)
    assert changed is False
    assert progress.actions == {}


def test_render_progress_limits_actions(monkeypatch):
    monkeypatch.setattr(bot, "PROGRESS_MAX_ACTIONS", 2)
    progress = bot.ProgressState(started_at=0.0)

    for i in range(4):
        event = CodexProgressEvent(
            item_id=f"{i}",
            kind="command",
            phase="started",
            title=f"cmd-{i}",
        )
        bot.note_progress_event(progress, event)

    text = bot.render_progress(progress, label="working", pin=None)
    assert "cmd-0" not in text
    assert "cmd-1" not in text
    assert "cmd-2" in text
    assert "cmd-3" in text


def test_send_message_markdown_fallback():
    class FakeBot:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def send_message(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("parse_mode") == ParseMode.MARKDOWN:
                raise BadRequest("bad markdown")
            return SimpleNamespace(message_id=42)

    class FakeApp:
        def __init__(self) -> None:
            self.bot = FakeBot()

    app = FakeApp()

    async def run_case() -> int | None:
        return await bot.send_message(
            cast(Any, app),
            "hello *world*",
            chat_id=1,
            prefer_markdown=True,
        )

    message_id = asyncio.run(run_case())
    assert message_id == 42
    assert len(app.bot.calls) == 2
    assert app.bot.calls[0]["parse_mode"] == ParseMode.MARKDOWN
    assert "parse_mode" not in app.bot.calls[1]


def test_on_message_busy_replaces_one_deep_queue(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(bot, "TELEGRAM_USER_ID", None)
    state = bot.BotState()
    context = _context_with_state(state)
    sent: list[str] = []

    async def fake_send_message(_application, text: str, **_kwargs):
        sent.append(text)
        return None

    monkeypatch.setattr(bot, "send_message", fake_send_message)

    async def run_case() -> None:
        await state.run_lock.acquire()
        try:
            first_message, _ = _message(text="first", message_id=11)
            second_message, _ = _message(text="second", message_id=12)
            first = _update_with_message(chat_id=123, user_id=7, message=first_message)
            second = _update_with_message(chat_id=123, user_id=7, message=second_message)
            await bot.on_message(first, context)
            await bot.on_message(second, context)
        finally:
            state.run_lock.release()

    asyncio.run(run_case())

    assert state.queued_turn is not None
    assert state.queued_turn.user_text == "second"
    assert sent == [
        "Djinn is busy. Queued your message and will run it next.",
        "Djinn is busy. Replaced the queued message with your latest one.",
    ]


def test_last_cmd_resends_saved_result(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(bot, "TELEGRAM_USER_ID", None)
    state = bot.BotState(last_turn_result="done text")
    context = _context_with_state(state)
    message, _ = _message(text="/last", message_id=55)
    update = _update_with_message(chat_id=123, user_id=7, message=message)
    sent: dict[str, Any] = {}

    async def fake_send_message(_application, text: str, **kwargs):
        sent["text"] = text
        sent["chat_id"] = kwargs.get("chat_id")
        sent["reply_to_message_id"] = kwargs.get("reply_to_message_id")
        return 77

    monkeypatch.setattr(bot, "send_message", fake_send_message)
    asyncio.run(bot.last_cmd(update, context))

    assert sent == {
        "text": "done text",
        "chat_id": 123,
        "reply_to_message_id": 55,
    }


def test_last_cmd_reports_missing_result(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(bot, "TELEGRAM_USER_ID", None)
    state = bot.BotState(last_turn_result=None)
    context = _context_with_state(state)
    message, replies = _message(text="/last", message_id=56)
    update = _update_with_message(chat_id=123, user_id=7, message=message)

    asyncio.run(bot.last_cmd(update, context))

    assert replies == ["No turn result yet."]


def test_drain_queued_turns_processes_new_items_until_empty(monkeypatch):
    state = bot.BotState()
    context = _context_with_state(state)
    calls: list[str] = []

    async def fake_run_turn_for_input(
        *,
        context: Any,
        state: bot.BotState,
        chat_id: int,
        reply_to_message_id: int | None,
        user_text: str,
    ) -> None:
        calls.append(user_text)
        if user_text == "first":
            state.queued_turn = bot.QueuedTurn(
                user_text="second",
                chat_id=chat_id,
                reply_to_message_id=reply_to_message_id,
            )

    monkeypatch.setattr(bot, "run_turn_for_input", fake_run_turn_for_input)
    state.queued_turn = bot.QueuedTurn(user_text="first", chat_id=1, reply_to_message_id=2)

    asyncio.run(bot.drain_queued_turns(context, state))

    assert calls == ["first", "second"]
    assert state.queued_turn is None


def test_startup_notify_sends_online_message(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_CHAT_ID", "123")
    state = bot.BotState(workdir="/tmp/project", active_project="api")
    app = SimpleNamespace(bot_data={"state": state})
    sent: dict[str, Any] = {}

    async def fake_send_message(_application, text: str, **kwargs):
        sent["text"] = text
        sent["chat_id"] = kwargs.get("chat_id")
        return None

    monkeypatch.setattr(bot, "send_message", fake_send_message)
    asyncio.run(bot.startup_notify(cast(Any, app)))

    assert sent["chat_id"] == 123
    assert "Djinn online." in sent["text"]
    assert "project: api" in sent["text"]


def test_resolve_session_selection_by_index():
    state = bot.BotState(last_session_ids=["thread-a", "thread-b"])
    thread_id, error_text = bot.resolve_session_selection(state, "2")
    assert error_text is None
    assert thread_id == "thread-b"


def test_resolve_session_selection_rejects_missing_index():
    state = bot.BotState(last_session_ids=["thread-a"])
    thread_id, error_text = bot.resolve_session_selection(state, "3")
    assert thread_id is None
    assert error_text == "No session #3. Run /sessions first."


def test_format_sessions_text_marks_current_thread():
    sessions = [
        CodexThreadSummary(
            thread_id="01234567-0123-0123-0123-0123456789ab",
            cwd="/tmp/work",
            source="cli",
            updated_at=100,
        ),
        CodexThreadSummary(
            thread_id="thread-two",
            cwd="/tmp/other",
            source="vscode",
            updated_at=200,
        ),
    ]
    text = bot.format_sessions_text(
        sessions,
        scope_all=False,
        workdir="/tmp/work",
        current_thread_id="thread-two",
    )
    assert "Recent sessions (/tmp/work):" in text
    assert "(current)" in text
    assert "/sessions use <n>" in text


def test_latest_sessions_by_workdir_keeps_first_entry_for_each_cwd():
    sessions = [
        CodexThreadSummary(thread_id="thread-1", cwd="/tmp/work"),
        CodexThreadSummary(thread_id="thread-2", cwd="/tmp/work"),
        CodexThreadSummary(thread_id="thread-3", cwd="/tmp/other"),
    ]
    deduped = bot.latest_sessions_by_workdir(sessions, max_sessions=10)
    assert [session.thread_id for session in deduped] == ["thread-1", "thread-3"]


def test_sessions_cmd_lists_newest_session_per_workdir_by_default(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(bot, "TELEGRAM_USER_ID", None)

    state = bot.BotState(workdir="/tmp/work", thread_id="thread-2")
    context = _context_with_state(state)
    context.args = []
    message, _ = _message(text="/sessions", message_id=90)
    update = _update_with_message(chat_id=123, user_id=7, message=message)

    called: dict[str, Any] = {}
    sent: dict[str, Any] = {}

    class FakeClient:
        async def list_threads(self, **kwargs):
            called.update(kwargs)
            return CodexThreadListResult(
                threads=[
                    CodexThreadSummary(thread_id="thread-1", cwd="/tmp/work", source="cli"),
                    CodexThreadSummary(thread_id="thread-2", cwd="/tmp/work", source="vscode"),
                    CodexThreadSummary(thread_id="thread-3", cwd="/tmp/other", source="cli"),
                ],
                next_cursor=None,
            )

    async def fake_get_codex_client(_state: bot.BotState):
        return FakeClient()

    async def fake_send_message(_application, text: str, **kwargs):
        sent["text"] = text
        sent["reply_markup"] = kwargs.get("reply_markup")
        return 1

    monkeypatch.setattr(bot, "get_codex_client", fake_get_codex_client)
    monkeypatch.setattr(bot, "send_message", fake_send_message)
    asyncio.run(bot.sessions_cmd(update, context))

    assert called == {"limit": bot.SESSIONS_FETCH_LIMIT, "cwd": None, "sort_key": "updated_at"}
    assert state.last_session_ids == ["thread-1", "thread-3"]
    assert "Recent sessions (all workdirs):" in sent["text"]
    assert "Showing newest session per workdir." in sent["text"]
    assert sent["reply_markup"] is not None


def test_sessions_cmd_here_lists_current_workdir_history(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(bot, "TELEGRAM_USER_ID", None)

    state = bot.BotState(workdir="/tmp/work")
    context = _context_with_state(state)
    context.args = ["here"]
    message, _ = _message(text="/sessions here", message_id=93)
    update = _update_with_message(chat_id=123, user_id=7, message=message)

    called: dict[str, Any] = {}
    sent: dict[str, Any] = {}

    class FakeClient:
        async def list_threads(self, **kwargs):
            called.update(kwargs)
            return CodexThreadListResult(
                threads=[
                    CodexThreadSummary(thread_id="thread-1", cwd="/tmp/work", source="cli"),
                    CodexThreadSummary(thread_id="thread-2", cwd="/tmp/work", source="vscode"),
                ],
                next_cursor="cursor-1",
            )

    async def fake_get_codex_client(_state: bot.BotState):
        return FakeClient()

    async def fake_send_message(_application, text: str, **kwargs):
        sent["text"] = text
        sent["reply_markup"] = kwargs.get("reply_markup")
        return 1

    monkeypatch.setattr(bot, "get_codex_client", fake_get_codex_client)
    monkeypatch.setattr(bot, "send_message", fake_send_message)
    asyncio.run(bot.sessions_cmd(update, context))

    assert called == {
        "limit": bot.SESSIONS_LIST_LIMIT,
        "cwd": "/tmp/work",
        "sort_key": "updated_at",
    }
    assert state.last_session_ids == ["thread-1", "thread-2"]
    assert "Recent sessions (/tmp/work):" in sent["text"]
    assert "Use `/sessions` to browse newest sessions across workdirs." in sent["text"]
    assert sent["reply_markup"] is not None


def test_sessions_cmd_use_index_updates_thread(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(bot, "TELEGRAM_USER_ID", None)

    state = bot.BotState(thread_id="old", last_session_ids=["thread-a", "thread-b"])
    context = _context_with_state(state)
    context.args = ["use", "2"]
    message, replies = _message(text="/sessions use 2", message_id=91)
    update = _update_with_message(chat_id=123, user_id=7, message=message)

    saved: list[str] = []

    def fake_persist_state(s: bot.BotState):
        saved.append(s.thread_id or "")

    monkeypatch.setattr(bot, "persist_state", fake_persist_state)
    asyncio.run(bot.sessions_cmd(update, context))

    assert state.thread_id == "thread-b"
    assert saved == ["thread-b"]
    assert replies == ["Session set: thread-b"]


def test_sessions_callback_sets_thread(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(bot, "TELEGRAM_USER_ID", None)
    state = bot.BotState(thread_id="old")
    context = _context_with_state(state)

    answers: list[str] = []

    async def answer(text: str, show_alert: bool = False):
        del show_alert
        answers.append(text)

    async def edit_message_reply_markup(reply_markup=None):
        del reply_markup
        return None

    callback_message = SimpleNamespace(
        chat=SimpleNamespace(id=123),
        message_id=92,
    )
    query = SimpleNamespace(
        data="sessions:thread-c",
        answer=answer,
        edit_message_reply_markup=edit_message_reply_markup,
        message=callback_message,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=123),
        effective_user=SimpleNamespace(id=7),
    )

    sent: dict[str, Any] = {}

    async def fake_send_message(_application, text: str, **kwargs):
        sent["text"] = text
        sent["chat_id"] = kwargs.get("chat_id")
        return 1

    saved: list[str] = []

    def fake_persist_state(s: bot.BotState):
        saved.append(s.thread_id or "")

    monkeypatch.setattr(bot, "persist_state", fake_persist_state)
    monkeypatch.setattr(bot, "send_message", fake_send_message)
    asyncio.run(bot.sessions_callback(cast(Any, update), context))

    assert answers == ["Session selected"]
    assert state.thread_id == "thread-c"
    assert saved == ["thread-c"]
    assert sent["text"] == "Session set: thread-c"
    assert sent["chat_id"] == 123
