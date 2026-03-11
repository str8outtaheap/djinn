from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from telegram.constants import ParseMode
from telegram.error import BadRequest

import bot
import commands
import state as bot_state
from codex import (
    CodexCommandExecOutputDelta,
    CodexCommandExecResult,
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


def test_persist_state_saves_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_state, "STATE_PATH", str(tmp_path / "state.json"))
    workdir = tmp_path / "workspace"
    workdir.mkdir()

    state = bot.BotState(
        workdir=str(workdir),
        thread_id="thread-abc",
    )
    bot.persist_state(state)

    loaded = bot.load_runtime_state()
    assert loaded["workdir"] == str(workdir)
    assert loaded["thread_id"] == "thread-abc"


def test_build_prompt_returns_user_text():
    text = bot.build_prompt("fix the test")
    assert text == "fix the test"


def test_build_help_text_locks_lean_command_surface():
    text = bot.build_help_text()

    kept_commands = (
        "/start",
        "/help",
        "/cancel",
        "/status",
        "/sessions",
        "/sessions use <n|thread_id>",
        "/run <command>",
        "/cd <path>",
        "/reset",
    )
    removed_commands = (
        "/last",
        "/pwd",
        "/thread",
        "/proj",
        "/pin",
        "/unpin",
        "/approve",
        "/deny",
    )

    for command in kept_commands:
        assert command in text

    for command in removed_commands:
        assert command not in text


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
    monkeypatch.setattr(commands, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(commands, "TELEGRAM_USER_ID", None)

    assert bot.is_authorized(_update(chat_id=123, user_id=1))
    assert not bot.is_authorized(_update(chat_id=999, user_id=1))


def test_is_authorized_chat_and_user(monkeypatch):
    monkeypatch.setattr(commands, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(commands, "TELEGRAM_USER_ID", "77")

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
    monkeypatch.setattr(commands, "PROGRESS_MAX_ACTIONS", 2)
    progress = bot.ProgressState(started_at=0.0)

    for i in range(4):
        event = CodexProgressEvent(
            item_id=f"{i}",
            kind="command",
            phase="started",
            title=f"cmd-{i}",
        )
        bot.note_progress_event(progress, event)

    text = bot.render_progress(progress, label="working")
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
    monkeypatch.setattr(commands, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(commands, "TELEGRAM_USER_ID", None)
    state = bot.BotState()
    context = _context_with_state(state)
    sent: list[str] = []

    async def fake_send_message(_application, text: str, **_kwargs):
        sent.append(text)
        return None

    monkeypatch.setattr(commands, "send_message", fake_send_message)

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


def test_run_cmd_uses_command_exec_and_formats_result(monkeypatch):
    monkeypatch.setattr(commands, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(commands, "TELEGRAM_USER_ID", None)
    monkeypatch.setattr(commands, "format_elapsed", lambda _elapsed: "0s")
    state = bot.BotState(workdir="/tmp/project")
    context = _context_with_state(state)
    context.args = ["echo", "hello"]
    message, replies = _message(text="/run echo hello", message_id=90)
    update = _update_with_message(chat_id=123, user_id=7, message=message)
    sent: list[str] = []
    edited: list[str] = []
    deleted: list[tuple[int, int]] = []

    class FakeClient:
        def new_command_exec_process_id(self) -> str:
            return "proc-1"

        async def run_command_exec(self, *, spec, on_output=None):
            assert spec.to_rpc_params() == {
                "command": ["echo", "hello"],
                "cwd": "/tmp/project",
                "processId": "proc-1",
                "outputBytesCap": 8000,
                "streamStdoutStderr": True,
            }
            assert on_output is not None
            await on_output(
                CodexCommandExecOutputDelta(
                    process_id="proc-1",
                    stream="stdout",
                    data=b"hello\n",
                    cap_reached=False,
                )
            )
            return CodexCommandExecResult(exit_code=0, stdout="hello\n", stderr="")

    async def fake_get_codex_client(_state):
        return FakeClient()

    async def fake_send_message(_application, text: str, **_kwargs):
        sent.append(text)
        return 501 if len(sent) == 1 else 777

    async def fake_edit_message(_application, *, text: str, **_kwargs):
        edited.append(text)

    async def fake_delete_message(_application, *, chat_id: int, message_id: int):
        deleted.append((chat_id, message_id))

    monkeypatch.setattr(commands, "get_codex_client", fake_get_codex_client)
    monkeypatch.setattr(commands, "send_message", fake_send_message)
    monkeypatch.setattr(commands, "edit_message", fake_edit_message)
    monkeypatch.setattr(commands, "delete_message", fake_delete_message)

    asyncio.run(commands.run_cmd(update, context))

    assert replies == []
    assert sent == [
        "$ echo hello\n(running 0s)\n\n(waiting for output)",
        "$ echo hello\nhello\n(exit 0)",
    ]
    assert edited == ["$ echo hello\n(running 0s)\n\nstdout:\nhello"]
    assert deleted == [(123, 501)]
    assert state.active_command_exec_id is None


def test_run_cmd_reports_command_not_found_from_exec_result(monkeypatch):
    monkeypatch.setattr(commands, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(commands, "TELEGRAM_USER_ID", None)
    state = bot.BotState(workdir="/tmp/project")
    context = _context_with_state(state)
    context.args = ["missing-cmd"]
    message, replies = _message(text="/run missing-cmd", message_id=91)
    update = _update_with_message(chat_id=123, user_id=7, message=message)

    class FakeClient:
        def new_command_exec_process_id(self) -> str:
            return "proc-404"

        async def run_command_exec(self, *, spec, on_output=None):
            del spec, on_output
            return CodexCommandExecResult(
                exit_code=127,
                stdout="",
                stderr="No such file or directory",
            )

    async def fake_get_codex_client(_state):
        return FakeClient()

    async def fake_send_message(_application, text: str, **_kwargs):
        del text
        return None

    monkeypatch.setattr(commands, "get_codex_client", fake_get_codex_client)
    monkeypatch.setattr(commands, "send_message", fake_send_message)

    asyncio.run(commands.run_cmd(update, context))

    assert replies == ["Command not found: missing-cmd"]
    assert state.active_command_exec_id is None


def test_cancel_cmd_terminates_active_command_exec(monkeypatch):
    monkeypatch.setattr(commands, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(commands, "TELEGRAM_USER_ID", None)
    terminated: list[str] = []

    class FakeClient:
        async def terminate_command_exec(self, process_id: str) -> None:
            terminated.append(process_id)

        async def interrupt_active_turn(self) -> bool:
            return False

    state = bot.BotState(active_command_exec_id="proc-1", codex=cast(Any, FakeClient()))
    context = _context_with_state(state)
    message, replies = _message(text="/cancel", message_id=92)
    update = _update_with_message(chat_id=123, user_id=7, message=message)

    asyncio.run(commands.cancel_cmd(update, context))

    assert terminated == ["proc-1"]
    assert state.command_cancel_requested is True
    assert replies == ["Cancellation requested for run command."]


def test_run_cmd_reports_cancelled_output_after_streaming(monkeypatch):
    monkeypatch.setattr(commands, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(commands, "TELEGRAM_USER_ID", None)
    monkeypatch.setattr(commands, "format_elapsed", lambda _elapsed: "0s")
    state = bot.BotState(workdir="/tmp/project")
    context = _context_with_state(state)
    context.args = ["sleep", "10"]
    message, replies = _message(text="/run sleep 10", message_id=93)
    update = _update_with_message(chat_id=123, user_id=7, message=message)
    sent: list[str] = []
    edited: list[str] = []
    deleted: list[tuple[int, int]] = []

    class FakeClient:
        def new_command_exec_process_id(self) -> str:
            return "proc-cancel"

        async def run_command_exec(self, *, spec, on_output=None):
            assert spec.to_rpc_params() == {
                "command": ["sleep", "10"],
                "cwd": "/tmp/project",
                "processId": "proc-cancel",
                "outputBytesCap": 8000,
                "streamStdoutStderr": True,
            }
            assert on_output is not None
            await on_output(
                CodexCommandExecOutputDelta(
                    process_id="proc-cancel",
                    stream="stdout",
                    data=b"partial output\n",
                    cap_reached=False,
                )
            )
            state.command_cancel_requested = True
            raise RuntimeError("terminated")

    async def fake_get_codex_client(_state):
        return FakeClient()

    async def fake_send_message(_application, text: str, **_kwargs):
        sent.append(text)
        return 601 if len(sent) == 1 else 778

    async def fake_edit_message(_application, *, text: str, **_kwargs):
        edited.append(text)

    async def fake_delete_message(_application, *, chat_id: int, message_id: int):
        deleted.append((chat_id, message_id))

    monkeypatch.setattr(commands, "get_codex_client", fake_get_codex_client)
    monkeypatch.setattr(commands, "send_message", fake_send_message)
    monkeypatch.setattr(commands, "edit_message", fake_edit_message)
    monkeypatch.setattr(commands, "delete_message", fake_delete_message)

    asyncio.run(commands.run_cmd(update, context))

    assert replies == []
    assert sent == [
        "$ sleep 10\n(running 0s)\n\n(waiting for output)",
        "$ sleep 10\npartial output\n(cancelled by user)",
    ]
    assert edited == ["$ sleep 10\n(running 0s)\n\nstdout:\npartial output"]
    assert deleted == [(123, 601)]
    assert state.active_command_exec_id is None
    assert state.command_cancel_requested is False


def test_render_run_command_result_marks_output_cap():
    rendered = commands.render_run_command_result(
        "echo hello",
        stdout="hello\n",
        output_capped=True,
    )

    assert rendered == "$ echo hello\nhello\n(app-server output capped)"


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

    monkeypatch.setattr(commands, "run_turn_for_input", fake_run_turn_for_input)
    state.queued_turn = bot.QueuedTurn(user_text="first", chat_id=1, reply_to_message_id=2)

    asyncio.run(bot.drain_queued_turns(context, state))

    assert calls == ["first", "second"]
    assert state.queued_turn is None


def test_startup_notify_sends_online_message(monkeypatch):
    monkeypatch.setattr(commands, "TELEGRAM_CHAT_ID", "123")
    state = bot.BotState(workdir="/tmp/project")
    app = SimpleNamespace(bot_data={"state": state})
    sent: dict[str, Any] = {}

    async def fake_send_message(_application, text: str, **kwargs):
        sent["text"] = text
        sent["chat_id"] = kwargs.get("chat_id")
        return None

    monkeypatch.setattr(commands, "send_message", fake_send_message)
    asyncio.run(bot.startup_notify(cast(Any, app)))

    assert sent["chat_id"] == 123
    assert "Djinn online." in sent["text"]
    assert "workdir: /tmp/project" in sent["text"]


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
        current_thread_id="thread-two",
    )
    assert "Recent sessions (all workdirs):" in text
    assert "Showing newest session per workdir." in text
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
    monkeypatch.setattr(commands, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(commands, "TELEGRAM_USER_ID", None)

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

    monkeypatch.setattr(commands, "get_codex_client", fake_get_codex_client)
    monkeypatch.setattr(commands, "send_message", fake_send_message)
    asyncio.run(bot.sessions_cmd(update, context))

    assert called == {"limit": bot.SESSIONS_FETCH_LIMIT, "cwd": None, "sort_key": "updated_at"}
    assert state.last_session_ids == ["thread-1", "thread-3"]
    assert "Recent sessions (all workdirs):" in sent["text"]
    assert "Showing newest session per workdir." in sent["text"]
    assert sent["reply_markup"] is not None


def test_sessions_cmd_rejects_removed_here_mode(monkeypatch):
    monkeypatch.setattr(commands, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(commands, "TELEGRAM_USER_ID", None)

    state = bot.BotState(workdir="/tmp/work")
    context = _context_with_state(state)
    context.args = ["here"]
    message, replies = _message(text="/sessions here", message_id=93)
    update = _update_with_message(chat_id=123, user_id=7, message=message)
    asyncio.run(bot.sessions_cmd(update, context))

    assert replies == ["Usage: /sessions | /sessions use <n|thread_id>"]


def test_sessions_cmd_use_index_updates_thread(monkeypatch):
    monkeypatch.setattr(commands, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(commands, "TELEGRAM_USER_ID", None)

    state = bot.BotState(thread_id="old", last_session_ids=["thread-a", "thread-b"])
    context = _context_with_state(state)
    context.args = ["use", "2"]
    message, replies = _message(text="/sessions use 2", message_id=91)
    update = _update_with_message(chat_id=123, user_id=7, message=message)

    saved: list[str] = []

    def fake_persist_state(s: bot.BotState):
        saved.append(s.thread_id or "")

    monkeypatch.setattr(commands, "persist_state", fake_persist_state)
    asyncio.run(bot.sessions_cmd(update, context))

    assert state.thread_id == "thread-b"
    assert saved == ["thread-b"]
    assert replies == ["Session set: thread-b"]


def test_sessions_callback_sets_thread(monkeypatch):
    monkeypatch.setattr(commands, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(commands, "TELEGRAM_USER_ID", None)
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

    monkeypatch.setattr(commands, "persist_state", fake_persist_state)
    monkeypatch.setattr(commands, "send_message", fake_send_message)
    asyncio.run(bot.sessions_callback(cast(Any, update), context))

    assert answers == ["Session selected"]
    assert state.thread_id == "thread-c"
    assert saved == ["thread-c"]
    assert sent["text"] == "Session set: thread-c"
    assert sent["chat_id"] == 123
