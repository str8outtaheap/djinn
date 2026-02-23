from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from codex import CodexAppServerClient, CodexError


def test_event_from_command_item_completed_success():
    item = {
        "type": "commandExecution",
        "id": "cmd-1",
        "command": "pwd",
        "status": "completed",
        "exitCode": 0,
    }
    event = CodexAppServerClient._event_from_item(item, phase="completed")
    assert event is not None
    assert event.kind == "command"
    assert event.ok is True


def test_event_from_file_change_item():
    item = {
        "type": "fileChange",
        "id": "f1",
        "status": "completed",
        "changes": [
            {"path": "a.txt", "kind": {"type": "update"}},
            {"path": "b.txt", "kind": {"type": "add"}},
        ],
    }
    event = CodexAppServerClient._event_from_item(item, phase="completed")
    assert event is not None
    assert event.kind == "file_change"
    assert event.ok is True
    assert event.detail["changes"][0]["path"] == "a.txt"


def test_extract_thread_info_valid():
    response = {"thread": {"id": "thread-123"}, "cwd": "/tmp"}
    thread_id, cwd = CodexAppServerClient._extract_thread_info(response, "/fallback")
    assert thread_id == "thread-123"
    assert cwd == "/tmp"


def test_extract_thread_info_invalid():
    with pytest.raises(CodexError):
        CodexAppServerClient._extract_thread_info({}, "/fallback")


def test_handle_message_resolves_pending_future():
    client = CodexAppServerClient(codex_cmd="codex")

    async def run_case():
        loop = asyncio.get_running_loop()
        pending = loop.create_future()
        client._pending[1] = pending
        await client._handle_message({"id": 1, "result": {"ok": True}})
        return pending.result()

    result = asyncio.run(run_case())
    assert result == {"ok": True}


def test_ensure_thread_start_passes_developer_instructions():
    client = CodexAppServerClient(codex_cmd="codex")
    captured: list[tuple[str, dict[str, object]]] = []

    async def fake_start():
        return None

    async def fake_request(method, params, *, timeout_s=None):
        del timeout_s
        captured.append((method, params))
        return {"thread": {"id": "thread-1"}, "cwd": "/tmp"}

    client.start = fake_start  # type: ignore[method-assign]
    client._request = fake_request  # type: ignore[method-assign]

    thread_id, cwd = asyncio.run(
        client.ensure_thread(
            thread_id=None,
            cwd="/tmp",
            approval_policy="on-request",
            developer_instructions="be concise",
        )
    )

    assert thread_id == "thread-1"
    assert cwd == "/tmp"
    assert len(captured) == 1
    assert captured[0][0] == "thread/start"
    assert captured[0][1]["developerInstructions"] == "be concise"


def test_ensure_thread_resume_passes_developer_instructions():
    client = CodexAppServerClient(codex_cmd="codex")
    captured: list[tuple[str, dict[str, object]]] = []

    async def fake_start():
        return None

    async def fake_request(method, params, *, timeout_s=None):
        del timeout_s
        captured.append((method, params))
        return {"thread": {"id": "thread-2"}, "cwd": "/workspace"}

    client.start = fake_start  # type: ignore[method-assign]
    client._request = fake_request  # type: ignore[method-assign]

    thread_id, cwd = asyncio.run(
        client.ensure_thread(
            thread_id="thread-2",
            cwd="/workspace",
            approval_policy="on-request",
            developer_instructions="use tools",
        )
    )

    assert thread_id == "thread-2"
    assert cwd == "/workspace"
    assert len(captured) == 1
    assert captured[0][0] == "thread/resume"
    assert captured[0][1]["developerInstructions"] == "use tools"


def test_list_threads_parses_response_and_filters():
    client = CodexAppServerClient(codex_cmd="codex")
    captured: list[tuple[str, dict[str, object]]] = []

    async def fake_start():
        return None

    async def fake_request(method, params, *, timeout_s=None):
        del timeout_s
        captured.append((method, params))
        return {
            "data": [
                {
                    "id": "thread-1",
                    "cwd": "/tmp/work",
                    "source": "cli",
                    "updatedAt": 100,
                    "createdAt": 90,
                    "preview": "hello",
                }
            ],
            "nextCursor": "cursor-1",
        }

    client.start = fake_start  # type: ignore[method-assign]
    client._request = fake_request  # type: ignore[method-assign]

    result = asyncio.run(
        client.list_threads(
            limit=7,
            cwd="/tmp/work",
            source_kinds=["cli", "vscode"],
            cursor="cursor-0",
            archived=False,
            sort_key="updated_at",
        )
    )

    assert captured == [
        (
            "thread/list",
            {
                "limit": 7,
                "cwd": "/tmp/work",
                "sourceKinds": ["cli", "vscode"],
                "cursor": "cursor-0",
                "archived": False,
                "sortKey": "updated_at",
            },
        )
    ]
    assert result.next_cursor == "cursor-1"
    assert len(result.threads) == 1
    thread = result.threads[0]
    assert thread.thread_id == "thread-1"
    assert thread.cwd == "/tmp/work"
    assert thread.source == "cli"
    assert thread.updated_at == 100
    assert thread.created_at == 90
    assert thread.preview == "hello"


def test_list_threads_skips_invalid_entries():
    client = CodexAppServerClient(codex_cmd="codex")

    async def fake_request(method, params, *, timeout_s=None):
        del method, params, timeout_s
        return {
            "data": [
                None,
                {},
                {"id": ""},
                {"id": "thread-2", "updatedAt": "bad", "cwd": 7, "source": True},
            ],
            "nextCursor": 123,
        }

    client._request = fake_request  # type: ignore[method-assign]

    result = asyncio.run(client.list_threads(limit=0))
    assert result.next_cursor is None
    assert [thread.thread_id for thread in result.threads] == ["thread-2"]
    thread = result.threads[0]
    assert thread.updated_at is None
    assert thread.cwd is None
    assert thread.source is None


def test_list_threads_rejects_invalid_sort_key():
    client = CodexAppServerClient(codex_cmd="codex")

    with pytest.raises(ValueError):
        asyncio.run(client.list_threads(sort_key="newest"))


def test_interrupt_active_turn_returns_false_without_active_turn():
    client = CodexAppServerClient(codex_cmd="codex")
    assert asyncio.run(client.interrupt_active_turn()) is False


def test_interrupt_active_turn_sends_interrupt_request():
    client = CodexAppServerClient(codex_cmd="codex")
    captured: list[tuple[str, dict[str, object]]] = []

    async def fake_request(method, params, *, timeout_s=None):
        del timeout_s
        captured.append((method, params))
        return {}

    client._request = fake_request  # type: ignore[method-assign]

    async def run_case():
        loop = asyncio.get_running_loop()
        done = loop.create_future()
        client._active_turn = SimpleNamespace(  # type: ignore[assignment]
            thread_id="thread-1",
            turn_id="turn-1",
            done=done,
        )
        return await client.interrupt_active_turn()

    ok = asyncio.run(run_case())
    assert ok is True
    assert captured == [
        ("turn/interrupt", {"threadId": "thread-1", "turnId": "turn-1"})
    ]
