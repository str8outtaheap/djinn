from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest

from codex import (
    CodexAppServerClient,
    CodexCommandExecResult,
    CodexCommandExecSize,
    CodexCommandExecSpec,
    CodexError,
)


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


def test_codex_cmd_appends_yolo_when_missing():
    client = CodexAppServerClient(codex_cmd="codex --search")
    assert client._cmd_parts == ["codex", "--search", "--yolo"]


def test_codex_cmd_does_not_duplicate_yolo():
    client = CodexAppServerClient(codex_cmd="codex --yolo --search")
    assert client._cmd_parts == ["codex", "--yolo", "--search"]


def test_codex_cmd_respects_dangerous_bypass_flag():
    client = CodexAppServerClient(
        codex_cmd="codex --dangerously-bypass-approvals-and-sandbox --search"
    )
    assert client._cmd_parts == [
        "codex",
        "--dangerously-bypass-approvals-and-sandbox",
        "--search",
    ]


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
            developer_instructions="be concise",
        )
    )

    assert thread_id == "thread-1"
    assert cwd == "/tmp"
    assert len(captured) == 1
    assert captured[0][0] == "thread/start"
    assert captured[0][1]["approvalPolicy"] == "never"
    assert captured[0][1]["sandbox"] == "danger-full-access"
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
            developer_instructions="use tools",
        )
    )

    assert thread_id == "thread-2"
    assert cwd == "/workspace"
    assert len(captured) == 1
    assert captured[0][0] == "thread/resume"
    assert captured[0][1]["approvalPolicy"] == "never"
    assert captured[0][1]["sandbox"] == "danger-full-access"
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


def test_command_exec_spec_to_rpc_params_buffered_defaults():
    spec = CodexCommandExecSpec(command=("git", "status"), cwd="/tmp/project")

    assert spec.to_rpc_params() == {
        "command": ["git", "status"],
        "cwd": "/tmp/project",
    }


def test_command_exec_spec_tty_implies_streaming_and_requires_process_id():
    spec = CodexCommandExecSpec(
        command=("bash",),
        process_id="proc-1",
        tty=True,
        size=CodexCommandExecSize(rows=24, cols=80),
    )

    assert spec.to_rpc_params() == {
        "command": ["bash"],
        "processId": "proc-1",
        "tty": True,
        "streamStdin": True,
        "streamStdoutStderr": True,
        "size": {"rows": 24, "cols": 80},
    }


def test_command_exec_spec_requires_process_id_for_streaming():
    spec = CodexCommandExecSpec(command=("pytest",), stream_stdout_stderr=True)

    with pytest.raises(ValueError, match="process_id"):
        spec.to_rpc_params()


def test_command_exec_spec_rejects_single_string_command():
    spec = CodexCommandExecSpec(command="git status")

    with pytest.raises(ValueError, match="argv sequence"):
        spec.to_rpc_params()


def test_command_exec_spec_rejects_conflicting_limits():
    spec = CodexCommandExecSpec(
        command=("pytest",),
        output_bytes_cap=4096,
        disable_output_cap=True,
    )

    with pytest.raises(ValueError, match="output_bytes_cap"):
        spec.to_rpc_params()


def test_command_exec_spec_validates_runtime_field_types():
    with pytest.raises(ValueError, match="process_id"):
        CodexCommandExecSpec(command=("pwd",), process_id=123).to_rpc_params()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cwd"):
        CodexCommandExecSpec(command=("pwd",), cwd=123).to_rpc_params()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="env values"):
        CodexCommandExecSpec(
            command=("pwd",),
            env={"PATH": 7},  # type: ignore[dict-item]
        ).to_rpc_params()


def test_command_exec_spec_defensively_copies_json_safe_sandbox_policy():
    sandbox_policy = {"mode": "workspace-write", "nested": {"network_access": True}}
    params = CodexCommandExecSpec(
        command=("pwd",),
        sandbox_policy=sandbox_policy,
    ).to_rpc_params()

    sandbox_policy["nested"]["network_access"] = False

    assert params["sandboxPolicy"] == {
        "mode": "workspace-write",
        "nested": {"network_access": True},
    }


def test_command_exec_spec_rejects_non_json_safe_sandbox_policy():
    with pytest.raises(ValueError, match="JSON-serializable"):
        CodexCommandExecSpec(
            command=("pwd",),
            sandbox_policy={"bad": object()},
        ).to_rpc_params()


def test_extract_command_exec_result_valid():
    result = CodexAppServerClient._extract_command_exec_result(
        {"exitCode": 0, "stdout": "ok\n", "stderr": ""}
    )

    assert result == CodexCommandExecResult(exit_code=0, stdout="ok\n", stderr="")


def test_extract_command_exec_result_invalid():
    with pytest.raises(CodexError, match="exitCode"):
        CodexAppServerClient._extract_command_exec_result({"stdout": "", "stderr": ""})


def test_extract_command_exec_output_delta_decodes_base64():
    delta = CodexAppServerClient._extract_command_exec_output_delta(
        {
            "processId": "proc-1",
            "stream": "stdout",
            "deltaBase64": base64.b64encode(b"hello\n").decode("ascii"),
            "capReached": False,
        }
    )

    assert delta is not None
    assert delta.process_id == "proc-1"
    assert delta.stream == "stdout"
    assert delta.data == b"hello\n"
    assert delta.text == "hello\n"
    assert delta.cap_reached is False


def test_extract_command_exec_output_delta_rejects_invalid_payload():
    delta = CodexAppServerClient._extract_command_exec_output_delta(
        {
            "processId": "proc-1",
            "stream": "stdout",
            "deltaBase64": "!!!not-base64!!!",
            "capReached": False,
        }
    )

    assert delta is None


def test_build_command_exec_terminate_params():
    terminate = CodexAppServerClient._build_command_exec_terminate_params("proc-1")

    assert terminate == {"processId": "proc-1"}


def test_new_command_exec_process_id_is_non_empty():
    process_id = CodexAppServerClient.new_command_exec_process_id()

    assert isinstance(process_id, str)
    assert process_id


def test_run_command_exec_returns_buffered_result_without_streaming():
    client = CodexAppServerClient(codex_cmd="codex")
    captured: list[tuple[str, dict[str, object], float | None]] = []

    async def fake_start():
        return None

    async def fake_request(method, params, *, timeout_s=None):
        captured.append((method, params, timeout_s))
        return {"exitCode": 0, "stdout": "ok\n", "stderr": ""}

    client.start = fake_start  # type: ignore[method-assign]
    client._request = fake_request  # type: ignore[method-assign]

    result = asyncio.run(
        client.run_command_exec(
            spec=CodexCommandExecSpec(command=("pwd",)),
        )
    )

    assert result == CodexCommandExecResult(exit_code=0, stdout="ok\n", stderr="")
    assert captured == [
        (
            "command/exec",
            {"command": ["pwd"]},
            client._turn_timeout_s,
        )
    ]


def test_run_command_exec_streams_output_and_merges_stream_buffers():
    client = CodexAppServerClient(codex_cmd="codex")
    streamed: list[tuple[str, str]] = []

    async def fake_start():
        return None

    async def fake_request(method, params, *, timeout_s=None):
        del timeout_s
        assert method == "command/exec"
        await client._handle_notification(
            "command/exec/outputDelta",
            {
                "processId": "proc-1",
                "stream": "stdout",
                "deltaBase64": base64.b64encode(b"hello ").decode("ascii"),
                "capReached": False,
            },
        )
        await client._handle_notification(
            "command/exec/outputDelta",
            {
                "processId": "proc-1",
                "stream": "stdout",
                "deltaBase64": base64.b64encode(b"world\n").decode("ascii"),
                "capReached": False,
            },
        )
        await client._handle_notification(
            "command/exec/outputDelta",
            {
                "processId": "proc-1",
                "stream": "stderr",
                "deltaBase64": base64.b64encode(b"warn\n").decode("ascii"),
                "capReached": False,
            },
        )
        return {"exitCode": 0, "stdout": "", "stderr": ""}

    client.start = fake_start  # type: ignore[method-assign]
    client._request = fake_request  # type: ignore[method-assign]

    async def on_output(delta):
        streamed.append((delta.stream, delta.text))

    result = asyncio.run(
        client.run_command_exec(
            spec=CodexCommandExecSpec(
                command=("bash", "-lc", "printf hello"),
                process_id="proc-1",
                stream_stdout_stderr=True,
            ),
            on_output=on_output,
        )
    )

    assert streamed == [
        ("stdout", "hello "),
        ("stdout", "world\n"),
        ("stderr", "warn\n"),
    ]
    assert result == CodexCommandExecResult(
        exit_code=0,
        stdout="hello world\n",
        stderr="warn\n",
    )


def test_run_command_exec_requires_streaming_when_on_output_is_set():
    client = CodexAppServerClient(codex_cmd="codex")

    async def fake_start():
        return None

    client.start = fake_start  # type: ignore[method-assign]

    async def on_output(_delta):
        return None

    with pytest.raises(ValueError, match="on_output requires"):
        asyncio.run(
            client.run_command_exec(
                spec=CodexCommandExecSpec(
                    command=("pwd",),
                    process_id="proc-1",
                ),
                on_output=on_output,
            )
        )


def test_terminate_command_exec_calls_expected_rpc_request():
    client = CodexAppServerClient(codex_cmd="codex")
    captured: list[tuple[str, dict[str, object]]] = []

    async def fake_request(method, params, *, timeout_s=None):
        del timeout_s
        captured.append((method, params))
        return {}

    client._request = fake_request  # type: ignore[method-assign]

    async def run_case():
        await client.terminate_command_exec("proc-1")

    asyncio.run(run_case())

    assert captured == [
        (
            "command/exec/terminate",
            {"processId": "proc-1"},
        ),
    ]
