from __future__ import annotations

import asyncio
import json
import logging
import shlex
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)

# Some app-server JSON-RPC messages (notably large thread listings) can exceed the
# default asyncio StreamReader 64 KiB line limit. Raise it to avoid reader crashes.
APP_SERVER_STREAM_LIMIT = 4 * 1024 * 1024
YOLO_FLAG = "--yolo"
DANGEROUS_BYPASS_FLAG = "--dangerously-bypass-approvals-and-sandbox"
YOLO_SANDBOX_MODE = "danger-full-access"


class CodexError(RuntimeError):
    """Base error for Codex App Server failures."""


class CodexRPCError(CodexError):
    """JSON-RPC error returned by the server."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"RPC error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


@dataclass(slots=True)
class CodexProgressEvent:
    item_id: str
    kind: str
    phase: str
    title: str
    ok: bool | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CodexTurnResult:
    thread_id: str
    turn_id: str
    message: str | None
    error: str | None
    status: str


@dataclass(slots=True)
class CodexThreadSummary:
    thread_id: str
    cwd: str | None = None
    source: str | None = None
    updated_at: int | None = None
    created_at: int | None = None
    preview: str | None = None


@dataclass(slots=True)
class CodexThreadListResult:
    threads: list[CodexThreadSummary]
    next_cursor: str | None = None


@dataclass
class _ActiveTurn:
    thread_id: str
    turn_id: str
    on_progress: Callable[[CodexProgressEvent], Awaitable[None]] | None
    done: asyncio.Future[CodexTurnResult]
    latest_message: str | None = None
    message_deltas: dict[str, list[str]] = field(default_factory=dict)


class CodexAppServerClient:
    """Long-lived JSON-RPC client for `codex app-server`."""

    def __init__(
        self,
        *,
        codex_cmd: str = "codex",
        client_name: str = "djinn",
        client_version: str = "0.1.0",
        initialize_timeout_s: float = 20.0,
        request_timeout_s: float = 60.0,
        turn_timeout_s: float = 1800.0,
    ) -> None:
        cmd_parts = shlex.split(codex_cmd)
        if not cmd_parts:
            raise ValueError("codex_cmd must not be empty")
        if YOLO_FLAG not in cmd_parts and DANGEROUS_BYPASS_FLAG not in cmd_parts:
            cmd_parts.append(YOLO_FLAG)

        self._cmd_parts = cmd_parts
        self._client_name = client_name
        self._client_version = client_version
        self._initialize_timeout_s = initialize_timeout_s
        self._request_timeout_s = request_timeout_s
        self._turn_timeout_s = turn_timeout_s

        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=50)

        self._request_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()

        self._active_turn: _ActiveTurn | None = None

    async def start(self) -> None:
        async with self._start_lock:
            if self._is_running():
                return

            self._stderr_lines.clear()
            self._proc = await asyncio.create_subprocess_exec(
                *self._cmd_parts,
                "app-server",
                "--listen",
                "stdio://",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=APP_SERVER_STREAM_LIMIT,
            )
            if self._proc.stdin is None or self._proc.stdout is None:
                raise CodexError("codex app-server failed to open stdio pipes")

            self._reader_task = asyncio.create_task(self._reader_loop())
            if self._proc.stderr is not None:
                self._stderr_task = asyncio.create_task(self._stderr_loop())

            try:
                await asyncio.wait_for(
                    self._request(
                        "initialize",
                        {
                            "clientInfo": {
                                "name": self._client_name,
                                "version": self._client_version,
                            },
                            "capabilities": {
                                "experimentalApi": True,
                            },
                        },
                        timeout_s=self._initialize_timeout_s,
                    ),
                    timeout=self._initialize_timeout_s,
                )
            except Exception as exc:
                await self.close()
                raise CodexError(f"failed to initialize codex app-server: {exc}") from exc

    async def close(self) -> None:
        if self._proc is None:
            return

        proc = self._proc
        self._proc = None

        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except TimeoutError:
                proc.kill()
                await proc.wait()

        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._reader_task = None
        self._stderr_task = None

        failure = CodexError("codex app-server closed")
        self._fail_pending_requests(failure)
        self._fail_active_turn(failure)

    async def ensure_thread(
        self,
        *,
        thread_id: str | None,
        cwd: str,
        developer_instructions: str | None = None,
    ) -> tuple[str, str]:
        await self.start()

        if thread_id:
            resume_params: dict[str, Any] = {
                "threadId": thread_id,
                "cwd": cwd,
                "approvalPolicy": "never",
                "sandbox": YOLO_SANDBOX_MODE,
            }
            if developer_instructions:
                resume_params["developerInstructions"] = developer_instructions
            try:
                response = await self._request(
                    "thread/resume",
                    resume_params,
                )
            except CodexRPCError as exc:
                LOGGER.warning(
                    "thread/resume failed for %s (code=%s): %s; starting new thread",
                    thread_id,
                    exc.code,
                    exc.message,
                )
            else:
                resumed_id, resolved_cwd = self._extract_thread_info(response, cwd)
                return resumed_id, resolved_cwd

        start_params: dict[str, Any] = {
            "cwd": cwd,
            "approvalPolicy": "never",
            "sandbox": YOLO_SANDBOX_MODE,
        }
        if developer_instructions:
            start_params["developerInstructions"] = developer_instructions
        response = await self._request("thread/start", start_params)
        started_id, resolved_cwd = self._extract_thread_info(response, cwd)
        return started_id, resolved_cwd

    async def run_turn(
        self,
        *,
        prompt: str,
        thread_id: str | None,
        cwd: str,
        developer_instructions: str | None = None,
        on_progress: Callable[[CodexProgressEvent], Awaitable[None]] | None = None,
    ) -> CodexTurnResult:
        async with self._turn_lock:
            resolved_thread_id, _ = await self.ensure_thread(
                thread_id=thread_id,
                cwd=cwd,
                developer_instructions=developer_instructions,
            )

            response = await self._request(
                "turn/start",
                {
                    "threadId": resolved_thread_id,
                    "input": [{"type": "text", "text": prompt}],
                },
            )
            turn = response.get("turn") if isinstance(response, dict) else None
            if not isinstance(turn, dict):
                raise CodexError("turn/start returned an invalid response")
            turn_id = turn.get("id")
            if not isinstance(turn_id, str) or not turn_id:
                raise CodexError("turn/start response missing turn id")

            loop = asyncio.get_running_loop()
            active_turn = _ActiveTurn(
                thread_id=resolved_thread_id,
                turn_id=turn_id,
                on_progress=on_progress,
                done=loop.create_future(),
            )

            self._active_turn = active_turn

            # Handle edge case where turn is already terminal on response.
            status = turn.get("status")
            if isinstance(status, str) and status in {"completed", "failed", "interrupted"}:
                error_msg = self._extract_turn_error(turn)
                active_turn.done.set_result(
                    CodexTurnResult(
                        thread_id=resolved_thread_id,
                        turn_id=turn_id,
                        message=active_turn.latest_message,
                        error=error_msg,
                        status=status,
                    )
                )

            try:
                result = await asyncio.wait_for(active_turn.done, timeout=self._turn_timeout_s)
                return result
            finally:
                if self._active_turn is active_turn:
                    self._active_turn = None

    async def list_threads(
        self,
        *,
        limit: int = 10,
        cwd: str | None = None,
        source_kinds: Sequence[str] | None = None,
        cursor: str | None = None,
        archived: bool | None = None,
        sort_key: str | None = None,
    ) -> CodexThreadListResult:
        params: dict[str, Any] = {"limit": max(1, int(limit))}
        if cwd is not None:
            params["cwd"] = cwd
        if source_kinds is not None:
            params["sourceKinds"] = [kind for kind in source_kinds if isinstance(kind, str)]
        if cursor is not None:
            params["cursor"] = cursor
        if archived is not None:
            params["archived"] = archived
        if sort_key is not None:
            if sort_key not in {"created_at", "updated_at"}:
                raise ValueError("sort_key must be 'created_at' or 'updated_at'")
            params["sortKey"] = sort_key

        response = await self._request("thread/list", params)
        if not isinstance(response, dict):
            raise CodexError("thread/list returned an invalid response")

        data = response.get("data")
        if not isinstance(data, list):
            raise CodexError("thread/list response missing data list")

        threads: list[CodexThreadSummary] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            thread_id = item.get("id")
            if not isinstance(thread_id, str) or not thread_id:
                continue

            updated_at = item.get("updatedAt")
            if not isinstance(updated_at, int):
                updated_at = None

            created_at = item.get("createdAt")
            if not isinstance(created_at, int):
                created_at = None

            cwd_value = item.get("cwd")
            if not isinstance(cwd_value, str) or not cwd_value:
                cwd_value = None

            source = item.get("source")
            if not isinstance(source, str) or not source:
                source = None

            preview = item.get("preview")
            if not isinstance(preview, str) or not preview:
                preview = None

            threads.append(
                CodexThreadSummary(
                    thread_id=thread_id,
                    cwd=cwd_value,
                    source=source,
                    updated_at=updated_at,
                    created_at=created_at,
                    preview=preview,
                )
            )

        next_cursor = response.get("nextCursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            next_cursor = None

        return CodexThreadListResult(threads=threads, next_cursor=next_cursor)

    def has_active_turn(self) -> bool:
        active = self._active_turn
        return active is not None and not active.done.done()

    async def interrupt_active_turn(self) -> bool:
        active = self._active_turn
        if active is None or active.done.done():
            return False

        try:
            await self._request(
                "turn/interrupt",
                {
                    "threadId": active.thread_id,
                    "turnId": active.turn_id,
                },
                timeout_s=min(10.0, self._request_timeout_s),
            )
            return True
        except Exception as exc:
            LOGGER.warning(
                "failed to interrupt active turn %s/%s: %s",
                active.thread_id,
                active.turn_id,
                exc,
            )
            return False

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Any:
        if not self._is_running():
            await self.start()
        if self._proc is None:
            raise CodexError("codex app-server is not running")

        self._request_id += 1
        request_id = self._request_id

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future

        await self._send_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )

        request_timeout = timeout_s if timeout_s is not None else self._request_timeout_s
        try:
            return await asyncio.wait_for(future, timeout=request_timeout)
        finally:
            self._pending.pop(request_id, None)

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise CodexError("codex app-server stdin is unavailable")
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        async with self._write_lock:
            self._proc.stdin.write((raw + "\n").encode("utf-8"))
            await self._proc.stdin.drain()

    async def _reader_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        try:
            while True:
                raw = await self._proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("ignoring invalid JSON from app-server: %s", line)
                    continue
                if not isinstance(message, dict):
                    continue
                await self._handle_message(message)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            LOGGER.exception("app-server reader loop failed: %s", exc)
        finally:
            failure = CodexError(self._build_shutdown_error("app-server stdout closed"))
            self._fail_pending_requests(failure)
            self._fail_active_turn(failure)

    async def _stderr_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stderr is not None
        try:
            while True:
                raw = await self._proc.stderr.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                self._stderr_lines.append(line)
                LOGGER.warning("codex app-server stderr: %s", line)
        except asyncio.CancelledError:
            return

    async def _handle_message(self, message: dict[str, Any]) -> None:
        if "method" in message:
            method = message.get("method")
            if not isinstance(method, str):
                return
            if "id" in message and "result" not in message and "error" not in message:
                asyncio.create_task(self._handle_server_request(message))
            else:
                params = message.get("params")
                if not isinstance(params, dict):
                    params = {}
                await self._handle_notification(method, params)
            return

        if "id" not in message:
            return

        request_id = message.get("id")
        if not isinstance(request_id, int):
            return
        future = self._pending.get(request_id)
        if future is None or future.done():
            return

        if "error" in message and isinstance(message.get("error"), dict):
            error = message["error"]
            code = error.get("code")
            msg = error.get("message")
            data = error.get("data")
            if not isinstance(code, int):
                code = -32000
            if not isinstance(msg, str):
                msg = "unknown RPC error"
            future.set_exception(CodexRPCError(code, msg, data))
            return

        future.set_result(message.get("result"))

    async def _handle_server_request(self, request: dict[str, Any]) -> None:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params")

        if not isinstance(method, str):
            return
        if not isinstance(params, dict):
            params = {}

        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            if request_id is None:
                return
            await self._send_json(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"decision": "accept"},
                }
            )
            return

        if method == "item/tool/requestUserInput":
            if request_id is None:
                return
            LOGGER.warning("received tool user input request; returning empty answers")
            await self._send_json(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"answers": {}},
                }
            )
            return

        if request_id is None:
            return
        await self._send_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unsupported request method: {method}",
                },
            }
        )

    async def _handle_notification(self, method: str, params: dict[str, Any]) -> None:
        active = self._active_turn
        if active is None:
            return

        if method == "error":
            message = params.get("message")
            if isinstance(message, str) and message:
                if not active.done.done():
                    active.done.set_result(
                        CodexTurnResult(
                            thread_id=active.thread_id,
                            turn_id=active.turn_id,
                            message=active.latest_message,
                            error=message,
                            status="failed",
                        )
                    )
            return

        if not self._is_for_active_turn(active, params):
            return

        if method == "turn/completed":
            turn = params.get("turn")
            if not isinstance(turn, dict):
                return
            status = turn.get("status")
            if not isinstance(status, str):
                status = "failed"
            error_msg = self._extract_turn_error(turn)
            message = active.latest_message
            if not message and active.message_deltas:
                # If we somehow missed item/completed for agent messages, recover from deltas.
                _, chunks = next(reversed(active.message_deltas.items()))
                recovered = "".join(chunks).strip()
                message = recovered or None
            if not active.done.done():
                active.done.set_result(
                    CodexTurnResult(
                        thread_id=active.thread_id,
                        turn_id=active.turn_id,
                        message=message,
                        error=error_msg,
                        status=status,
                    )
                )
            return

        if method == "item/started":
            item = params.get("item")
            event = self._event_from_item(item, phase="started")
            if event is not None and active.on_progress is not None:
                await active.on_progress(event)
            return

        if method == "item/completed":
            item = params.get("item")
            if isinstance(item, dict):
                item_type = item.get("type")
                item_id = item.get("id")
                if item_type == "agentMessage" and isinstance(item_id, str):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        active.latest_message = text.strip()
                    elif item_id in active.message_deltas:
                        recovered = "".join(active.message_deltas[item_id]).strip()
                        if recovered:
                            active.latest_message = recovered
            event = self._event_from_item(item, phase="completed")
            if event is not None and active.on_progress is not None:
                await active.on_progress(event)
            return

        if method == "item/agentMessage/delta":
            item_id = params.get("itemId")
            delta = params.get("delta")
            if isinstance(item_id, str) and isinstance(delta, str):
                active.message_deltas.setdefault(item_id, []).append(delta)
                if active.on_progress is not None:
                    await active.on_progress(
                        CodexProgressEvent(
                            item_id=item_id,
                            kind="agent_message",
                            phase="delta",
                            title="message",
                            detail={"delta": delta},
                        )
                    )
            return

        if method == "item/commandExecution/outputDelta":
            item_id = params.get("itemId")
            delta = params.get("delta")
            if (
                isinstance(item_id, str)
                and isinstance(delta, str)
                and active.on_progress is not None
            ):
                await active.on_progress(
                    CodexProgressEvent(
                        item_id=item_id,
                        kind="command",
                        phase="delta",
                        title="command output",
                        detail={"delta": delta},
                    )
                )
            return

        if method == "item/fileChange/outputDelta":
            item_id = params.get("itemId")
            delta = params.get("delta")
            if (
                isinstance(item_id, str)
                and isinstance(delta, str)
                and active.on_progress is not None
            ):
                await active.on_progress(
                    CodexProgressEvent(
                        item_id=item_id,
                        kind="file_change",
                        phase="delta",
                        title="file change",
                        detail={"delta": delta},
                    )
                )
            return

        if method == "item/mcpToolCall/progress":
            item_id = params.get("itemId")
            delta = params.get("delta")
            if (
                isinstance(item_id, str)
                and isinstance(delta, str)
                and active.on_progress is not None
            ):
                await active.on_progress(
                    CodexProgressEvent(
                        item_id=item_id,
                        kind="tool",
                        phase="delta",
                        title="tool progress",
                        detail={"delta": delta},
                    )
                )
            return

    @staticmethod
    def _extract_turn_error(turn: dict[str, Any]) -> str | None:
        error = turn.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        return None

    @staticmethod
    def _is_for_active_turn(active: _ActiveTurn, params: dict[str, Any]) -> bool:
        thread_id = params.get("threadId")
        turn_id = params.get("turnId")
        if method_turn := params.get("turn"):
            if isinstance(method_turn, dict):
                maybe_id = method_turn.get("id")
                if isinstance(maybe_id, str):
                    turn_id = maybe_id
        return thread_id == active.thread_id and turn_id == active.turn_id

    @staticmethod
    def _short_tool_name(server: str | None, tool: str | None) -> str:
        parts = [part for part in (server, tool) if isinstance(part, str) and part]
        return ".".join(parts) if parts else "tool"

    @classmethod
    def _event_from_item(
        cls,
        item: Any,
        *,
        phase: str,
    ) -> CodexProgressEvent | None:
        if not isinstance(item, dict):
            return None

        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            return None

        item_type = item.get("type")
        if not isinstance(item_type, str):
            item_type = "unknown"

        if item_type == "commandExecution":
            title = str(item.get("command") or "")
            ok = None
            if phase == "completed":
                status = item.get("status")
                ok = status == "completed"
                exit_code = item.get("exitCode")
                if isinstance(exit_code, int):
                    ok = ok and exit_code == 0
            return CodexProgressEvent(
                item_id=item_id,
                kind="command",
                phase=phase,
                title=title,
                ok=ok,
                detail={"status": item.get("status")},
            )

        if item_type == "mcpToolCall":
            title = cls._short_tool_name(item.get("server"), item.get("tool"))
            ok = None
            if phase == "completed":
                status = item.get("status")
                error = item.get("error")
                ok = status == "completed" and error is None
            return CodexProgressEvent(
                item_id=item_id,
                kind="tool",
                phase=phase,
                title=title,
                ok=ok,
                detail={"status": item.get("status")},
            )

        if item_type == "fileChange":
            changes = item.get("changes")
            normalized_changes: list[dict[str, str]] = []
            if isinstance(changes, list):
                for change in changes:
                    if not isinstance(change, dict):
                        continue
                    path = change.get("path")
                    if not isinstance(path, str) or not path:
                        continue
                    normalized: dict[str, str] = {"path": path}
                    kind = change.get("kind")
                    if isinstance(kind, dict):
                        kind_type = kind.get("type")
                        if isinstance(kind_type, str) and kind_type:
                            normalized["kind"] = kind_type
                    normalized_changes.append(normalized)
            ok = None
            if phase == "completed":
                ok = item.get("status") == "completed"
            return CodexProgressEvent(
                item_id=item_id,
                kind="file_change",
                phase=phase,
                title="files",
                ok=ok,
                detail={"changes": normalized_changes, "status": item.get("status")},
            )

        title = ""
        if item_type == "agentMessage":
            text = item.get("text")
            title = text if isinstance(text, str) else ""
            return CodexProgressEvent(
                item_id=item_id,
                kind="agent_message",
                phase=phase,
                title=title,
                ok=None,
            )

        return CodexProgressEvent(
            item_id=item_id,
            kind=item_type,
            phase=phase,
            title=title,
            ok=None,
        )

    @staticmethod
    def _extract_thread_info(response: Any, fallback_cwd: str) -> tuple[str, str]:
        if not isinstance(response, dict):
            raise CodexError("thread response has unexpected type")
        thread = response.get("thread")
        if not isinstance(thread, dict):
            raise CodexError("thread response missing thread payload")
        thread_id = thread.get("id")
        if not isinstance(thread_id, str) or not thread_id:
            raise CodexError("thread response missing thread id")
        cwd = response.get("cwd")
        if not isinstance(cwd, str) or not cwd:
            cwd = fallback_cwd
        return thread_id, cwd

    def _is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def _fail_pending_requests(self, exc: Exception) -> None:
        for request_id, future in list(self._pending.items()):
            if not future.done():
                future.set_exception(exc)
            self._pending.pop(request_id, None)

    def _fail_active_turn(self, exc: Exception) -> None:
        active = self._active_turn
        if active is None or active.done.done():
            return
        active.done.set_exception(exc)

    def _build_shutdown_error(self, prefix: str) -> str:
        if not self._stderr_lines:
            return prefix
        joined = "\n".join(self._stderr_lines)
        return f"{prefix}\n{joined}"
