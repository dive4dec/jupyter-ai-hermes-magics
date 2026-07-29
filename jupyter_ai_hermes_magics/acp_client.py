"""ACP client for %%hermes magic — persistent Hermes subprocess.

Spawns `hermes acp` ONCE as a persistent subprocess and communicates via
JSON-RPC over stdio (the same architecture jupyter-ai uses). This eliminates
the ~15s per-query startup overhead of `hermes chat` subprocesses.

Architecture:
    AcpConnection (singleton)
        ├── background asyncio event loop (daemon thread)
        ├── hermes acp subprocess (persistent)
        ├── ClientSideConnection (JSON-RPC over stdio)
        └── MagicAcpClient (handles session/update events)

Usage from synchronous cell magic:
    conn = AcpConnection.get()
    conn.initialize()  # spawns subprocess, creates session
    response = conn.prompt("hello", on_chunk=callback, ...)
    conn.cancel()  # stop generation
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MagicAcpClient:
    """Minimal ACP Client for %%hermes magic.

    Handles session/update events (streaming text, tool calls) and
    request_permission (accept/reject tool calls).
    """

    def __init__(self):
        self._callbacks: dict[str, Callable] = {}
        self._permission_futures: dict[str, asyncio.Future] = {}
        self._permission_loops: dict[str, asyncio.AbstractEventLoop] = {}
        self._tool_calls: dict[str, dict] = {}  # tool_call_id → state
        self._accumulated_text: list[str] = []

    def set_callbacks(
        self,
        on_chunk: Callable[[str], None] | None = None,
        on_tool_call: Callable[[dict], None] | None = None,
        on_permission: Callable[[dict], None] | None = None,
    ):
        self._callbacks = {
            "on_chunk": on_chunk,
            "on_tool_call": on_tool_call,
            "on_permission": on_permission,
        }

    def get_accumulated_text(self) -> str:
        return "".join(self._accumulated_text)

    def reset(self):
        self._accumulated_text.clear()
        self._tool_calls.clear()
        self._callbacks.clear()
        # Don't clear _permission_futures — may still be pending

    # ── ACP Client interface (called by agent over stdio) ──

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        """Handle streaming events from Hermes."""
        from acp.schema import AgentMessageChunk, ToolCallStart, ToolCallProgress

        if isinstance(update, AgentMessageChunk):
            text = ""
            content = update.content
            if hasattr(content, "text"):
                text = content.text
            elif hasattr(content, "uri"):
                text = content.uri or ""
            else:
                text = str(content)

            if text:
                self._accumulated_text.append(text)
                if cb := self._callbacks.get("on_chunk"):
                    try:
                        cb(text)
                    except Exception:
                        logger.debug("on_chunk callback failed", exc_info=True)

        elif isinstance(update, ToolCallStart):
            tc_id = update.tool_call_id
            title = update.title or "tool"
            kind = update.kind or "other"
            status = update.status or "pending"
            self._tool_calls[tc_id] = {
                "id": tc_id,
                "title": title,
                "kind": kind,
                "status": status,
            }
            if cb := self._callbacks.get("on_tool_call"):
                try:
                    cb({"type": "start", "id": tc_id, "title": title,
                        "kind": kind, "status": status})
                except Exception:
                    logger.debug("on_tool_call callback failed", exc_info=True)

        elif isinstance(update, ToolCallProgress):
            tc_id = update.tool_call_id
            status = update.status or "in_progress"
            if tc_id in self._tool_calls:
                self._tool_calls[tc_id]["status"] = status
            if cb := self._callbacks.get("on_tool_call"):
                try:
                    cb({"type": "progress", "id": tc_id, "status": status})
                except Exception:
                    logger.debug("on_tool_call callback failed", exc_info=True)

    async def request_permission(
        self, options: list, session_id: str, tool_call: Any, **kwargs: Any
    ) -> Any:
        """Hermes asks permission for a tool call. Show Accept/Reject buttons."""
        from acp.schema import RequestPermissionResponse, AllowedOutcome, DeniedOutcome

        tc_id = tool_call.tool_call_id
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._permission_futures[tc_id] = future
        self._permission_loops[tc_id] = loop  # store loop for cross-thread resolve

        perm_options = []
        for o in options:
            perm_options.append({
                "id": o.option_id,
                "name": o.name,
                "kind": getattr(o, "kind", ""),
            })

        if cb := self._callbacks.get("on_permission"):
            try:
                cb({
                    "tool_call_id": tc_id,
                    "tool_name": getattr(tool_call, "title", "tool"),
                    "options": perm_options,
                })
            except Exception:
                logger.debug("on_permission callback failed", exc_info=True)

        # Suspend until user resolves (or timeout)
        try:
            selected = await asyncio.wait_for(future, timeout=300.0)
        except asyncio.TimeoutError:
            return RequestPermissionResponse(
                outcome=DeniedOutcome(outcome="cancelled")
            )

        if selected is None:
            return RequestPermissionResponse(
                outcome=DeniedOutcome(outcome="cancelled")
            )

        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=selected, outcome="selected")
        )

    def resolve_permission(self, tool_call_id: str, option_id: str | None):
        """Called from main thread when user clicks Accept/Reject.

        Must use call_soon_threadsafe because the future lives in the
        background asyncio event loop, not the main thread.
        """
        future = self._permission_futures.get(tool_call_id)
        loop = self._permission_loops.get(tool_call_id)
        if future and not future.done() and loop and not loop.is_closed():
            loop.call_soon_threadsafe(future.set_result, option_id)
        # Cleanup
        self._permission_futures.pop(tool_call_id, None)
        self._permission_loops.pop(tool_call_id, None)

    async def read_text_file(self, path: str, session_id: str, **kwargs: Any):
        """Read a file — delegated to the Jupyter server filesystem."""
        from acp.schema import ReadTextFileResponse, RequestError
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            return ReadTextFileResponse(content=text)
        except FileNotFoundError:
            raise RequestError.resource_not_found(path)
        except Exception as e:
            raise RequestError.internal_error({"path": path, "error": str(e)})

    async def write_text_file(self, content: str, path: str, session_id: str, **kwargs: Any):
        """Write a file — delegated to the Jupyter server filesystem."""
        from acp.schema import WriteTextFileResponse, RequestError
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return WriteTextFileResponse()
        except Exception as e:
            raise RequestError.internal_error({"path": path, "error": str(e)})

    async def create_terminal(self, *args, **kwargs):
        from acp.schema import RequestError
        raise RequestError.method_not_found("create_terminal")

    async def terminal_output(self, *args, **kwargs):
        from acp.schema import RequestError
        raise RequestError.method_not_found("terminal_output")

    async def release_terminal(self, *args, **kwargs):
        from acp.schema import RequestError
        raise RequestError.method_not_found("release_terminal")

    async def wait_for_terminal_exit(self, *args, **kwargs):
        from acp.schema import RequestError
        raise RequestError.method_not_found("wait_for_terminal_exit")

    async def kill_terminal(self, *args, **kwargs):
        from acp.schema import RequestError
        raise RequestError.method_not_found("kill_terminal")

    async def ext_method(self, method: str, params: dict):
        from acp.schema import RequestError
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict) -> None:
        pass

    def on_connect(self, conn) -> None:
        pass


class AcpConnection:
    """Manages a persistent 'hermes acp' subprocess + asyncio event loop.

    Singleton — only one Hermes subprocess per kernel. All %%hermes calls
    reuse the same connection and session.
    """

    _instance: Optional["AcpConnection"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._conn = None  # ClientSideConnection
        self._session_id: Optional[str] = None
        self._client: Optional[MagicAcpClient] = None
        self._initialized = False
        self._initializing = False
        self._init_error: Optional[str] = None
        self._init_lock = threading.Lock()
        self._prompt_in_progress = False
        self._cancel_event = threading.Event()

    @classmethod
    def get(cls) -> "AcpConnection":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Force a fresh connection (e.g., after %hermes reset)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._shutdown()
            cls._instance = None

    def _ensure_loop(self):
        if self._loop is not None and not self._loop.is_closed():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop_forever, daemon=True, name="hermes-acp-loop"
        )
        self._thread.start()

    def _run_loop_forever(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro, timeout: float = 120) -> Any:
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ── Public API ──

    def initialize(self) -> tuple[bool, str]:
        """Spawn hermes acp, connect, create session.

        Returns (success, message).
        Thread-safe — if init is already in progress (e.g. started by
        %load_ext background thread), waits for it to finish instead of
        spawning a second subprocess.
        """
        with self._init_lock:
            if self._initialized:
                return True, "Already initialized"
            if self._init_error:
                return False, self._init_error
            if self._initializing:
                # Another thread is doing the init — wait outside the lock
                do_init = False
            else:
                # We are the one to init
                self._initializing = True
                do_init = True

        if not do_init:
            # Wait for the other thread's init to complete
            while self._initializing and not self._initialized and not self._init_error:
                import time as _time
                _time.sleep(0.2)
            if self._initialized:
                sid = self._session_id or ""
                return True, f"Session {sid[:12]}…"
            if self._init_error:
                return False, self._init_error
            return False, "Init completed but no session"

        # We are the initializer — do the actual work (outside the lock
        # so other threads can see _initializing=True and wait)
        try:
            result = self._run_async(self._async_init(), timeout=60)
            self._initialized = True
            sid = self._session_id or ""
            return True, f"Session {sid[:12]}…"
        except Exception as e:
            self._init_error = str(e)
            logger.error("ACP init failed: %s", e, exc_info=True)
            return False, str(e)
        finally:
            with self._init_lock:
                self._initializing = False

    async def _async_init(self):
        from acp import PROTOCOL_VERSION, connect_to_agent
        from acp.schema import (
            ClientCapabilities, FileSystemCapabilities, Implementation,
        )

        # Find hermes binary
        hermes_bin = os.environ.get("HERMES_BIN_PATH") or shutil.which("hermes")
        if not hermes_bin:
            raise RuntimeError("Hermes binary not found")

        # Spawn hermes acp subprocess
        self._proc = await asyncio.create_subprocess_exec(
            hermes_bin, "acp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=50 * 1024 * 1024,
            start_new_session=True,
        )

        # Create client + connect
        self._client = MagicAcpClient()
        self._conn = connect_to_agent(
            self._client,
            self._proc.stdin,
            self._proc.stdout,
        )

        # Initialize handshake
        init_response = await self._conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapabilities(
                    read_text_file=True,
                    write_text_file=True,
                ),
                terminal=False,
            ),
            client_info=Implementation(
                name="HermesMagic",
                title="Jupyter Hermes Magic",
                version="0.4.0",
            ),
        )

        # Create session — pass the Jupyter MCP server so Hermes has
        # add_cell, run_cell, read_notebook_cells tools available.
        cwd = os.getcwd()
        mcp_url = os.environ.get(
            "JUPYTER_MCP_URL", "http://localhost:3001/mcp"
        )
        from acp.schema import HttpMcpServer, HttpHeader
        jupyter_mcp = HttpMcpServer(
            type="http",
            name="jupyter",
            url=mcp_url,
            headers=[],
        )
        session = await self._conn.new_session(
            cwd=cwd, mcp_servers=[jupyter_mcp]
        )
        self._session_id = session.session_id

        logger.info("ACP initialized: session %s", self._session_id)

    def prompt(
        self,
        text: str,
        on_chunk: Callable[[str], None] | None = None,
        on_tool_call: Callable[[dict], None] | None = None,
        on_permission: Callable[[dict], None] | None = None,
    ) -> tuple[str, str]:
        """Send prompt to Hermes. Returns (response_text, stop_reason).

        Callbacks are called from the background asyncio thread.
        """
        if not self._initialized:
            ok, msg = self.initialize()
            if not ok:
                raise RuntimeError(f"ACP not initialized: {msg}")

        if self._prompt_in_progress:
            raise RuntimeError("A prompt is already in progress")

        self._prompt_in_progress = True
        self._cancel_event.clear()

        # Reset client state + set callbacks
        self._client.reset()
        self._client.set_callbacks(
            on_chunk=on_chunk,
            on_tool_call=on_tool_call,
            on_permission=on_permission,
        )

        try:
            response = self._run_async(
                self._async_prompt(text), timeout=300
            )
            return self._client.get_accumulated_text(), response.stop_reason
        except Exception as e:
            logger.error("Prompt failed: %s", e, exc_info=True)
            raise
        finally:
            self._prompt_in_progress = False

    async def _async_prompt(self, text: str):
        from acp.schema import TextContentBlock

        response = await self._conn.prompt(
            prompt=[TextContentBlock(text=text, type="text")],
            session_id=self._session_id,
        )
        return response

    def cancel(self):
        """Cancel the current prompt."""
        if not self._prompt_in_progress or not self._conn or not self._session_id:
            return
        self._cancel_event.set()
        try:
            self._run_async(
                self._conn.cancel(self._session_id), timeout=10
            )
        except Exception as e:
            logger.debug("Cancel failed: %s", e)

    def resolve_permission(self, tool_call_id: str, option_id: str | None):
        """Resolve a pending permission request (from main thread)."""
        if self._client:
            self._client.resolve_permission(tool_call_id, option_id)

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _shutdown(self):
        """Kill subprocess, stop loop."""
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.kill()
            except Exception:
                pass
        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        self._initialized = False
        self._session_id = None
        self._conn = None
        self._client = None
        self._proc = None
