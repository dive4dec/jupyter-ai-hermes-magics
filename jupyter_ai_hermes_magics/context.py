"""Notebook context gathering — lets Hermes see the notebook.

Calls the Jupyter MCP server directly via **synchronous HTTP** (urllib),
avoiding the ``jupyter-mcp-cli`` subprocess which hangs due to an asyncio
event-loop cleanup bug in ``streamablehttp_client``.

The MCP server runs inside JupyterLab at ``http://localhost:3001/mcp``
and speaks the MCP Streamable HTTP protocol (JSON-RPC over SSE).

**Protocol flow**:
  1. ``initialize`` → server returns ``Mcp-Session-Id`` header
  2. ``notifications/initialized`` → acknowledge (with session ID header)
  3. ``tools/call`` → include ``Mcp-Session-Id`` header

If the MCP server is unavailable, falls back to kernel introspection
(execution history + user namespace variables).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

# MCP server URL (same default as jupyter-mcp-cli)
MCP_URL = os.environ.get("JUPYTER_MCP_URL", "http://localhost:3001/mcp")

# Request counter for JSON-RPC IDs
_rpc_id = 0

# Cached MCP session ID (set after initialize, reused across calls)
_mcp_session_id: Optional[str] = None


def _post_mcp(payload: dict, extra_headers: dict | None = None) -> Optional[str]:
    """POST a JSON-RPC request to the MCP server, return raw response body.

    Includes the ``Mcp-Session-Id`` header if we have one (set after
    ``initialize``).
    """
    global _rpc_id
    _rpc_id += 1
    if "id" not in payload:
        payload["id"] = _rpc_id

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if _mcp_session_id:
        headers["Mcp-Session-Id"] = _mcp_session_id
    if extra_headers:
        headers.update(extra_headers)

    req = urllib.request.Request(MCP_URL, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        logger.debug("MCP HTTP error %d: %s", e.code, e.read().decode("utf-8", errors="replace")[:200])
        return None
    except Exception as e:
        logger.debug("MCP call error: %s", e)
        return None


def _parse_sse(raw: str) -> Optional[dict]:
    """Parse an SSE-formatted response, return the JSON from the first ``data:`` line."""
    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        try:
            return json.loads(line[6:])
        except json.JSONDecodeError:
            continue
    return None


def _extract_text(data: dict) -> Optional[str]:
    """Extract text content from a JSON-RPC result."""
    if "error" in data:
        logger.debug("MCP error: %s", data["error"])
        return None
    result = data.get("result", {})
    content = result.get("content", [])
    for block in content:
        if isinstance(block, dict) and "text" in block:
            return block["text"]
    return None


def _mcp_ensure_session() -> bool:
    """Initialize the MCP session if not already done.

    Returns True if the session is ready (or was already), False on failure.
    """
    global _mcp_session_id

    if _mcp_session_id is not None:
        return True

    # Step 1: initialize
    body = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hermes-magics", "version": "1.0"},
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        MCP_URL, data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            _mcp_session_id = resp.headers.get("Mcp-Session-Id")
            if not _mcp_session_id:
                logger.debug("MCP initialize: no session ID header")
                return False
    except Exception as e:
        logger.debug("MCP initialize failed: %s", e)
        return False

    # Step 2: send notifications/initialized (no response expected)
    # Notifications must NOT have an "id" field — they are fire-and-forget.
    notify_body = json.dumps({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }).encode("utf-8")
    notify_req = urllib.request.Request(
        MCP_URL, data=notify_body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Mcp-Session-Id": _mcp_session_id,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(notify_req, timeout=5) as resp:
            resp.read()  # drain
    except Exception:
        pass  # notifications don't need a response

    return True


def _mcp_call(tool_name: str, **args) -> Optional[str]:
    """Call a Jupyter MCP tool synchronously via direct HTTP.

    Returns the text content from the tool result, or None on failure.
    """
    if not _mcp_ensure_session():
        return None

    raw = _post_mcp({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": args},
    })
    if raw is None:
        return None

    data = _parse_sse(raw)
    if data is None:
        return None

    return _extract_text(data)


def _mcp_initialize() -> bool:
    """Check if the MCP server is reachable (and initialize session)."""
    return _mcp_ensure_session()


def gather_context() -> str:
    """Gather notebook context synchronously via direct MCP HTTP calls.

    Returns a formatted string containing:
      - Active notebook path
      - Active cell ID
      - **Cells up to and including the active cell** (so "the code above"
        refers to cells before the ``%%hermes`` cell, not cells below it)

    Falls back to kernel introspection if MCP server is unavailable.
    """
    parts: list[str] = []

    # 1. Check MCP server is reachable + initialize session
    if not _mcp_initialize():
        logger.debug("MCP server not reachable, using kernel fallback")
        return gather_kernel_context()

    # 2. Active notebook
    active_nb = _mcp_call("get_active_notebook")
    if not active_nb:
        return gather_kernel_context()

    # Strip quotes if MCP returns a JSON string
    active_nb = active_nb.strip().strip('"')
    parts.append(f"Active notebook: {active_nb}")

    # 3. Active cell ID
    active_cell_id = _mcp_call("get_active_cell_id", notebook_path=active_nb)
    if active_cell_id:
        active_cell_id = active_cell_id.strip().strip('"')
        parts.append(f"Active cell ID: {active_cell_id}")
    else:
        active_cell_id = None

    # 4. Read cells — only up to and including the active cell
    cells_raw = _mcp_call("read_notebook_cells", notebook_path=active_nb)
    if cells_raw:
        try:
            cells = json.loads(cells_raw)
            if cells and isinstance(cells, list):
                # Find the active cell index
                active_index = None
                if active_cell_id:
                    for i, cell in enumerate(cells):
                        if cell.get("cell_id") == active_cell_id:
                            active_index = i
                            break

                # Only include cells up to and including the active cell.
                # Cells BELOW the magic cell are irrelevant for "explain the
                # code above" — including them causes Hermes to explain
                # unrelated code below the question.
                if active_index is not None:
                    visible_cells = cells[:active_index + 1]
                else:
                    # Can't find active cell — show all (shouldn't happen)
                    visible_cells = cells

                parts.append(f"\nNotebook cells ({len(visible_cells)} shown"
                             f"{f', out of {len(cells)} total' if active_index is not None else ''}):")
                for i, cell in enumerate(visible_cells):
                    cell_type = cell.get("cellType", "code")
                    source = cell.get("source", "")
                    cell_id = cell.get("cell_id", "")
                    exec_count = cell.get("execution_count")

                    # Truncate very long cells
                    max_len = 2000
                    truncated = ""
                    if len(source) > max_len:
                        truncated = f" …({len(source)} chars, truncated)"
                        source = source[:max_len]

                    # Build cell marker
                    marker = f"[Cell {i}]"
                    if exec_count is not None:
                        marker += f" (exec={exec_count})"
                    marker += f" {cell_type}"
                    if cell_id == active_cell_id:
                        marker += " <- ACTIVE (this %%hermes cell)"
                    parts.append(f"\n{marker}:")
                    fence = "python" if cell_type == "code" else ""
                    parts.append(f"```{fence}")
                    parts.append(source)
                    parts.append(f"```{truncated}")
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug("Failed to parse cells JSON: %s", e)

    return "\n".join(parts) if parts else gather_kernel_context()


# ── Kernel fallback (any IPython, no MCP) ──────────────────────────────

def gather_kernel_context() -> str:
    """Gather context from the IPython kernel itself.

    Works in any IPython environment (plain ipython, VS Code notebooks).
    No MCP dependency. Provides execution history and user namespace.
    """
    parts: list[str] = []

    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return "(No IPython kernel available)"

        parts.append(f"Execution count: {ip.execution_count}")

        # User namespace variables
        user_vars = {}
        for k, v in ip.user_ns.items():
            if k.startswith("_") or k.startswith("magic_"):
                continue
            if k in ("In", "Out", "Exit", "exit", "quit", "get_ipython"):
                continue
            try:
                import sys
                if sys.getsizeof(v) > 10000:
                    user_vars[k] = f"<{type(v).__name__}, large>"
                else:
                    user_vars[k] = repr(v)[:200]
            except Exception:
                user_vars[k] = f"<{type(v).__name__}>"

        if user_vars:
            parts.append("\nKernel variables:")
            for k, v in sorted(user_vars.items()):
                parts.append(f"  {k} = {v}")

        # Recent execution history (last 5 inputs)
        hist = ip.history_manager.get_tail(5)
        if hist:
            parts.append("\nRecent inputs:")
            for _, _, src in hist:
                src = src.strip()[:300]
                if src:
                    parts.append(f"  > {src}")

    except Exception as e:
        parts.append(f"(Failed to gather kernel context: {e})")

    return "\n".join(parts)
