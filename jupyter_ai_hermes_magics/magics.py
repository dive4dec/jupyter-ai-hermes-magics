"""%%hermes cell magic — talk to Hermes Agent inside a Jupyter notebook.

Uses a persistent ACP subprocess (like jupyter-ai) for fast (~3s) responses
with real-time streaming, tool-call display, and permission handling.

Two-phase button design:
    Phase 1 (Shift+Enter): Cell output shows a ▶ button + tip.
        No request is sent yet.
    Phase 2 (Click ▶): Sends the magic cell content to Hermes via ACP.
        Button toggles to ⏹.  Click ⏹ to stop.
        Streaming text + tool-call cards appear in the output area.
        On completion: response goes to a new transcript cell below.
        On stop: no transcript cell is created.

Transcript cell type is dynamic:
    - Single fenced code block → code cell (fences stripped)
    - Raw code (def/class/import) → code cell
    - Otherwise → markdown cell

Session tree with dot-notation labels for branching.

Subcommands:
    %hermes reset   — clear all sessions
    %hermes list    — show session tree
    %hermes version — show version
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import threading
import time
from typing import Optional

from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.display import HTML, display

from .session import SessionTree, SessionNode
from .context import gather_context, _mcp_call, _mcp_initialize
from .acp_client import AcpConnection
from .version import __version__

logger = logging.getLogger(__name__)

# ── Module-level state ─────────────────────────────────────────────────
_lock = threading.Lock()


# ── Cell type detection ────────────────────────────────────────────────

def _detect_cell_type(response: str) -> tuple[str, str]:
    """Detect whether the response should be a code or markdown cell.

    Returns (cell_type, content):
        - ("code", stripped_code)  — when response is a single code block
        - ("markdown", response)   — otherwise (prose, mixed, etc.)
    """
    stripped = response.strip()

    # Case 1: Single fenced code block with no surrounding prose.
    fence_pattern = re.compile(r"^```(\w*)\n(.*?)```$", re.DOTALL)
    m = fence_pattern.match(stripped)
    if m:
        inner = m.group(2)
        if "```" not in inner:
            return "code", inner.strip()

    # Case 2: Raw code (starts with common keywords, no prose sentences)
    first_line = stripped.split("\n")[0].strip() if stripped else ""
    code_starters = ("def ", "class ", "import ", "from ", "#!/", "async def ")
    if first_line.startswith(code_starters) and not any(
        line.strip().endswith(".") and not line.strip().startswith(("#", "//"))
        for line in stripped.split("\n")[:3]
    ):
        return "code", stripped

    return "markdown", stripped


# ── Widget creation ────────────────────────────────────────────────────

def _create_button_widget():
    """Create an ipywidgets Button + HTML output area in a VBox.

    The output area is where streaming text and tool-call cards appear.
    """
    import ipywidgets as widgets

    button = widgets.Button(
        description="",
        icon="play",
        button_style="success",
        tooltip="Click to send to Hermes",
        layout=widgets.Layout(width="44px", height="32px"),
    )
    status_label = widgets.HTML(
        value='<span style="color:#888;font-size:0.85em;margin-left:8px;">'
              'Click ▶ to send to Hermes</span>'
    )
    # Streaming output area — updated in real-time from bg thread
    output_html = widgets.HTML(
        value="",
        layout=widgets.Layout(
            margin="4px 0 0 0",
            max_height="400px",
            overflow_y="auto",
        ),
    )
    # Permission button container (hidden until needed)
    perm_container = widgets.HBox([], layout=widgets.Layout(
        display="none", margin="4px 0 0 0"
    ))

    top_row = widgets.HBox([button, status_label])
    vbox = widgets.VBox([top_row, output_html, perm_container])
    return vbox, button, status_label, output_html, perm_container


def _set_button_state(button, label, state: str, tip: str = ""):
    """Update button icon, style, and label text for a given state."""
    if state == "idle":
        button.icon = "play"
        button.button_style = "success"
        button.tooltip = "Click to send to Hermes"
        label.value = (
            '<span style="color:#888;font-size:0.85em;margin-left:8px;">'
            'Click ▶ to send to Hermes</span>'
        )
    elif state == "streaming":
        button.icon = "stop"
        button.button_style = "danger"
        button.tooltip = "Click to stop Hermes"
        label.value = (
            '<span style="color:#c33;font-size:0.85em;margin-left:8px;">'
            f'⏳ Hermes is thinking… {html.escape(tip)}</span>'
        )
    elif state == "done":
        button.icon = "play"
        button.button_style = "success"
        button.tooltip = "Click to send again"
        label.value = (
            '<span style="color:#070;font-size:0.85em;margin-left:8px;">'
            '✓ Done — response written below ↓</span>'
        )
    elif state == "stopped":
        button.icon = "play"
        button.button_style = "success"
        button.tooltip = "Click to send to Hermes"
        label.value = (
            '<span style="color:#888;font-size:0.85em;margin-left:8px;">'
            '⏹ Stopped — no transcript cell created.</span>'
        )
    elif state == "error":
        button.icon = "play"
        button.button_style = "warning"
        button.tooltip = "Click to retry"
        label.value = (
            '<span style="color:#c00;font-size:0.85em;margin-left:8px;">'
            f'✗ {html.escape(tip)}</span>'
        )


# ── Streaming HTML builder ─────────────────────────────────────────────

def _build_streaming_html(
    text_chunks: list[str],
    tool_calls: list[dict],
    show_details: bool = True,
) -> str:
    """Build HTML for the streaming output area.

    Shows:
    - Accumulated response text (scrollable)
    - Tool-call cards in collapsed <details> blocks
    """
    parts = []

    # Tool-call details (collapsed)
    if tool_calls:
        items = []
        for tc in tool_calls:
            icon = {
                "read": "📖", "edit": "✏️", "delete": "🗑️",
                "execute": "▶", "search": "🔍", "think": "💭",
                "fetch": "🌐", "other": "🔧",
            }.get(tc.get("kind", "other"), "🔧")
            status_icon = {
                "pending": "⏳", "in_progress": "⚙️",
                "completed": "✓", "failed": "✗",
            }.get(tc.get("status", ""), "⚙️")
            title = html.escape(tc.get("title", "tool"))
            items.append(
                f'<div style="margin:2px 0;font-size:0.85em;color:#555;">'
                f'{icon} {status_icon} {title}</div>'
            )
        details_html = (
            '<details style="margin-top:4px;">'
            '<summary style="cursor:pointer;color:#666;font-size:0.8em;">'
            f'Tool calls ({len(tool_calls)})</summary>'
            '<div style="margin:4px 0;padding-left:12px;">'
            + "\n".join(items) +
            '</div></details>'
        )
        parts.append(details_html)

    # Response text
    full_text = "".join(text_chunks)
    if full_text:
        escaped = html.escape(full_text)
        parts.append(
            f'<div style="margin-top:4px;padding:4px 8px;'
            f'background:#f8f8f8;border-radius:4px;'
            f'white-space:pre-wrap;font-size:0.9em;">{escaped}</div>'
        )

    return "\n".join(parts)


def _render_static_html(state: str, tip: str = "") -> str:
    """Render a static (non-interactive) HTML snapshot of the button state."""
    if state == "done":
        return (
            '<span style="font-size:1.4em;">▶</span>'
            '<span style="color:#070;font-size:0.85em;margin-left:8px;">'
            '✓ Done — response written below ↓</span>'
        )
    elif state == "stopped":
        return (
            '<span style="font-size:1.4em;color:#888;">▶</span>'
            '<span style="color:#888;font-size:0.85em;margin-left:8px;">'
            '⏹ Stopped — no transcript cell created.</span>'
        )
    elif state == "error":
        return (
            '<span style="font-size:1.4em;color:#888;">▶</span>'
            '<span style="color:#c00;font-size:0.85em;margin-left:8px;">'
            f'✗ {html.escape(tip)}</span>'
        )
    return ""


# ── Permission UI ──────────────────────────────────────────────────────

def _show_permission_buttons(
    perm_container,
    tool_call_id: str,
    tool_name: str,
    options: list[dict],
    on_resolve,
):
    """Show Accept/Reject buttons for a tool-call permission request."""
    import ipywidgets as widgets

    buttons = []
    for opt in options:
        btn = widgets.Button(
            description=opt["name"],
            button_style="primary" if opt["id"] == "allow" else "warning",
            layout=widgets.Layout(width="auto", height="28px"),
        )
        # Capture option_id in closure
        def _make_handler(oid, tcid):
            def _handler(b):
                on_resolve(tcid, oid)
                perm_container.children = []
                perm_container.layout.display = "none"
            return _handler
        btn.on_click(_make_handler(opt["id"], tool_call_id))
        buttons.append(btn)

    label = widgets.HTML(
        value=f'<span style="font-size:0.85em;color:#c80;">'
              f'🔐 Hermes wants to use: {html.escape(tool_name)} — '
              f'Allow?</span>'
    )
    perm_container.children = [label] + buttons
    perm_container.layout.display = "flex"


# ── Start / Stop Hermes ────────────────────────────────────────────────

def _start_hermes(
    state: dict,
    button,
    label,
    output_html,
    perm_container,
    display_handle,
) -> None:
    """Start the Hermes ACP prompt for this cell."""
    prompt_text = state.get("prompt", "")
    exec_count = state.get("exec_count", 0)
    magic_cell_id = state.get("magic_cell_id")
    is_new = state.get("is_new", False)
    session_label = state.get("label", "main")
    tree = state.get("tree")
    node = state.get("node")

    # Get ACP connection
    acp = AcpConnection.get()

    # Initialize if needed (should already be done by %load_ext, but handle
    # edge cases: reload, reset, race condition, etc.)
    if not acp.is_initialized:
        ok, msg = acp.initialize()
        if not ok:
            _set_button_state(button, label, "error", f"ACP init failed: {msg}")
            return

    # ── Resolve session ──
    # Create a new ACP session if:
    #   - is_new flag is set (--new), or
    #   - the label has no ACP session yet (first use), or
    #   - the label's stored session_id is stale (kernel restart)
    # Otherwise resume the existing session.
    acp_sid = acp.get_session_id(session_label) if acp.has_session(session_label) else None

    if is_new or acp_sid is None:
        try:
            acp_sid = acp.new_session(session_label)
            # Persist in notebook metadata
            if tree is not None and node is not None:
                tree.update_session_id(session_label, acp_sid)
            logger.info("Created new ACP session %s for label %r", acp_sid, session_label)
        except Exception as e:
            _set_button_state(button, label, "error", f"Session creation failed: {e}")
            return

    # Set up streaming state
    text_chunks: list[str] = []
    tool_calls: list[dict] = []
    start_time = time.time()
    stop_timer = threading.Event()

    def _timer():
        while not stop_timer.is_set():
            elapsed = time.time() - start_time
            _set_button_state(button, label, "streaming", f"({elapsed:.0f}s)")
            time.sleep(1.0)

    timer_thread = threading.Thread(target=_timer, daemon=True)
    timer_thread.start()

    # ── Callbacks (called from background asyncio thread) ──

    def _on_chunk(text: str):
        text_chunks.append(text)
        # Update widget from bg thread — ipywidgets is thread-safe
        output_html.value = _build_streaming_html(text_chunks, tool_calls)

    def _on_tool_call(info: dict):
        if info["type"] == "start":
            tool_calls.append({
                "id": info["id"],
                "title": info.get("title", "tool"),
                "kind": info.get("kind", "other"),
                "status": info.get("status", "pending"),
            })
        elif info["type"] == "progress":
            for tc in tool_calls:
                if tc["id"] == info["id"]:
                    tc["status"] = info.get("status", "in_progress")
        output_html.value = _build_streaming_html(text_chunks, tool_calls)

    def _on_permission(info: dict):
        # Show accept/reject buttons (called from bg thread)
        # ipywidgets updates are thread-safe
        _show_permission_buttons(
            perm_container,
            info["tool_call_id"],
            info.get("tool_name", "tool"),
            info["options"],
            on_resolve=acp.resolve_permission,
        )

    # ── Run prompt in background thread ──

    def _worker():
        try:
            response_text, stop_reason = acp.prompt(
                prompt_text,
                on_chunk=_on_chunk,
                on_tool_call=_on_tool_call,
                on_permission=_on_permission,
                session_id=acp_sid,
            )
            stop_timer.set()
            timer_thread.join(timeout=2)

            if stop_reason == "cancelled":
                _finalize(display_handle, "stopped")
                return

            if not response_text:
                _finalize(display_handle, "error", "Empty response from Hermes")
                return

            # Detect cell type and write transcript
            cell_type, content = _detect_cell_type(response_text)

            # Write transcript cell via MCP
            transcript_ok = _write_transcript_via_mcp(
                exec_count, content, cell_type, magic_cell_id
            )

            if transcript_ok:
                _finalize(display_handle, "done")
            else:
                # Fallback: show response in cell output
                static = _render_static_html("done")
                escaped = html.escape(response_text)
                display_handle.update(HTML(
                    static + f'<pre style="margin-top:8px;white-space:pre-wrap;">{escaped}</pre>'
                ))

        except Exception as e:
            stop_timer.set()
            timer_thread.join(timeout=2)
            logger.error("Hermes prompt failed: %s", e, exc_info=True)
            _finalize(display_handle, "error", str(e))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _stop_hermes(button, label) -> None:
    """Cancel the current Hermes prompt."""
    acp = AcpConnection.get()
    acp.cancel()


def _finalize(display_handle, state: str, tip: str = ""):
    """Replace the widget with static HTML to avoid widget state saving."""
    static_html = _render_static_html(state, tip)
    if static_html:
        display_handle.update(HTML(static_html))


# ── Transcript cell writing ────────────────────────────────────────────

def _write_transcript_via_mcp(
    exec_count: int,
    content: str,
    cell_type: str,
    magic_cell_id: str | None = None,
) -> bool:
    """Create a new transcript cell below the magic cell via MCP.

    Always creates a NEW cell (never edits).
    """
    # Fresh MCP init (session may have expired)
    import jupyter_ai_hermes_magics.context as ctx_mod
    ctx_mod._mcp_session_id = None

    if not _mcp_initialize():
        logger.debug("MCP not available for transcript writing")
        return False

    try:
        active_nb = _mcp_call("get_active_notebook")
        if not active_nb:
            return False
        active_nb = active_nb.strip().strip('"')

        # ── Resolve the magic cell ID ──
        # Strategy:
        # 1. Use magic_cell_id from parent metadata (most reliable — set
        #    at magic execution time from ip.get_parent()["metadata"]["cellId"])
        # 2. Fall back to execution_count matching only if cell_id is missing
        resolved_cell_id = magic_cell_id

        if not resolved_cell_id:
            cells_raw = _mcp_call("read_notebook_cells", notebook_path=active_nb)
            if not cells_raw:
                return False
            cells = json.loads(cells_raw)

            # Find the magic cell by execution_count (fallback)
            for cell in cells:
                if cell.get("execution_count") == exec_count:
                    resolved_cell_id = cell.get("cell_id")
                    break
            if not resolved_cell_id and exec_count > 1:
                for cell in cells:
                    if cell.get("execution_count") == exec_count - 1:
                        resolved_cell_id = cell.get("cell_id")
                        break
            if not resolved_cell_id:
                for cell in reversed(cells):
                    if cell.get("execution_count") is not None:
                        resolved_cell_id = cell.get("cell_id")
                        break

        if not resolved_cell_id:
            logger.debug("Could not find magic cell (cell_id=%s, exec_count=%d)",
                        magic_cell_id, exec_count)
            return False

        # Snapshot cell IDs BEFORE insertion so we can diff afterwards.
        # read_notebook_cells reads from the FILE, but add_cell writes to
        # the YDoc (in-memory).  File sync is async (~0.5s), so we must
        # poll to discover the new cell's ID rather than reading immediately.
        ids_before: set[str] = set()
        cells_raw_pre = _mcp_call("read_notebook_cells", notebook_path=active_nb)
        if cells_raw_pre:
            try:
                for c in json.loads(cells_raw_pre):
                    cid = c.get("cell_id")
                    if cid:
                        ids_before.add(cid)
            except Exception:
                pass

        _mcp_call(
            "add_cell",
            file_path=active_nb,
            cell_id=resolved_cell_id,
            cell_type=cell_type,
            content=content,
        )

        # ── Render the new cell (exit edit mode) ──
        # add_cell inserts via YDoc but leaves the cell in edit mode
        # (raw source).  Execute the new cell to render it (markdown →
        # rendered, code → executed).  We must first wait for the YDoc→file
        # sync to discover the new cell's ID, then call run_cell on it.
        try:
            import time as _time
            new_cell_id = None
            for _ in range(10):  # poll up to ~5s
                _time.sleep(0.5)
                cells_raw2 = _mcp_call("read_notebook_cells", notebook_path=active_nb)
                if not cells_raw2:
                    continue
                cells2 = json.loads(cells_raw2)
                for c in cells2:
                    cid2 = c.get("cell_id")
                    if cid2 and cid2 not in ids_before:
                        new_cell_id = cid2
                        break
                if new_cell_id:
                    break

            if new_cell_id:
                _mcp_call("run_cell", cell_id=new_cell_id)
        except Exception:
            # Non-fatal — cell is inserted, just not rendered
            pass

        return True

    except Exception as e:
        logger.debug("MCP transcript write failed: %s", e)
        return False


# ── Prompt construction ────────────────────────────────────────────────

# NOTE: these are invoked as NATIVE MCP tools on the Jupyter server
# (registered at ACP session creation), NOT as a shell CLI. Tools are
# addressed by name only (no `jupyter-mcp-cli` wrapper, no --arg syntax).
#
# Parameter convention (this is the #1 source of silent failures — state it
# explicitly):
#   • notebook_path : get_active_cell_id, read_notebook_cells
#   • file_path     : read_notebook, read_cell, get_cell_id_from_index,
#                     add_cell, insert_cell, edit_cell, delete_cell,
#                     set_cell_metadata, get_cell_metadata, list_cell_tags,
#                     get_notebook_info
#   • cell_id       : run_cell (operates on the ACTIVE notebook)
MCP_TOOLS_DOC = """## Jupyter MCP Tools

You have Jupyter MCP tools (served at localhost:3001). **Always use these
instead of raw `nbformat`/file writes** — they apply collaboratively via YDoc
(preserve cell tags/metadata, update the JupyterLab UI instantly).

Notebook cells were given to you as a **one-line outline** (index, type,
cell `id`, short preview) — NOT full source. **Fetch full content on demand**
before quoting or editing anything beyond the preview.

### Reading context
| Tool | Args | Purpose |
|------|------|---------|
| `get_open_documents` | — | List all open documents |
| `get_active_notebook` | — | Active notebook path |
| `get_active_cell_id` | `notebook_path` | Currently focused cell ID |
| `read_notebook_cells` | `notebook_path`, optional `specific_cell_id` | All cells (JSON), or one cell |
| `read_notebook` | `file_path` | Whole notebook as markdown |
| `read_cell` | `file_path`, `cell_id` | One cell as markdown |
| `get_cell_id_from_index` | `file_path`, `cell_index` | Resolve index → cell ID |

### Editing cells
| Tool | Args | Purpose |
|------|------|---------|
| `add_cell` | `file_path`, `cell_id`, `cell_type`, `content`, optional `add_above` | New cell above/below target |
| `insert_cell` | `file_path`, `insert_index`, `cell_type`, `content` | Insert at index |
| `edit_cell` | `file_path`, `cell_id`, `content` | Modify a cell's content |
| `delete_cell` | `file_path`, `cell_id` | Delete a cell |

### Running cells
| Tool | Args | Purpose |
|------|------|---------|
| `run_cell` | `cell_id` | Execute one cell (active notebook) |
| `run_all_cells` | — | Execute all cells |

### Metadata, tags & navigation
| Tool | Args | Purpose |
|------|------|---------|
| `set_cell_metadata` | `file_path`, `cell_id`, `metadata` (JSON object) | Set metadata, e.g. `{"slideshow":{"slide_type":"slide"}}` |
| `get_cell_metadata` | `file_path`, `cell_id` | View metadata |
| `list_cell_tags` | `file_path` | All tagged cells |
| `select_cell` | `cell_id` | Move UI focus to a cell |
| `open_file` | `file_path` | Open a file in JupyterLab |
| `get_notebook_info` | `file_path` | Notebook format (jupytext) — check before editing |

### JupyterLab commands (escape hatch)
| Tool | Args | Purpose |
|------|------|---------|
| `list_all_commands` | optional `query` | Discover available Lab commands |
| `execute_command` | `command_id`, optional `args` | Run a Lab command |

### Cell-content conventions
- Use **real newlines** in `content`, never literal `\\n`.
- One logical idea per cell; `#`/`##`/`###` headers go in their own markdown cells.
- Keep cells concise — prefer several small cells over one dense wall of text.
- For jupytext `.py`/`.md` formats, check `get_notebook_info` first.

**The correct pattern for "explain/edit the cell above":** it is the outline
row with index = (ACTIVE index − 1). Call `read_notebook_cells` with
`notebook_path="<active notebook path>"` and `specific_cell_id="<that row's
id=>"` to fetch it, THEN act.
"""


# ── Magics class ───────────────────────────────────────────────────────

@magics_class
class HermesMagics(Magics):
    """%%hermes cell magic for talking to Hermes Agent inside notebooks."""

    def __init__(self, shell):
        super().__init__(shell)
        self._session_tree: Optional[SessionTree] = None

    def _get_session_tree(self) -> SessionTree:
        if self._session_tree is not None:
            return self._session_tree

        nb_metadata = {}
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip and hasattr(ip, "get_parent"):
                parent = ip.get_parent()
                if parent and "metadata" in parent:
                    nb_metadata = parent["metadata"].get("notebook", {}).get("metadata", {})
        except Exception:
            pass

        self._session_tree = SessionTree(notebook_metadata=nb_metadata)
        return self._session_tree

    def _parse_cell_args(self, line: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="%%hermes", add_help=False)
        parser.add_argument("--label", "-l", default=None,
                            help="Session label (dot-notation for tree)")
        parser.add_argument("--no-context", action="store_true",
                            help="Skip notebook context injection")
        parser.add_argument("--new", action="store_true",
                            help="Start a fresh session even if label exists")
        parser.add_argument("--version", action="store_true",
                            help="Show version")
        args, _ = parser.parse_known_args(line.split())
        return args

    @line_cell_magic
    def hermes(self, line: str, cell: Optional[str] = None) -> None:
        """%%hermes cell magic — talk to Hermes Agent inside a notebook."""
        # ── Line magic subcommands ──
        if cell is None:
            parts = line.strip().split()
            if not parts:
                print("Usage: %%hermes [options] <prompt>  or  %hermes <subcommand>")
                return
            cmd = parts[0]
            if cmd == "reset":
                self._reset_sessions()
                return
            if cmd == "list":
                self._list_sessions()
                return
            if cmd == "version":
                print(f"jupyter-ai-hermes-magics v{__version__}")
                return
            if cmd == "help":
                self._print_help()
                return
            print(f"Unknown subcommand: {cmd}. Use: %hermes help, list, reset, version")
            return

        # ── Cell magic ──
        args = self._parse_cell_args(line)
        if args.version:
            print(f"jupyter-ai-hermes-magics v{__version__}")
            return

        prompt = cell.strip()
        if not prompt:
            print("Error: empty prompt. Usage: %%hermes\\n<prompt>")
            return

        # Interpolate variables from kernel namespace
        ip = self.shell
        try:
            prompt = prompt.format_map(_FormatDict(ip.user_ns))
        except Exception:
            pass

        # Resolve session
        tree = self._get_session_tree()
        label = args.label or tree.latest_label() or tree.root_label
        node = tree.get_or_create(label)
        is_new = args.new or node.session_id is None

        # Capture the magic cell's ID directly from the kernel parent
        # metadata.  JupyterLab sends cellId in the execute_request metadata;
        # ipykernel stores it at parent["metadata"]["cellId"].  This is the
        # reliable anchor for the context outline (the server's "active cell"
        # = cursor focus may point at a different cell).
        magic_cell_id = None
        try:
            parent = ip.get_parent()
            if parent and "metadata" in parent:
                magic_cell_id = parent["metadata"].get("cellId")
        except Exception:
            pass

        # Gather notebook context (anchored on the real magic cell)
        context_text = ""
        if not args.no_context:
            try:
                context_text = gather_context(magic_cell_id)
            except Exception as e:
                logger.debug("Context gathering failed: %s", e)
                context_text = ""

        # Build full prompt
        full_prompt_parts = [f"**User question:**\n{prompt}"]

        if context_text and context_text != "(No IPython kernel available)":
            full_prompt_parts.append(
                "## Notebook Context\n\n"
                "A one-line outline of the cells at/above the user's `%%hermes` "
                "cell (marked ACTIVE) is below. Full source is NOT shown — "
                "fetch any cell you need with the `read_notebook_cells` MCP "
                "tool by its `id`. \"The code above\" = cells before the "
                "ACTIVE cell.\n\n"
                + context_text
            )

        full_prompt_parts.append(MCP_TOOLS_DOC)
        full_prompt_parts.append(
            "\n---\n\nAnswer the user's question about the notebook. "
            "Focus on what they asked. Do NOT explain the tool documentation."
        )

        full_prompt = "\n\n".join(full_prompt_parts)

        # Fork: prepend parent history
        if is_new and node.parent_session_id:
            history = SessionTree.get_session_history(node.parent_session_id)
            if history:
                history_text = "\n".join(
                    f"[{m['role']}]: {m['content'][:500]}" for m in history[-10:]
                )
                full_prompt = (
                    f"## Previous conversation context (from parent session)\n{history_text}\n\n"
                    + full_prompt
                )

        # ── Phase 1: Display the button ──
        exec_count = ip.execution_count

        # (magic_cell_id was captured above, before context gathering, and is
        #  reused here for the transcript cell.)
        hermes_state = {
            "prompt": full_prompt,
            "session_id": node.session_id if not is_new else None,
            "label": label,
            "shell": self.shell,
            "tree": tree,
            "node": node,
            "is_new": is_new,
            "exec_count": exec_count,
            "magic_cell_id": magic_cell_id,
        }

        # Create widget
        vbox, button, label_widget, output_html, perm_container = \
            _create_button_widget()

        # Display with display_id so we can replace widget with HTML later
        display_handle = display(vbox, display_id=True)

        # Wire up click handler
        def _on_click(b):
            with _lock:
                acp = AcpConnection.get()
                is_streaming = acp._prompt_in_progress

            if is_streaming:
                _stop_hermes(button, label_widget)
            else:
                _start_hermes(
                    hermes_state, button, label_widget,
                    output_html, perm_container, display_handle
                )

        button.on_click(_on_click)

        # Print session info
        acp = AcpConnection.get()
        existing_sid = acp.get_session_id(label) if acp.has_session(label) else None
        if existing_sid and not is_new:
            sid_short = existing_sid[:12] + "…" if len(existing_sid) > 12 else existing_sid
            print(f"↻ Session: {label} (ACP: {sid_short})")
        elif is_new:
            print(f"🌿 Fork/new: {label} (will create on click)")
        else:
            print(f"🔴 New session: {label} (ACP will start on first use)")


    def _reset_sessions(self) -> None:
        tree = self._get_session_tree()
        for label_name in tree.list_labels():
            node = tree.get(label_name)
            if node:
                node.session_id = None
        if SessionTree.METADATA_KEY in tree._metadata:
            tree._metadata[SessionTree.METADATA_KEY]["sessions"] = {}
        self._session_tree = None

        # Also reset ACP connection
        AcpConnection.reset_instance()
        print("✅ All sessions cleared. Next %%hermes will start fresh.")

    def _print_help(self) -> None:
        """Print comprehensive help for %hermes and %%hermes."""
        from textwrap import dedent

        tree = self._get_session_tree()
        labels = tree.list_labels()

        # Build session tree display
        if labels:
            tree_lines = []
            for label_name in labels:
                node = tree.get(label_name)
                if node is None:
                    continue
                depth = 0 if label_name == tree.root_label else label_name.count(".")
                indent = "    " * depth
                marker = "📂" if label_name == tree.root_label else "↳"
                sid = node.session_id[:12] + "…" if node.session_id else "—"
                fork = f" (fork of {node.parent_label})" if node.parent_session_id else ""
                tree_lines.append(f"  {indent}{marker} {label_name}: {sid}{fork}")
            tree_display = "\n".join(tree_lines)
        else:
            tree_display = "  (no sessions yet — run %%hermes to create one)"

        acp = AcpConnection.get()
        acp_sid = "—"
        if acp.is_initialized and acp.session_id:
            acp_sid = acp.session_id[:12] + "…"

        help_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║  jupyter-ai-hermes-magics v{__version__}                          ║
╚══════════════════════════════════════════════════════════════════╝

─── CELL MAGIC: %%hermes ───────────────────────────────────────────

Talk to Hermes Agent directly inside a notebook cell.  Hermes can read
your notebook, add/run cells, and answer questions with full context.

USAGE:
    %%hermes [options]
    <your prompt here>

FLAGS:
    --label NAME, -l NAME   Session label (dot-notation for tree).
                            Default: "main" (or the most recently used).
    --new                   Force a fresh session even if the label
                            already exists.  Parent history is prepended
                            (fork semantics).
    --no-context            Skip injecting notebook cells into the
                            prompt.  Hermes can still read them via MCP.
    --version               Print version and exit.

EXAMPLES:
    %%hermes
    Explain what the code above does.

    %%hermes --label main.explain
    Why is this function returning None?

    %%hermes --label main.fix --new
    Rewrite the function to return a DataFrame instead.

    %%hermes --no-context
    What is the difference between list and tuple in Python?

VARIABLE INTERPOLATION:
    {{variable}} in the prompt is replaced with the kernel namespace
    value, e.g.  "Summarise {{df}}" injects the DataFrame's repr.

─── LINE MAGIC: %hermes ─────────────────────────────────────────────

SUBCOMMANDS:
    %hermes help        Show this help.
    %hermes list        Show the session tree and ACP status.
    %hermes reset       Clear all sessions and restart ACP.
    %hermes version     Print the installed version.

─── SESSION MODEL ───────────────────────────────────────────────────

Each notebook kernel spawns one persistent `hermes acp` subprocess.
The first %%hermes call creates session "main" (root).  All subsequent
calls without --label auto-resume "main" — conversation accumulates.

Labels create a FORK TREE:
    main              — root session (fresh)
    main.explain      — child: inherits main's history at creation
    main.explain.fix  — grandchild of main.explain
    main.test         — sibling of main.explain (forks from main)

Each fork prepends the parent's last 10 messages as context.  Labels
and session IDs are stored in notebook metadata and survive kernel
restarts.  Use --new to start a fresh session on an existing label.

After a kernel restart, stored session IDs become stale (the ACP
subprocess was killed).  The next %%hermes on that label automatically
creates a fresh session — no manual intervention needed.

─── CONTEXT INJECTION ───────────────────────────────────────────────

By default, notebook cells UP TO AND INCLUDING the %%hermes cell are
injected as a **one-line outline** (index, type, cell `id`, short preview)
— NOT the full source. This keeps the prompt small even for large
notebooks. Hermes fetches any cell's full content on demand via the
`read_notebook_cells` MCP tool (by `cell_id`). Cells below the magic cell
are not outlined at all.

─── CURRENT STATUS ──────────────────────────────────────────────────

ACP connection: {acp_sid}
Session tree:
{tree_display}
"""
        print(dedent(help_text).strip())

    def _list_sessions(self) -> None:
        tree = self._get_session_tree()
        labels = tree.list_labels()
        if not labels:
            print("No sessions yet. Run %%hermes to create one.")
            return

        acp = AcpConnection.get()
        acp_sessions = acp.sessions if acp.is_initialized else {}
        active_sid = acp.active_session_id
        active_short = active_sid[:12] + "…" if active_sid else None

        print(f"ACP active: {active_short if active_sid else '—'}")
        print(f"ACP sessions: {len(acp_sessions)}")
        print("Session tree:")
        for label_name in labels:
            node = tree.get(label_name)
            if node is None:
                continue
            indent = "  " * (label_name.count(".") if label_name != tree.root_label else 0)
            marker = "📂" if label_name == tree.root_label else "↳"
            # Show the ACP session ID (live) if available, else the stored one
            live_sid = acp_sessions.get(label_name)
            sid = (live_sid or node.session_id or None)
            sid_short = sid[:12] + "…" if sid else "—"
            stale = " ⚠️stale" if node.session_id and not live_sid else ""
            fork = f" (fork of {node.parent_label})" if node.parent_session_id else ""
            active_mark = " ← active" if (active_sid and live_sid == active_sid) else ""
            print(f"  {indent}{marker} {label_name}: {sid_short}{fork}{stale}{active_mark}")


class _FormatDict(dict):
    """Dict that leaves unknown {keys} unchanged when used with str.format_map."""

    def __missing__(self, key):
        return key.join("{}")


def _init_acp_background():
    """Initialize ACP connection in a background thread.

    Called at %load_ext time so the first %%hermes call is fast.
    Also serves as a health check — if hermes acp is unavailable,
    the error is reported immediately at load time, not on first use.
    """
    import sys

    acp = AcpConnection.get()
    if acp.is_initialized:
        sid = acp.session_id or ""
        logger.debug("Hermes ACP already connected (session %s…)", sid[:12])
        return

    # Check hermes binary exists before starting
    import shutil
    hermes_bin = os.environ.get("HERMES_BIN_PATH") or shutil.which("hermes")
    if not hermes_bin:
        print(
            "⚠ Hermes ACP: 'hermes' binary not found. %%hermes magic will not work.\n"
            "  Install hermes-agent or set HERMES_BIN_PATH.",
            file=sys.stderr,
        )
        return

    logger.debug("Hermes ACP: starting background connection…")

    def _worker():
        try:
            ok, msg = acp.initialize()
            if ok:
                sid = acp.session_id or ""
                logger.debug("Hermes ACP connected (session %s…)", sid[:12])
            else:
                print(
                    f"✗ Hermes ACP init failed: {msg}\n"
                    "  %%hermes magic will not work until the issue is resolved.",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"✗ Hermes ACP init error: {e}", file=sys.stderr)

    t = threading.Thread(target=_worker, daemon=True, name="hermes-acp-init")
    t.start()


def load_ipython_extension(ipython):
    """Register the %%hermes magic and start ACP connection.

    The ACP connection (hermes acp subprocess) is started in a background
    thread so the extension loads instantly. By the time the user runs
    their first %%hermes cell, the connection is usually ready. This also
    serves as a health check — if hermes is not installed, the error is
    reported immediately.
    """
    ipython.register_magics(HermesMagics)
    _init_acp_background()
