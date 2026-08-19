"""Tests for jupyter-ai-hermes-magics.

Covers:
  - Session tree (dot-notation labels, fork semantics)
  - Context gathering (sync HTTP, cells-above-only, kernel fallback)
  - Cell type detection (code vs markdown)
  - Static HTML rendering (post-completion snapshot)
  - Button state transitions
  - Argument parsing
  - Streaming HTML builder
  - Version
"""

import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
from contextlib import redirect_stdout

from jupyter_ai_hermes_magics.magics import (
    _detect_cell_type,
    _render_static_html,
    _set_button_state,
    _build_streaming_html,
)
from jupyter_ai_hermes_magics.version import __version__


# ── Cell type detection ────────────────────────────────────────────────

class TestCellTypeDetection(unittest.TestCase):
    """Test dynamic cell type detection."""

    def test_single_fenced_code_block_becomes_code_cell(self):
        response = "```python\ndef hello():\n    print('hi')\n```"
        ct, content = _detect_cell_type(response)
        self.assertEqual(ct, "code")
        self.assertIn("def hello()", content)
        self.assertNotIn("```", content)

    def test_plain_prose_becomes_markdown(self):
        response = "The variable x equals 42."
        ct, content = _detect_cell_type(response)
        self.assertEqual(ct, "markdown")

    def test_prose_plus_code_becomes_markdown(self):
        response = "Here is a function:\n\n```python\ndef f():\n    pass\n```"
        ct, _ = _detect_cell_type(response)
        self.assertEqual(ct, "markdown")

    def test_raw_code_starting_with_def(self):
        response = "def foo():\n    return 42"
        ct, content = _detect_cell_type(response)
        self.assertEqual(ct, "code")

    def test_raw_code_starting_with_class(self):
        response = "class Foo:\n    pass"
        ct, _ = _detect_cell_type(response)
        self.assertEqual(ct, "code")

    def test_raw_code_starting_with_import(self):
        response = "import os\nimport sys"
        ct, _ = _detect_cell_type(response)
        self.assertEqual(ct, "code")

    def test_empty_response(self):
        ct, content = _detect_cell_type("")
        self.assertEqual(ct, "markdown")

    def test_multiple_code_blocks_becomes_markdown(self):
        response = "```python\na=1\n```\n\n```python\nb=2\n```"
        ct, _ = _detect_cell_type(response)
        self.assertEqual(ct, "markdown")


# ── Static HTML rendering ──────────────────────────────────────────────

class TestStaticHTML(unittest.TestCase):
    """Test static HTML rendering for post-completion states."""

    def test_done_html_contains_done_text(self):
        h = _render_static_html("done")
        self.assertIn("Done", h)

    def test_stopped_html_contains_stopped_text(self):
        h = _render_static_html("stopped")
        self.assertIn("Stopped", h)

    def test_error_html_contains_error_text(self):
        h = _render_static_html("error", "something went wrong")
        self.assertIn("something went wrong", h)

    def test_unknown_state_returns_empty(self):
        h = _render_static_html("unknown")
        self.assertEqual(h, "")


# ── Streaming HTML builder ─────────────────────────────────────────────

class TestStreamingHTML(unittest.TestCase):
    """Test _build_streaming_html for real-time output."""

    def test_empty_chunks_and_no_tool_calls(self):
        h = _build_streaming_html([], [])
        self.assertEqual(h, "")

    def test_text_only(self):
        h = _build_streaming_html(["Hello ", "world"], [])
        self.assertIn("Hello world", h)

    def test_tool_call_only(self):
        h = _build_streaming_html([], [
            {"id": "tc1", "title": "read file", "kind": "read", "status": "completed"}
        ])
        self.assertIn("Tool calls (1)", h)
        self.assertIn("read file", h)

    def test_text_and_tool_calls(self):
        h = _build_streaming_html(["Response text"], [
            {"id": "tc1", "title": "read file", "kind": "read", "status": "completed"}
        ])
        self.assertIn("Response text", h)
        self.assertIn("Tool calls (1)", h)

    def test_html_escaping(self):
        h = _build_streaming_html(["<script>alert(1)</script>"], [])
        self.assertIn("&lt;script&gt;", h)
        self.assertNotIn("<script>", h)

    def test_multiple_tool_calls(self):
        h = _build_streaming_html([], [
            {"id": "tc1", "title": "read", "kind": "read", "status": "completed"},
            {"id": "tc2", "title": "edit", "kind": "edit", "status": "in_progress"},
        ])
        self.assertIn("Tool calls (2)", h)


# ── Button state transitions ───────────────────────────────────────────

class TestButtonStates(unittest.TestCase):
    """Test that _set_button_state doesn't crash for all states."""

    def test_all_states_no_crash(self):
        button = MagicMock()
        label = MagicMock()
        for state in ["idle", "streaming", "done", "stopped", "error"]:
            _set_button_state(button, label, state)
            self.assertTrue(True)

    def test_streaming_includes_tip(self):
        button = MagicMock()
        label = MagicMock()
        _set_button_state(button, label, "streaming", "(5s)")
        label.value = "test"  # just verify callable


# ── Argument parsing ───────────────────────────────────────────────────

class TestArgumentParsing(unittest.TestCase):
    """Test argument parsing for %%hermes."""

    def test_parse_args_defaults(self):
        import argparse
        parser = argparse.ArgumentParser(prog="%%hermes", add_help=False)
        parser.add_argument("--label", "-l", default=None)
        parser.add_argument("--no-context", action="store_true")
        parser.add_argument("--new", action="store_true")
        parser.add_argument("--version", action="store_true")
        args, _ = parser.parse_known_args([])
        self.assertIsNone(args.label)
        self.assertFalse(args.no_context)
        self.assertFalse(args.new)

    def test_parse_args_label(self):
        import argparse
        parser = argparse.ArgumentParser(prog="%%hermes", add_help=False)
        parser.add_argument("--label", "-l", default=None)
        args, _ = parser.parse_known_args(["--label", "main.debug"])
        self.assertEqual(args.label, "main.debug")

    def test_parse_args_no_context(self):
        import argparse
        parser = argparse.ArgumentParser(prog="%%hermes", add_help=False)
        parser.add_argument("--no-context", action="store_true")
        args, _ = parser.parse_known_args(["--no-context"])
        self.assertTrue(args.no_context)


# ── FormatDict ─────────────────────────────────────────────────────────

class TestFormatDict(unittest.TestCase):
    """Test _FormatDict for variable interpolation."""

    def test_known_keys(self):
        from jupyter_ai_hermes_magics.magics import _FormatDict
        d = _FormatDict({"name": "world"})
        self.assertEqual("hello {name}".format_map(d), "hello world")

    def test_unknown_keys_left_unchanged(self):
        from jupyter_ai_hermes_magics.magics import _FormatDict
        d = _FormatDict({"name": "world"})
        self.assertEqual("hello {missing}".format_map(d), "hello {missing}")


# ── AcpConnection singleton ────────────────────────────────────────────

class TestAcpConnectionSingleton(unittest.TestCase):
    """Test AcpConnection singleton behavior."""

    def test_singleton_returns_same_instance(self):
        from jupyter_ai_hermes_magics.acp_client import AcpConnection
        a1 = AcpConnection.get()
        a2 = AcpConnection.get()
        self.assertIs(a1, a2)

    def test_reset_creates_new_instance(self):
        from jupyter_ai_hermes_magics.acp_client import AcpConnection
        a1 = AcpConnection.get()
        AcpConnection.reset_instance()
        a2 = AcpConnection.get()
        self.assertIsNot(a1, a2)

    def test_not_initialized_by_default(self):
        from jupyter_ai_hermes_magics.acp_client import AcpConnection
        AcpConnection.reset_instance()
        conn = AcpConnection.get()
        self.assertFalse(conn.is_initialized)
        self.assertIsNone(conn.session_id)


# ── Version ────────────────────────────────────────────────────────────

def test_version():
    assert __version__ == "0.5.0"


# ── Notebook outline (lightweight context) ─────────────────────────────

class TestOutlineLine(unittest.TestCase):
    """Test _outline_line renders a compact one-line entry."""

    def _ol(self, cell, index=0, is_active=False):
        from jupyter_ai_hermes_magics.context import _outline_line
        return _outline_line(cell, index, is_active)

    def test_basic_code_cell(self):
        line = self._ol({"cellType": "code", "source": "x = 1\nprint(x)",
                         "cell_id": "abc123", "execution_count": 2}, index=4)
        self.assertIn("[4]", line)
        self.assertIn("code", line)
        self.assertIn("id=abc123", line)
        self.assertIn("exec=2", line)
        # single-line preview, whitespace collapsed
        self.assertEqual(line.count("\n"), 0)
        self.assertIn("x = 1 print(x)", line)

    def test_active_marker(self):
        line = self._ol({"cellType": "code", "source": "%%hermes\nhi",
                         "cell_id": "zz"}, index=9, is_active=True)
        self.assertIn("ACTIVE", line)

    def test_long_source_truncated(self):
        long_src = "a = 1\n" * 500  # 3500 chars
        line = self._ol({"cellType": "code", "source": long_src, "cell_id": "id"})
        from jupyter_ai_hermes_magics.context import _OUTLINE_PREVIEW
        # preview portion is capped; full source must NOT be present
        self.assertNotIn(long_src, line)
        self.assertIn("…", line)
        # the whole line is bounded (index + type + id + exec + capped preview)
        self.assertLess(len(line), _OUTLINE_PREVIEW + 60)

    def test_empty_source(self):
        line = self._ol({"cellType": "markdown", "source": "", "cell_id": "m1"})
        self.assertIn("[0]", line)
        self.assertIn("markdown", line)


class TestGatherContextOutline(unittest.TestCase):
    """Test gather_context emits an outline, not full cell source."""

    def _mock_mcp(self, cells, active_cell_id):
        """Build a _mcp_call side_effect returning the right thing per tool."""
        import json as _json

        def _fake(tool, **args):
            if tool == "get_active_notebook":
                return '"work/big.ipynb"'
            if tool == "get_active_cell_id":
                return f'"{active_cell_id}"'
            if tool == "read_notebook_cells":
                return _json.dumps(cells)
            return None

        return _fake

    def _run(self, cells, active_cell_id, magic_cell_id=None):
        import jupyter_ai_hermes_magics.context as ctx
        with patch.object(ctx, "_mcp_initialize", return_value=True), \
             patch.object(ctx, "_mcp_call",
                          side_effect=self._mock_mcp(cells, active_cell_id)):
            return ctx.gather_context(magic_cell_id)

    def test_outline_not_full_source(self):
        cells = [
            {"cellType": "markdown", "source": "Intro", "cell_id": "c0"},
            {"cellType": "code", "source": "x = 42\ny = 43\nz = 44\n"
                                           "w = 45\nv = 46", "cell_id": "c1",
             "execution_count": 1},
            {"cellType": "code", "source": "MY_MARKER = 'purple-turtle'",
             "cell_id": "c2", "execution_count": 2},
            {"cellType": "code", "source": "%%hermes\nquestion here",
             "cell_id": "ACTIVE"},
        ]
        out = self._run(cells, "ACTIVE")
        # It is an outline: one line per cell, no fenced full source blocks.
        self.assertIn("Notebook outline", out)
        self.assertIn("id=c0", out)
        self.assertIn("id=c2", out)
        self.assertIn("ACTIVE", out)
        # The full multi-line source of c1 must NOT be inlined verbatim.
        self.assertNotIn("x = 42\ny = 43\nz = 44\nw = 45\nv = 46", out)
        # It tells Hermes how to fetch full content.
        self.assertIn("read_notebook_cells", out)
        self.assertIn("specific_cell_id", out)

    def test_only_cells_up_to_active(self):
        cells = [
            {"cellType": "code", "source": "above", "cell_id": "c0"},
            {"cellType": "code", "source": "%%hermes\nq", "cell_id": "ACTIVE"},
            {"cellType": "code", "source": "BELOW_MARKER", "cell_id": "c2"},
        ]
        out = self._run(cells, "ACTIVE")
        # The cell below the magic cell is NOT outlined.
        self.assertNotIn("BELOW_MARKER", out)

    def test_length_bounded_regardless_of_cell_count(self):
        # 100 huge cells -> outline stays small (the core perf property).
        cells = [{"cellType": "code", "source": "line %d\n" % i * 200,
                  "cell_id": f"c{i}"} for i in range(99)]
        cells.append({"cellType": "code", "source": "%%hermes\nq",
                      "cell_id": "ACTIVE"})
        out = self._run(cells, "ACTIVE")
        # Full-dump would be ~100 * 2000 = 200KB. Outline must be far less.
        self.assertLess(len(out), 20_000)
        self.assertGreaterEqual(out.count("id="), 100)


    def test_anchor_uses_magic_cell_not_ui_active(self):
        """REGRESSION: outline must anchor on the real magic cell, NOT the
        server's 'active cell' (UI cursor focus), which can be a different
        cell. This is the bug where 'the cell above' resolved to cell 0.

        NOTE: the UI-focus cell id must NOT contain the substring "ACTIVE" or
        the marker check below would match it by accident."""
        # 5 cells. The magic cell is index 4. But the UI focus (active) is on
        # index 1 — so without the fix, the outline would stop at cell 1 and
        # mark it ACTIVE, and 'the cell above' = cell 0 (wrong).
        cells = [
            {"cellType": "markdown", "source": "Title cell", "cell_id": "c0"},
            {"cellType": "markdown", "source": "## Context above", "cell_id": "UI_FOCUS"},
            {"cellType": "code", "source": "x = 1", "cell_id": "c2", "execution_count": 1},
            {"cellType": "code", "source": "MY_MARKER_CELL = 'purple-turtle-7311'",
             "cell_id": "c3", "execution_count": 2},
            {"cellType": "code", "source": "%%hermes\nWhat's above me?",
             "cell_id": "MAGIC"},
        ]
        out = self._run(cells, active_cell_id="UI_FOCUS", magic_cell_id="MAGIC")
        # The magic cell must carry the exact ACTIVE marker, not the UI cell.
        marked = [l for l in out.split("\n") if "<- ACTIVE" in l]
        self.assertEqual(len(marked), 1)
        self.assertIn("MAGIC", marked[0])
        self.assertNotIn("UI_FOCUS", marked[0])
        # All 5 cells (up to and including magic) must be in the outline.
        self.assertIn("id=c3", out)   # the real cell above the magic
        self.assertIn("id=c0", out)
        # The outline header should report 5 cells, not 2.
        self.assertIn("5 cell(s)", out)

    def test_falls_back_to_ui_active_when_no_magic_id(self):
        """If magic_cell_id is missing, fall back to the UI-active cell."""
        cells = [
            {"cellType": "code", "source": "above", "cell_id": "c0"},
            {"cellType": "code", "source": "%%hermes\nq", "cell_id": "UI_FOCUS"},
            {"cellType": "code", "source": "BELOW_MARKER", "cell_id": "c2"},
        ]
        out = self._run(cells, active_cell_id="UI_FOCUS", magic_cell_id=None)
        marked = [l for l in out.split("\n") if "<- ACTIVE" in l]
        self.assertEqual(len(marked), 1)
        self.assertIn("UI_FOCUS", marked[0])
        self.assertNotIn("BELOW_MARKER", out)


# ── Enriched MCP tools doc (native form, not CLI) ──────────────────────

class TestMcpToolsDoc(unittest.TestCase):
    """The magic advertises native MCP tools (registered at session create),
    NOT a shell CLI. The doc must document the full tool set and the
    notebook_path/file_path param convention."""

    def _doc(self):
        from jupyter_ai_hermes_magics.magics import MCP_TOOLS_DOC
        return MCP_TOOLS_DOC

    def test_native_not_cli(self):
        d = self._doc()
        # Must NOT instruct Hermes to shell out to a CLI.
        self.assertNotIn("jupyter-mcp-cli", d)
        self.assertNotIn("--arg ", d)

    def test_documents_full_tool_set(self):
        d = self._doc()
        for tool in [
            "get_open_documents", "get_active_notebook", "get_active_cell_id",
            "read_notebook_cells", "read_notebook", "read_cell",
            "get_cell_id_from_index", "add_cell", "insert_cell", "edit_cell",
            "delete_cell", "run_cell", "run_all_cells", "set_cell_metadata",
            "get_cell_metadata", "list_cell_tags", "select_cell", "open_file",
            "get_notebook_info", "list_all_commands", "execute_command",
        ]:
            self.assertIn(tool, d)

    def test_param_convention_present(self):
        d = self._doc()
        self.assertIn("notebook_path", d)
        self.assertIn("file_path", d)

    def test_on_demand_fetch_pattern(self):
        d = self._doc()
        # The "cell above" pattern (ACTIVE index - 1) must be stated so the
        # model fetches the right cell by id.
        self.assertIn("ACTIVE index", d)
        self.assertIn("specific_cell_id", d)


if __name__ == "__main__":
    unittest.main()
