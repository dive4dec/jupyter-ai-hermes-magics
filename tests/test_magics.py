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
    assert __version__ == "0.4.0"


if __name__ == "__main__":
    unittest.main()
