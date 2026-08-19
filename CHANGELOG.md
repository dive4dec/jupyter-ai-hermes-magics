# Changelog

All notable changes to `jupyter-ai-hermes-magics` are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.5.0] — 2026-08-19

### Changed
- **Context injection now sends a one-line outline instead of full cell
  source.** Each cell up to and including the `%%hermes` cell is rendered as
  a single line (index, type, cell `id`, exec count, ~100-char preview).
  Hermes fetches any cell's full content on demand via `read_notebook_cells`
  by `cell_id`. On a 60-cell notebook with several ~1.5 KB plot cells this
  cut the injected context from ~26.7 KB to ~6.7 KB (≈75% smaller), and the
  reduction grows with notebook size — so large notebooks no longer blow out
  the prompt or trip context compaction.

- **Enriched `MCP_TOOLS_DOC` in native-MCP form.** The magic now documents the
  full Jupyter MCP tool set (~21 tools: reading, editing, running,
  metadata/tags, navigation, JupyterLab-command escape hatch) as **native MCP
  tools** — the same way the ACP session actually invokes them (registered at
  session creation). The doc now states the `notebook_path` / `file_path` /
  `cell_id` parameter convention (the #1 source of silent tool failures) and
  the "cell above = ACTIVE index − 1, fetch by `id`" pattern. This removes
  the `tool_search` → `tool_describe` detour Hermes previously needed to
  discover `read_notebook_cells`.

### Fixed
- **Outline anchored on the real magic cell, not the UI-focused cell.**
  `gather_context()` previously anchored on `get_active_cell_id()` — the cell
  with the user's cursor in the JupyterLab UI — which is **not** reliably the
  cell containing the `%%hermes` magic. When the user's focus was on a
  different cell, the outline stopped at the wrong cell and "the cell above"
  resolved to the wrong one. The magic's own cell ID (taken reliably from
  IPython's `ip.get_parent()["metadata"]["cellId"]`) is now passed into
  `gather_context(magic_cell_id=...)` and used as the primary anchor, with
  `get_active_cell_id()` only as a fallback. This was a latent bug in the
  previous full-dump code as well (same anchor), surfaced by the outline work.

- **Connection status no longer leaks into notebook output.** The
  `✓ Hermes ACP connected …`, `✓ … already connected`, and
  `⟳ … starting background connection` `print()`s were firing from the
  background init thread, whose stdout attaches to whichever cell was
  executing — so the message appeared in a random "current output cell".
  These are now `logger.debug`. The `✗ … init failed` error message is kept
  (on stderr) since a broken connection is the case that actually needs
  surfacing.

### Tests
- Added outline unit tests (`_outline_line` truncation/shape, `gather_context`
  outline-vs-full-source, length boundedness independent of cell count).
- Added anchor regression tests reproducing the "magic cell ≠ UI-focus cell"
  case (asserts the magic cell — not the focused one — is marked ACTIVE).
- Added `MCP_TOOLS_DOC` tests (native form not CLI, full tool set, param
  convention, on-demand-fetch pattern).

### Notes
- `%hermes_help` "CONTEXT INJECTION" section updated to describe the outline
  model (previously described the old per-cell 2000-char truncation).
- Version bump `0.4.0 → 0.5.0` (behavioral change to context handling — minor
  bump per release policy).

## [0.4.0] — 2026-08-19
- Persistent ACP connection started in a background thread at `%load_ext`,
  with transcript cell placement fix.
- Multi-session ACP: per-label session creation and isolation.
- `%hermes help` subcommand with full docs.
