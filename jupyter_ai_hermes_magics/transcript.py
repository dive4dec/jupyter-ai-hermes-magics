"""Transcript cell group — manages the output cells below a %%hermes cell.

When ``%%hermes`` runs, the response goes into a **transcription cell group**
below the magic cell.  This group is linked to the magic cell so that
re-executing the magic cell **updates** the group instead of creating a new one.

The group can be:
  - A single markdown cell (default — Hermes responses are markdown)
  - A single code cell (when ``--format code`` is used)
  - A mix of markdown and code cells (when Hermes generates code + prose)

Linking is done via cell metadata:
  ``cell.metadata["hermes_transcript"]["source_cell_id"] = <magic cell id>``
  ``cell.metadata["hermes_transcript"]["session_label"] = <label>``

This module works at the ``nbformat`` level so it functions in any
environment that has the notebook object — JupyterLab, VS Code, nbconvert.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import nbformat

logger = logging.getLogger(__name__)

TRANSCRIPT_META_KEY = "hermes_transcript"


def _find_cell_index_by_id(nb: nbformat.NotebookNode, cell_id: str) -> Optional[int]:
    """Find the index of a cell by its ID."""
    for i, cell in enumerate(nb.cells):
        if cell.get("id") == cell_id:
            return i
    return None


def _ensure_cell_ids(nb: nbformat.NotebookNode) -> None:
    """Ensure all cells have IDs (nbformat requires them but some envs don't set them)."""
    for cell in nb.cells:
        if not cell.get("id"):
            cell["id"] = str(uuid.uuid4())


def find_transcript_group(
    nb: nbformat.NotebookNode, magic_cell_id: str
) -> list[int]:
    """Find the indices of all cells in the transcript group for a magic cell.

    Returns a list of cell indices (possibly empty if no group exists yet).
    """
    indices = []
    for i, cell in enumerate(nb.cells):
        meta = cell.get("metadata", {}).get(TRANSCRIPT_META_KEY, {})
        if meta.get("source_cell_id") == magic_cell_id:
            indices.append(i)
    return indices


def clear_transcript_group(
    nb: nbformat.NotebookNode, magic_cell_id: str
) -> None:
    """Remove all cells in the transcript group for a magic cell.

    Called before writing a new response when re-executing a magic cell.
    """
    # Remove from the end so indices don't shift during iteration
    indices = find_transcript_group(nb, magic_cell_id)
    for i in reversed(indices):
        nb.cells.pop(i)


def get_magic_cell_id(nb: nbformat.NotebookNode, magic_cell_index: int) -> str:
    """Get or assign the ID of the magic cell."""
    _ensure_cell_ids(nb)
    return nb.cells[magic_cell_index]["id"]


def find_magic_cell_index(nb: nbformat.NotebookNode, magic_cell_id: str) -> Optional[int]:
    """Find the index of the magic cell by its ID."""
    return _find_cell_index_by_id(nb, magic_cell_id)


def insert_transcript_cell(
    nb: nbformat.NotebookNode,
    magic_cell_id: str,
    content: str,
    cell_type: str = "markdown",
    session_label: str = "main",
    position: str = "after",
) -> int:
    """Insert a single transcript cell linked to the magic cell.

    Args:
        nb: The notebook object.
        magic_cell_id: The ID of the %%hermes cell.
        content: The cell source text.
        cell_type: "markdown" or "code".
        session_label: The session tree label for this transcript.
        position: "after" (below magic cell) or "before" (above).

    Returns:
        The index of the newly inserted cell.
    """
    _ensure_cell_ids(nb)

    magic_idx = _find_cell_index_by_id(nb, magic_cell_id)
    if magic_idx is None:
        raise ValueError(f"Magic cell {magic_cell_id} not found in notebook")

    new_cell = nbformat.v4.new_markdown_cell(content) if cell_type == "markdown" else nbformat.v4.new_code_cell(content)
    new_cell["id"] = str(uuid.uuid4())
    new_cell["metadata"][TRANSCRIPT_META_KEY] = {
        "source_cell_id": magic_cell_id,
        "session_label": session_label,
    }

    insert_idx = magic_idx + 1 if position == "after" else magic_idx
    nb.cells.insert(insert_idx, new_cell)
    return insert_idx


def write_transcript(
    nb: nbformat.NotebookNode,
    magic_cell_id: str,
    response_text: str,
    session_label: str = "main",
    format: str = "markdown",
) -> None:
    """Write the Hermes response as a transcript cell group.

    Clears any existing group for this magic cell, then inserts a new cell
    with the response.  If the response contains code blocks and format is
    "markdown", the response goes into a single markdown cell (Hermes
    already produces markdown with fenced code blocks).

    If format is "code", any fenced code blocks in the response are extracted
    into separate code cells, with surrounding prose in markdown cells.

    Args:
        nb: The notebook object.
        magic_cell_id: The ID of the %%hermes cell.
        response_text: The full Hermes response text.
        session_label: The session tree label.
        format: "markdown" (single md cell) or "code" (split into md + code cells).
    """
    # Clear existing group
    clear_transcript_group(nb, magic_cell_id)

    if format == "code":
        _write_split_transcript(nb, magic_cell_id, response_text, session_label)
    else:
        insert_transcript_cell(
            nb,
            magic_cell_id,
            response_text,
            cell_type="markdown",
            session_label=session_label,
        )


def _write_split_transcript(
    nb: nbformat.NotebookNode,
    magic_cell_id: str,
    response_text: str,
    session_label: str,
) -> None:
    """Split response into markdown prose + code cells.

    Extracts fenced code blocks from the response and creates separate cells:
    - Text before/between/after code blocks → markdown cells
    - Code blocks → code cells (without the fence)
    """
    import re

    # Pattern: ```lang\ncode\n```
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    last_end = 0
    magic_idx = _find_cell_index_by_id(nb, magic_cell_id)
    if magic_idx is None:
        raise ValueError(f"Magic cell {magic_cell_id} not found")

    # We need to insert cells in order, tracking the insertion offset
    insert_offset = 1  # start right after magic cell

    for match in pattern.finditer(response_text):
        # Prose before this code block
        prose = response_text[last_end:match.start()].strip()
        if prose:
            insert_transcript_cell(
                nb, magic_cell_id, prose, "markdown", session_label,
            )
            insert_offset += 1

        # Code block
        lang = match.group(1)
        code = match.group(2).strip()
        if code:
            insert_transcript_cell(
                nb, magic_cell_id, code, "code", session_label,
            )
            insert_offset += 1

        last_end = match.end()

    # Trailing prose
    prose = response_text[last_end:].strip()
    if prose:
        insert_transcript_cell(
            nb, magic_cell_id, prose, "markdown", session_label,
        )
