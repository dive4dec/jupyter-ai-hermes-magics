"""Session tree with dot-notation labels.

Labels like ``main.sub1.sub2`` create a tree of dependent sessions.
Each child inherits the parent's conversation history (fork/branch).

Session IDs are stored in the notebook's cell metadata under
``cell.metadata["hermes"]["session_id"]`` and resolved at runtime via
``hermes sessions list`` + the Hermes state database.

Tree semantics:
  ``main``             — root session, fresh
  ``main.explain``     — child: forks ``main``'s history at creation time
  ``main.explain.fix`` — child of ``main.explain``
  ``main.test``        — sibling of ``main.explain``, also forks from ``main``

The first ``%%hermes`` call in a notebook creates the root (default label:
``main``).  Subsequent calls without a label auto-resume the most recent
session in the tree.  ``--label foo.bar`` creates or resumes ``foo.bar``.
"""

from __future__ import annotations

import logging
import sqlite3
import subprocess
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionNode:
    """A node in the session tree."""

    label: str  # dot-notation label, e.g. "main.explain.fix"
    session_id: Optional[str] = None  # Hermes session ID (e.g. "20260728_155751_cd497f")
    parent_label: Optional[str] = None  # parent label (e.g. "main.explain")
    parent_session_id: Optional[str] = None  # parent's session ID at fork time

    def __post_init__(self):
        if self.parent_label is None and "." in self.label:
            self.parent_label = self.label.rsplit(".", 1)[0]


class SessionTree:
    """Manages a tree of Hermes sessions with dot-notation labels.

    The tree is stored in the notebook's metadata under
    ``nb.metadata["hermes_magics"]["sessions"]`` as a dict of label → node.
    """

    ROOT_LABEL = "main"
    METADATA_KEY = "hermes_magics"

    def __init__(self, notebook_metadata: dict | None = None):
        """Initialize from notebook metadata.

        Args:
            notebook_metadata: The live ``notebook.metadata`` dict.
                Changes are written in-place so the notebook saves them.
        """
        self._metadata = notebook_metadata if notebook_metadata is not None else {}
        if self.METADATA_KEY not in self._metadata:
            self._metadata[self.METADATA_KEY] = {"sessions": {}}
        elif "sessions" not in self._metadata[self.METADATA_KEY]:
            self._metadata[self.METADATA_KEY]["sessions"] = {}

        self._nodes: dict[str, SessionNode] = {}
        for label, data in self._metadata[self.METADATA_KEY]["sessions"].items():
            node = SessionNode(
                label=label,
                session_id=data.get("session_id"),
                parent_label=data.get("parent_label"),
                parent_session_id=data.get("parent_session_id"),
            )
            self._nodes[label] = node

    @property
    def root_label(self) -> str:
        return self.ROOT_LABEL

    def get(self, label: str) -> Optional[SessionNode]:
        """Get a node by label, or None if it doesn't exist."""
        return self._nodes.get(label)

    def get_or_create(self, label: str) -> SessionNode:
        """Get an existing node, or create a new one.

        New nodes inherit the parent's session_id as parent_session_id
        (for fork semantics).
        """
        if label in self._nodes:
            return self._nodes[label]

        node = SessionNode(label=label)
        # Resolve parent
        if node.parent_label and node.parent_label in self._nodes:
            parent = self._nodes[node.parent_label]
            node.parent_session_id = parent.session_id
        elif not node.parent_label:
            # Root node
            pass
        else:
            # Parent doesn't exist yet — create it recursively
            parent = self.get_or_create(node.parent_label)
            node.parent_session_id = parent.session_id

        self._nodes[label] = node
        self._persist(node)
        return node

    def update_session_id(self, label: str, session_id: str) -> None:
        """Update the Hermes session ID for a label (after first call or resume)."""
        node = self._nodes.get(label)
        if node is None:
            node = self.get_or_create(label)
        node.session_id = session_id
        self._persist(node)

    def latest_label(self) -> Optional[str]:
        """Return the most recently used label, or None if tree is empty."""
        if not self._nodes:
            return None
        # Return root if it exists, else the first node
        if self.ROOT_LABEL in self._nodes:
            return self.ROOT_LABEL
        return next(iter(self._nodes))

    def list_labels(self) -> list[str]:
        """Return all labels in the tree, sorted hierarchically."""
        return sorted(self._nodes.keys(), key=lambda l: l.split("."))

    def _persist(self, node: SessionNode) -> None:
        """Write node to notebook metadata (in-place)."""
        self._metadata[self.METADATA_KEY]["sessions"][node.label] = {
            "session_id": node.session_id,
            "parent_label": node.parent_label,
            "parent_session_id": node.parent_session_id,
        }

    # ── Hermes session DB helpers ──────────────────────────────────────

    @staticmethod
    def get_hermes_db_path() -> Path:
        """Locate the Hermes state.db."""
        hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        return Path(hermes_home) / "state.db"

    @staticmethod
    def get_latest_session_id(source: str = "cli") -> Optional[str]:
        """Get the most recent Hermes session ID from the state DB.

        Used after a ``hermes -z`` call to find the session that was just
        created/resumed.
        """
        db_path = SessionTree.get_hermes_db_path()
        if not db_path.exists():
            return None
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id FROM sessions WHERE source = ? ORDER BY started_at DESC LIMIT 1",
                (source,),
            ).fetchone()
            conn.close()
            return row["id"] if row else None
        except Exception as e:
            logger.debug("Failed to query Hermes DB: %s", e)
            return None

    @staticmethod
    def get_session_history(session_id: str) -> list[dict]:
        """Retrieve conversation history for a session from Hermes DB.

        Used when forking: the child session inherits the parent's messages.
        """
        db_path = SessionTree.get_hermes_db_path()
        if not db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            conn.close()
            return [{"role": r["role"], "content": r["content"]} for r in rows]
        except Exception as e:
            logger.debug("Failed to query session history: %s", e)
            return []
