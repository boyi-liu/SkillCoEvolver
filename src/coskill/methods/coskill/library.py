"""SQLite-backed storage and retrieval for CoSkill's progressive trees."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .tree import ProgressiveSkillTree


@dataclass(slots=True)
class LibraryEntry:
    task_type: str
    goal: str
    tree: ProgressiveSkillTree
    state: str
    uses: int
    successes: int
    revisions: int
    covered_task_types: set[str]


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[\w-]+", text.casefold()))


class SkillLibrary:
    """SQLite persistence for Idle -> Active -> Resident cloud skills."""

    def __init__(
        self, path: str | Path = "coskill.db", *, enable_discard: bool = False
    ) -> None:
        self.path = str(path)
        self.enable_discard = enable_discard
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS skills (
                task_type TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                tree_json TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'Idle',
                uses INTEGER NOT NULL DEFAULT 0,
                successes INTEGER NOT NULL DEFAULT 0,
                revisions INTEGER NOT NULL DEFAULT 0,
                covered_task_types TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SkillLibrary":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def save(self, tree: ProgressiveSkillTree) -> None:
        """Persist a tree while preserving its usage and lifecycle counters."""
        existing = self.get(tree.task_type, include_discarded=True)
        uses = existing.uses if existing else 0
        successes = existing.successes if existing else 0
        state = existing.state if existing else "Idle"
        covered = existing.covered_task_types if existing else {tree.task_type}
        self._connection.execute(
            """
            INSERT INTO skills (
                task_type, goal, tree_json, state, uses, successes,
                revisions, covered_task_types, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(task_type) DO UPDATE SET
                goal=excluded.goal,
                tree_json=excluded.tree_json,
                state=excluded.state,
                uses=excluded.uses,
                successes=excluded.successes,
                revisions=excluded.revisions,
                covered_task_types=excluded.covered_task_types,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                tree.task_type,
                tree.goal,
                tree.to_json(),
                state,
                uses,
                successes,
                tree.revision,
                json.dumps(sorted(covered)),
            ),
        )
        self._connection.commit()

    def get(
        self, task_type: str, *, include_discarded: bool = False
    ) -> LibraryEntry | None:
        """Return one stored skill entry unless it has been discarded."""
        row = self._connection.execute(
            "SELECT * FROM skills WHERE task_type = ?", (task_type,)
        ).fetchone()
        if row is None or (row["state"] == "Discarded" and not include_discarded):
            return None
        return self._row_to_entry(row)

    def retrieve(self, task_type: str, goal: str) -> LibraryEntry | None:
        """Find an exact task skill or the closest reusable goal by token overlap."""
        exact = self.get(task_type)
        if exact:
            return exact
        query = _terms(f"{task_type} {goal}")
        best: tuple[float, LibraryEntry] | None = None
        for row in self._connection.execute(
            "SELECT * FROM skills WHERE state != 'Discarded'"
        ):
            entry = self._row_to_entry(row)
            candidate = _terms(f"{entry.task_type} {entry.goal}")
            score = len(query & candidate) / max(1, len(query | candidate))
            if score > 0 and (best is None or score > best[0]):
                best = (score, entry)
        return best[1] if best else None

    def record_use(
        self, task_type: str, success: bool, *, observed_task_type: str | None = None
    ) -> str:
        """Update usage statistics and return the skill's resulting lifecycle state."""
        entry = self.get(task_type, include_discarded=True)
        if entry is None:
            raise KeyError(task_type)
        uses = entry.uses + 1
        successes = entry.successes + int(success)
        covered = set(entry.covered_task_types)
        covered.add(observed_task_type or task_type)
        rate = successes / uses
        # Give a cold-start skill time to receive its first cloud update.  The
        # previous 10-use gate could discard a skill before the default
        # 16-trajectory update had a chance to run.
        if self.enable_discard and uses >= 32 and entry.revisions >= 1 and rate < 0.30:
            state = "Discarded"
        elif len(covered) >= 3:
            state = "Resident"
        elif entry.revisions >= 2 or uses >= 10:
            state = "Active"
        else:
            state = "Idle"
        self._connection.execute(
            """
            UPDATE skills SET uses=?, successes=?, state=?, covered_task_types=?,
                updated_at=CURRENT_TIMESTAMP WHERE task_type=?
            """,
            (uses, successes, state, json.dumps(sorted(covered)), task_type),
        )
        self._connection.commit()
        return state

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> LibraryEntry:
        return LibraryEntry(
            task_type=row["task_type"],
            goal=row["goal"],
            tree=ProgressiveSkillTree.from_json(row["tree_json"]),
            state=row["state"],
            uses=int(row["uses"]),
            successes=int(row["successes"]),
            revisions=int(row["revisions"]),
            covered_task_types=set(json.loads(row["covered_task_types"])),
        )
