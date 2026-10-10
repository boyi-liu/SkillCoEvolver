"""SQLite-backed storage and retrieval for CoSkill's progressive trees."""

from __future__ import annotations

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


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[\w-]+", text.casefold()))


class SkillLibrary:
    """SQLite persistence and retrieval for progressive skill trees."""

    def __init__(self, path: str | Path = "coskill.db") -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS skills (
                task_type TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                tree_json TEXT NOT NULL,
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
        """Persist the latest revision of a skill tree."""
        self._connection.execute(
            """
            INSERT INTO skills (task_type, goal, tree_json, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(task_type) DO UPDATE SET
                goal=excluded.goal,
                tree_json=excluded.tree_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                tree.task_type,
                tree.goal,
                tree.to_json(),
            ),
        )
        self._connection.commit()

    def get(self, task_type: str) -> LibraryEntry | None:
        """Return one stored skill entry."""
        row = self._connection.execute(
            "SELECT * FROM skills WHERE task_type = ?", (task_type,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def retrieve(self, task_type: str, goal: str) -> LibraryEntry | None:
        """Find an exact task skill or the closest reusable goal by token overlap."""
        exact = self.get(task_type)
        if exact:
            return exact
        query = _terms(f"{task_type} {goal}")
        best: tuple[float, LibraryEntry] | None = None
        for row in self._connection.execute("SELECT * FROM skills"):
            entry = self._row_to_entry(row)
            candidate = _terms(f"{entry.task_type} {entry.goal}")
            score = len(query & candidate) / max(1, len(query | candidate))
            if score > 0 and (best is None or score > best[0]):
                best = (score, entry)
        return best[1] if best else None

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> LibraryEntry:
        return LibraryEntry(
            task_type=row["task_type"],
            goal=row["goal"],
            tree=ProgressiveSkillTree.from_json(row["tree_json"]),
        )
