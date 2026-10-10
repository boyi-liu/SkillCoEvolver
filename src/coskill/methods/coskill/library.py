"""JSON-backed storage and retrieval for CoSkill's progressive trees."""

from __future__ import annotations

import json
import os
import re
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
    """Persist and retrieve progressive skill trees in a JSON file.

    The ``:memory:`` path keeps the historical in-memory mode used by tests
    and short-lived callers. For regular paths, the file contains all skills
    in a single JSON document and is rewritten atomically after each save.
    """

    FORMAT_VERSION = 1

    def __init__(self, path: str | Path = "coskill.json") -> None:
        self.path = str(path)
        self._in_memory = self.path == ":memory:"
        self._file = None if self._in_memory else Path(path).expanduser()
        self._entries: dict[str, LibraryEntry] = {}
        if self._file is not None:
            self._load()

    def close(self) -> None:
        """Close the library.

        JSON storage has no open resource, so closing is a no-op.
        The method remains part of the API for context-manager compatibility.
        """

    def __enter__(self) -> "SkillLibrary":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def save(self, tree: ProgressiveSkillTree) -> None:
        """Persist the latest revision of a skill tree."""
        self._entries[tree.task_type] = LibraryEntry(
            task_type=tree.task_type,
            goal=tree.goal,
            tree=tree,
        )
        if self._file is not None:
            self._write()

    def get(self, task_type: str) -> LibraryEntry | None:
        """Return one stored skill entry."""
        return self._entries.get(task_type)

    def retrieve(self, task_type: str, goal: str) -> LibraryEntry | None:
        """Find an exact task skill or the closest reusable goal by token overlap."""
        exact = self.get(task_type)
        if exact:
            return exact
        query = _terms(f"{task_type} {goal}")
        best: tuple[float, LibraryEntry] | None = None
        for entry in self._entries.values():
            candidate = _terms(f"{entry.task_type} {entry.goal}")
            score = len(query & candidate) / max(1, len(query | candidate))
            if score > 0 and (best is None or score > best[0]):
                best = (score, entry)
        return best[1] if best else None

    def _load(self) -> None:
        assert self._file is not None
        if not self._file.exists():
            return
        try:
            payload = json.loads(self._file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Skill library {self._file} is not a valid JSON file. "
                "Use a new .json path or convert the existing library first."
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Skill library {self._file} must contain a JSON object")

        raw_entries = payload.get("skills", payload)
        if not isinstance(raw_entries, dict):
            raise ValueError(f"Skill library {self._file} has an invalid skills object")
        for task_type, raw_entry in raw_entries.items():
            if not isinstance(raw_entry, dict):
                continue
            raw_tree = raw_entry.get("tree", raw_entry)
            if not isinstance(raw_tree, dict):
                continue
            tree = ProgressiveSkillTree.from_dict(raw_tree)
            resolved_task_type = str(task_type)
            tree.task_type = resolved_task_type
            self._entries[resolved_task_type] = LibraryEntry(
                task_type=resolved_task_type,
                goal=str(raw_entry.get("goal", tree.goal)),
                tree=tree,
            )

    def _write(self) -> None:
        assert self._file is not None
        self._file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.FORMAT_VERSION,
            "skills": {
                task_type: {
                    "goal": entry.goal,
                    "tree": entry.tree.to_dict(),
                }
                for task_type, entry in sorted(self._entries.items())
            },
        }
        temporary = self._file.with_name(f".{self._file.name}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self._file)
