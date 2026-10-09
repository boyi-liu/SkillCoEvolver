"""Durable JSONL and latest-snapshot recording for experiment artifacts."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .models import Trajectory
from ..skills import ProgressiveSkillTree


class RunArtifactRecorder:
    """Persist trajectories and skill-tree snapshots as a run progresses."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_skills = self.root / "skills"
        self.latest_skills.mkdir(parents=True, exist_ok=True)
        self.trajectories_path = self.root / "trajectories.jsonl"
        self.skill_trees_path = self.root / "skill_trees.jsonl"

    def record(
        self,
        trajectory: Trajectory,
        tree: ProgressiveSkillTree,
        *,
        episode_index: int | None = None,
        evolution_triggered: bool = False,
        evolution_summary: str = "",
    ) -> None:
        """Append an episode record and its corresponding skill-tree snapshot."""
        saved_at = datetime.now(timezone.utc).isoformat()
        self._append_jsonl(
            self.trajectories_path,
            {
                "saved_at": saved_at,
                "episode": episode_index,
                "trajectory": trajectory.to_dict(),
                "evolution_triggered": evolution_triggered,
                "evolution_summary": evolution_summary,
            },
        )
        self.record_tree(
            tree,
            episode_index=episode_index,
            event="episode",
            saved_at=saved_at,
        )

    def record_tree(
        self,
        tree: ProgressiveSkillTree,
        *,
        episode_index: int | None = None,
        event: str = "snapshot",
        saved_at: str | None = None,
    ) -> None:
        """Append a tree snapshot and atomically refresh its latest JSON file."""
        saved_at = saved_at or datetime.now(timezone.utc).isoformat()
        tree_value = tree.to_dict()
        self._append_jsonl(
            self.skill_trees_path,
            {
                "saved_at": saved_at,
                "episode": episode_index,
                "event": event,
                "task_type": tree.task_type,
                "revision": tree.revision,
                "tree": tree_value,
            },
        )
        self._write_latest_tree(tree)

    def _write_latest_tree(self, tree: ProgressiveSkillTree) -> None:
        filename = f"{self._safe_name(tree.task_type)}.json"
        target = self.latest_skills / filename
        temporary = target.with_name(f".{filename}.tmp")
        payload = json.dumps(tree.to_dict(), ensure_ascii=False, indent=2) + "\n"
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)

    @staticmethod
    def _append_jsonl(path: Path, value: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _safe_name(value: str) -> str:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        return name or "unknown"
