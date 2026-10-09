"""Prefix-merged trajectory trie used as compact cloud-side evidence."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..utils.models import Trajectory


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _node_key(state: str, action: str) -> str:
    canonical = f"{_normalize(state)}\x1f{_normalize(action)}"
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


class ObservationCodec:
    """Encode repeated observations as full text or compact prefix deltas."""

    @staticmethod
    def encode(previous: str, current: str) -> dict[str, Any]:
        """Encode the current observation relative to the previous one."""
        if not previous or not current:
            return {"mode": "full", "text": current}
        prefix = 0
        limit = min(len(previous), len(current))
        while prefix < limit and previous[prefix] == current[prefix]:
            prefix += 1
        suffix = 0
        suffix_limit = min(len(previous) - prefix, len(current) - prefix)
        while suffix < suffix_limit and previous[-suffix - 1] == current[-suffix - 1]:
            suffix += 1
        replacement_end = len(current) - suffix if suffix else len(current)
        delta = {
            "mode": "delta",
            "prefix": prefix,
            "remove": len(previous) - prefix - suffix,
            "insert": current[prefix:replacement_end],
            "suffix": suffix,
        }
        # The paper uses deltas only when they are shorter than half the full text.
        compact_size = len(delta["insert"]) + len(str(prefix)) + len(str(delta["remove"])) + 8
        if compact_size < len(current) / 2:
            return delta
        return {"mode": "full", "text": current}

    @staticmethod
    def decode(previous: str, encoded: dict[str, Any]) -> str:
        """Reconstruct an observation from a full value or encoded delta."""
        if encoded.get("mode") == "full":
            return str(encoded.get("text", ""))
        prefix = int(encoded["prefix"])
        suffix = int(encoded["suffix"])
        tail = previous[len(previous) - suffix :] if suffix else ""
        return previous[:prefix] + str(encoded["insert"]) + tail


@dataclass(slots=True)
class TrieNode:
    """Aggregated statistics and branches for one state-action prefix."""

    key: str
    state: str = ""
    action: str = "<root>"
    observation: dict[str, Any] = field(
        default_factory=lambda: {"mode": "full", "text": ""}
    )
    count: int = 0
    successes: int = 0
    failures: int = 0
    terminal_successes: int = 0
    terminal_failures: int = 0
    children: dict[str, "TrieNode"] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successes / self.count if self.count else 0.0

    @property
    def is_fork(self) -> bool:
        return len(self.children) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "state": self.state,
            "action": self.action,
            "observation": self.observation,
            "count": self.count,
            "successes": self.successes,
            "failures": self.failures,
            "terminal_successes": self.terminal_successes,
            "terminal_failures": self.terminal_failures,
            "children": [child.to_dict() for child in self.children.values()],
        }


class TrajectoryTrie:
    """Merge trajectories by normalized state-action prefixes."""

    def __init__(self, task_type: str) -> None:
        self.task_type = task_type
        self.root = TrieNode(key="root")
        self.trajectory_count = 0

    @classmethod
    def from_trajectories(
        cls, task_type: str, trajectories: Iterable[Trajectory]
    ) -> "TrajectoryTrie":
        trie = cls(task_type)
        for trajectory in trajectories:
            trie.add(trajectory)
        return trie

    def add(self, trajectory: Trajectory) -> None:
        """Insert one trajectory and update counts along its prefix path."""
        if trajectory.task_type != self.task_type:
            raise ValueError(
                f"Expected task type {self.task_type!r}, got {trajectory.task_type!r}"
            )
        self.trajectory_count += 1
        node = self.root
        node.count += 1
        node.successes += int(trajectory.success)
        node.failures += int(not trajectory.success)
        previous_observation = ""
        for step in trajectory.steps:
            key = _node_key(step.state, step.action)
            if key not in node.children:
                node.children[key] = TrieNode(
                    key=key,
                    state=step.state,
                    action=step.action,
                    observation=ObservationCodec.encode(
                        previous_observation, step.observation
                    ),
                )
            node = node.children[key]
            node.count += 1
            node.successes += int(trajectory.success)
            node.failures += int(not trajectory.success)
            previous_observation = step.observation
        node.terminal_successes += int(trajectory.success)
        node.terminal_failures += int(not trajectory.success)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "trajectory_count": self.trajectory_count,
            "root": self.root.to_dict(),
        }

    def to_markdown(self, *, observation_limit: int = 300) -> str:
        """Render the trie as readable evidence for the cloud maintainer."""
        lines = [
            f"# Prefix-Merged Trajectory Trie: {self.task_type}",
            f"Trajectories: {self.trajectory_count}",
        ]

        def visit(node: TrieNode, prefix: str, previous_observation: str) -> None:
            children = list(node.children.values())
            for index, child in enumerate(children):
                last = index == len(children) - 1
                branch = "└─" if last else "├─"
                fork = " [FORK]" if node.is_fork else ""
                stats = (
                    f"n={child.count}, S={child.successes}, F={child.failures}, "
                    f"rate={child.success_rate:.2f}"
                )
                lines.append(
                    f"{prefix}{branch} {child.action}{fork} "
                    f"[key={child.key}] ({stats})"
                )
                if child.state:
                    clean_state = child.state.replace("\n", " ")[:observation_limit]
                    lines.append(
                        f"{prefix}{'   ' if last else '│  '} state: {clean_state}"
                    )
                observation = ObservationCodec.decode(
                    previous_observation, child.observation
                )
                if observation:
                    clean = observation.replace("\n", " ")[:observation_limit]
                    mode = child.observation.get("mode", "full")
                    lines.append(f"{prefix}{'   ' if last else '│  '} obs[{mode}]: {clean}")
                if child.terminal_successes or child.terminal_failures:
                    lines.append(
                        f"{prefix}{'   ' if last else '│  '} END "
                        f"(S={child.terminal_successes}, F={child.terminal_failures})"
                    )
                visit(
                    child,
                    prefix + ("   " if last else "│  "),
                    observation,
                )

        visit(self.root, "", "")
        return "\n".join(lines)
