"""Progressive, serializable skill trees used by CoSkill."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable
from uuid import uuid4


@dataclass(slots=True)
class SkillNode:
    """One reusable instruction node with optional progressively added children."""

    title: str
    instruction: str
    node_id: str = field(default_factory=lambda: uuid4().hex[:12])
    children: list["SkillNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["children"] = [child.to_dict() for child in self.children]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SkillNode":
        value = dict(value)
        # Ignore counters written by databases created before lifecycle
        # tracking was removed.
        value.pop("uses", None)
        value.pop("successes", None)
        value["children"] = [cls.from_dict(child) for child in value.get("children", [])]
        return cls(**value)


class ProgressiveSkillTree:
    """Task-scoped tree that accepts constrained refine and expand operations."""

    VALID_OPERATIONS = {"refine", "expand"}

    def __init__(self, task_type: str, goal: str, root: SkillNode | None = None) -> None:
        self.task_type = task_type
        self.goal = goal
        self.root = root or SkillNode(title="Task goal", instruction=goal)
        self.revision = 0

    def walk(self) -> Iterable[tuple[SkillNode, int]]:
        stack = [(self.root, 0)]
        while stack:
            node, level = stack.pop()
            yield node, level
            stack.extend((child, level + 1) for child in reversed(node.children))

    def find(self, node_id: str) -> SkillNode | None:
        return next((node for node, _ in self.walk() if node.node_id == node_id), None)

    @property
    def max_depth(self) -> int:
        return max(level for _, level in self.walk())

    def apply_operations(self, operations: list[dict[str, Any]]) -> list[str]:
        """Apply valid cloud operations and return IDs of changed nodes."""
        applied: list[str] = []
        for change in operations:
            operation = str(change.get("operation", "")).lower()
            if operation not in self.VALID_OPERATIONS:
                continue
            node = self.find(str(change.get("node_id", "")))
            if node is None:
                continue
            changed = False
            title = str(change.get("title", "")).strip()
            instruction = str(change.get("instruction", "")).strip()
            if title and title != node.title:
                node.title = title
                changed = True
            if instruction and instruction != node.instruction:
                node.instruction = instruction
                changed = True
            raw_children = change.get("children", [])
            if not isinstance(raw_children, list):
                raw_children = []
            children = [
                SkillNode(
                    title=str(child.get("title", "Step")),
                    instruction=str(child.get("instruction", "")),
                )
                for child in raw_children[:3]
                if isinstance(child, dict)
            ]
            if operation == "expand":
                # Progressive growth: a single cloud update adds one level only.
                existing = {
                    (child.title.casefold(), child.instruction.casefold())
                    for child in node.children
                }
                additions = [
                    child
                    for child in children
                    if (child.title.casefold(), child.instruction.casefold())
                    not in existing
                ]
                node.children.extend(additions)
                changed = changed or bool(additions)
            if changed:
                applied.append(f"{operation}:{node.node_id}")
        if applied:
            self.revision += 1
        return applied

    def guidance(self, *, max_depth: int | None = None) -> str:
        """Render tree instructions in an indented format for the edge prompt."""
        lines: list[str] = []
        for node, level in self.walk():
            if max_depth is not None and level > max_depth:
                continue
            indent = "  " * level
            lines.append(
                f"{indent}- [L{level} id={node.node_id}] {node.title}: {node.instruction}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "goal": self.goal,
            "revision": self.revision,
            "root": self.root.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProgressiveSkillTree":
        tree = cls(
            task_type=str(value["task_type"]),
            goal=str(value["goal"]),
            root=SkillNode.from_dict(value["root"]),
        )
        tree.revision = int(value.get("revision", 0))
        return tree

    @classmethod
    def from_json(cls, value: str) -> "ProgressiveSkillTree":
        return cls.from_dict(json.loads(value))
