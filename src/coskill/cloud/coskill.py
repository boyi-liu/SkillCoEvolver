"""Co-Skill cloud specialization: trie evidence and progressive evolution."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..llm import LLMClient, Message
from ..skills import ProgressiveSkillTree, SkillLibrary
from ..utils.models import Task, TokenUsage, Trajectory
from .base import BaseCloud, EvolutionDraft, SkillArtifact
from .utils.prompt import CLOUD_SYSTEM_PROMPT, cloud_prompt


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


def _object_list(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)][:limit]


@dataclass(slots=True)
class EvolutionResult:
    """Summary of one cloud evolution request and its applied operations."""

    summary: str
    diagnoses: list[dict[str, Any]] = field(default_factory=list)
    contrasts: list[dict[str, Any]] = field(default_factory=list)
    applied_operations: list[str] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)


class CloudSkillEvolver:
    """Use a cloud LLM to maintain a progressive skill tree."""

    def __init__(
        self,
        client: LLMClient,
        *,
        observation_limit: int = 300,
        max_operations: int = 3,
    ) -> None:
        if observation_limit <= 0:
            raise ValueError("observation_limit must be positive")
        if not 1 <= max_operations <= 3:
            raise ValueError("max_operations must be between 1 and 3")
        self.client = client
        self.observation_limit = observation_limit
        self.max_operations = max_operations

    async def evolve(
        self,
        tree: ProgressiveSkillTree,
        trie: TrajectoryTrie,
        trigger_reasons: list[str],
    ) -> EvolutionResult:
        """Request a minimal tree update from the cloud using the trajectory trie."""

        payload, response = await self.client.complete_json(
            [
                Message("system", CLOUD_SYSTEM_PROMPT),
                Message(
                    "user",
                    cloud_prompt(
                        tree.guidance(),
                        trie.to_markdown(observation_limit=self.observation_limit),
                        trigger_reasons,
                    ),
                ),
            ]
        )
        operations = _object_list(
            payload.get("operations"), limit=self.max_operations
        )
        for operation in operations:
            operation["children"] = _object_list(
                operation.get("children"), limit=3
            )
        applied = tree.apply_operations(operations)
        return EvolutionResult(
            summary=str(payload.get("summary", "Skill tree updated")),
            diagnoses=_object_list(
                payload.get("diagnoses"), limit=self.max_operations
            ),
            contrasts=_object_list(
                payload.get("contrasts"), limit=self.max_operations
            ),
            applied_operations=applied,
            usage=response.usage,
        )


class CoSkillCloud(BaseCloud):
    """Co-Skill cloud specialization using trie evidence and tree patches."""

    def __init__(
        self,
        client: LLMClient,
        library: SkillLibrary,
        *,
        observation_limit: int = 300,
        max_operations: int = 3,
    ) -> None:
        super().__init__(client)
        self.library = library
        self.evolver = CloudSkillEvolver(
            client,
            observation_limit=observation_limit,
            max_operations=max_operations,
        )
        self.borrowed_from: dict[str, str] = {}

    def retrieve_skill(self, task: Task) -> ProgressiveSkillTree:
        reusable_goal = task.skill_template or task.goal
        entry = self.library.retrieve(task.task_type, reusable_goal)
        if entry is not None:
            if entry.task_type == task.task_type:
                return entry.tree
            tree = ProgressiveSkillTree.from_dict(entry.tree.to_dict())
            tree.task_type = task.task_type
            tree.goal = reusable_goal
            tree.root.instruction = reusable_goal
            self.commit_skill(tree)
            self.borrowed_from[task.task_type] = entry.task_type
            return tree
        tree = ProgressiveSkillTree(task.task_type, reusable_goal)
        self.commit_skill(tree)
        return tree

    def lookup_skill(self, task_type: str) -> ProgressiveSkillTree | None:
        entry = self.library.get(task_type)
        return entry.tree if entry is not None else None

    def commit_skill(self, skill: SkillArtifact) -> None:
        if not isinstance(skill, ProgressiveSkillTree):
            raise TypeError("CoSkill cloud registry expects ProgressiveSkillTree")
        self.library.save(skill)

    def is_skill_discarded(self, task_type: str) -> bool:
        entry = self.library.get(task_type, include_discarded=True)
        return entry is not None and entry.state == "Discarded"

    def record_skill_use(
        self,
        skill: SkillArtifact,
        task: Task,
        trajectory: Trajectory,
    ) -> None:
        if not isinstance(skill, ProgressiveSkillTree):
            raise TypeError("CoSkill usage tracking expects ProgressiveSkillTree")
        self.library.record_use(
            skill.task_type,
            trajectory.success,
            observed_task_type=task.task_type,
        )
        source_task_type = self.borrowed_from.get(task.task_type)
        if source_task_type is not None:
            self.library.record_use(
                source_task_type,
                trajectory.success,
                observed_task_type=task.task_type,
            )

    def compile_evidence(
        self, task_type: str, trajectories: list[Trajectory]
    ) -> TrajectoryTrie:
        return TrajectoryTrie.from_trajectories(task_type, trajectories)

    async def evolve_skill(
        self,
        skill: SkillArtifact,
        evidence: Any,
        trigger_reasons: list[str],
        *,
        framework_name: str,
    ) -> EvolutionDraft:
        del framework_name
        if not isinstance(skill, ProgressiveSkillTree):
            raise TypeError("CoSkill cloud evolution expects ProgressiveSkillTree")
        if not isinstance(evidence, TrajectoryTrie):
            raise TypeError("CoSkill cloud evolution expects TrajectoryTrie")
        candidate = ProgressiveSkillTree.from_dict(skill.to_dict())
        evolution = await self.evolver.evolve(
            candidate, evidence, trigger_reasons
        )
        return EvolutionDraft(
            candidate=candidate,
            summary=evolution.summary,
            usage=evolution.usage,
            details={
                "diagnoses": evolution.diagnoses,
                "contrasts": evolution.contrasts,
                "applied_operations": evolution.applied_operations,
                "mode": "trie-progressive-tree-patching",
            },
        )


__all__ = [
    "CloudSkillEvolver",
    "CoSkillCloud",
    "EvolutionResult",
    "ObservationCodec",
    "TrajectoryTrie",
    "TrieNode",
]
