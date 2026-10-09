"""Coordinate edge execution, trajectory buffering, and cloud evolution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from ..evolution import (
    CloudSkillEvolver,
    TrajectoryPool,
    TrajectoryTrie,
    TriggerConfig,
)
from ..llm import LLMClient
from ..skills import ProgressiveSkillTree, SkillLibrary
from .agent import EdgeAgent, Environment
from .models import EpisodeResult, Task
from .recorder import RunArtifactRecorder


@dataclass(slots=True)
class CoSkillConfig:
    """Runtime limits and feature switches for the end-to-end controller."""

    max_steps: int = 40
    action_retries: int = 2
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    enable_evolution: bool = True
    artifact_dir: str | Path | None = None


class CoSkill:
    """End-to-end CCF loop with LLM APIs on both device and cloud sides."""

    def __init__(
        self,
        *,
        device_client: LLMClient,
        cloud_client: LLMClient,
        library: SkillLibrary,
        config: CoSkillConfig | None = None,
    ) -> None:
        self.config = config or CoSkillConfig()
        self.edge = EdgeAgent(
            device_client,
            max_steps=self.config.max_steps,
            action_retries=self.config.action_retries,
        )
        self.cloud = CloudSkillEvolver(cloud_client)
        self.library = library
        self.pool = TrajectoryPool(self.config.trigger)
        self.recorder = (
            RunArtifactRecorder(self.config.artifact_dir)
            if self.config.artifact_dir is not None
            else None
        )
        self._evolution_locks: dict[str, asyncio.Lock] = {}
        self._borrowed_from: dict[str, str] = {}

    def skill_tree_for(self, task: Task) -> ProgressiveSkillTree:
        """Load the best reusable skill tree or create a new one for the task."""
        reusable_goal = task.skill_template or task.goal
        entry = self.library.retrieve(task.task_type, reusable_goal)
        if entry:
            if entry.task_type == task.task_type:
                return entry.tree
            # Warm-start a new task from the closest reusable skill while keeping
            # an independent tree for future task-specific refinement.
            tree = ProgressiveSkillTree.from_dict(entry.tree.to_dict())
            tree.task_type = task.task_type
            tree.goal = reusable_goal
            tree.root.instruction = reusable_goal
            self.library.save(tree)
            self._borrowed_from[task.task_type] = entry.task_type
            return tree
        tree = ProgressiveSkillTree(task.task_type, reusable_goal)
        self.library.save(tree)
        return tree

    async def run_episode(
        self,
        task: Task,
        environment: Environment,
        *,
        episode_index: int | None = None,
    ) -> EpisodeResult:
        """Execute one episode and evolve its skill tree when a trigger fires."""
        tree = self.skill_tree_for(task)
        trajectory, edge_usage = await self.edge.run(task, environment, tree)
        self.library.record_use(
            tree.task_type,
            trajectory.success,
            observed_task_type=task.task_type,
        )
        result = EpisodeResult(trajectory=trajectory, token_usage=edge_usage)
        source_task_type = self._borrowed_from.get(task.task_type)
        if source_task_type:
            self.library.record_use(
                source_task_type,
                trajectory.success,
                observed_task_type=task.task_type,
            )
        if not self.config.enable_evolution:
            self._record_episode(tree, result, episode_index)
            return result
        decision = self.pool.add(trajectory)
        if not decision.triggered:
            self._record_episode(tree, result, episode_index)
            return result

        # Per-task locks preserve async uploads without racing tree revisions.
        lock = self._evolution_locks.setdefault(task.task_type, asyncio.Lock())
        async with lock:
            trajectories = self.pool.drain(task.task_type)
            if not trajectories:
                self._record_episode(tree, result, episode_index)
                return result
            trie = TrajectoryTrie.from_trajectories(task.task_type, trajectories)
            latest = self.library.get(task.task_type)
            tree = latest.tree if latest else tree
            evolution = await self.cloud.evolve(tree, trie, decision.reasons)
            self.library.save(tree)
            result.evolution_triggered = True
            result.evolution_summary = evolution.summary
            result.token_usage = result.token_usage + evolution.usage
        self._record_episode(tree, result, episode_index)
        return result

    async def flush(self, task_type: str, reason: str = "manual flush") -> str:
        """Force cloud evolution for all buffered trajectories of a task type."""
        trajectories = self.pool.drain(task_type)
        if not trajectories:
            return "No pending trajectories"
        trie = TrajectoryTrie.from_trajectories(task_type, trajectories)
        entry = self.library.get(task_type)
        if entry is None:
            discarded = self.library.get(task_type, include_discarded=True)
            if discarded is not None and discarded.state == "Discarded":
                return (
                    f"Skipped cloud evolution for discarded skill "
                    f"{task_type}"
                )
            raise KeyError(task_type)
        evolution = await self.cloud.evolve(entry.tree, trie, [reason])
        self.library.save(entry.tree)
        if self.recorder is not None:
            self.recorder.record_tree(entry.tree, event="flush")
        return evolution.summary

    def _record_episode(
        self,
        tree: ProgressiveSkillTree,
        result: EpisodeResult,
        episode_index: int | None,
    ) -> None:
        if self.recorder is not None:
            self.recorder.record(
                result.trajectory,
                tree,
                episode_index=episode_index,
                evolution_triggered=result.evolution_triggered,
                evolution_summary=result.evolution_summary,
            )
