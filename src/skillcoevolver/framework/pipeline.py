"""Method-independent control loop for edge-cloud skill evolution.

One episode is organized into six ordered steps:

1. **[Cloud] Retrieve skill** -- load an existing skill for the task type or
   initialize one when the task is first observed.
2. **[Edge] Execute task** -- run the task in the environment, collect its
   trajectory and token usage, and initialize the episode result.
3. **[Edge -> Cloud] Buffer and trigger evolution** -- buffer the trajectory,
   decide whether the cloud branch should run, serialize evolution per task
   type, and collect a consistent batch. Disabled evolution or an untriggered
   decision ends the episode here.
4. **[Cloud] Compile and evolve** -- turn the trajectory batch into
   method-specific evidence and produce a candidate skill revision.
5. **[Cloud] Validate and commit** -- check the candidate and persist it only
   when the validation policy accepts it.
6. **[Edge + Cloud] Record and return** -- combine edge/cloud usage, save
   optional artifacts, and return the final
   :class:`~skillcoevolver.utils.EpisodeResult`.

Steps 1--2 and 6 form the normal execution path. Steps 3--5 are the optional
collaboration/evolution path. :meth:`SkillEvolutionFramework.flush` manually
enters step 3 at batch collection for trajectories already in the buffer.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm import LLMClient
from .cloud import BaseCloudRuntime
from .contracts import (
    CloudRuntime,
    EdgeRuntime,
    Environment,
    EvolutionDraft,
    SkillArtifact,
    TriggerConfig,
    TriggerDecision,
    ValidationResult,
)
from .edge import BaseEdgeRuntime
from ..utils import EpisodeResult, RunArtifactRecorder, Task, TokenUsage, Trajectory


@dataclass(slots=True)
class SkillEvolutionConfig:
    """Method-level controls shared by skill-evolution pipelines."""

    action_retries: int = 2
    trigger: TriggerConfig = field(default_factory=TriggerConfig)


class SkillEvolutionFramework(ABC):
    """Coordinate replaceable edge and cloud operators in one control loop.

    A method supplies edge execution, collaboration scheduling, skill storage,
    evidence compilation, evolution, and validation through the runtime
    contracts. The framework contains no concrete algorithm implementation.
    """

    def __init__(
        self,
        *,
        edge_runtime: EdgeRuntime,
        cloud_runtime: CloudRuntime,
        config: SkillEvolutionConfig | None = None,
        max_steps: int | None = None,
        enable_evolution: bool | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.config = config or SkillEvolutionConfig()
        # Global run controls are explicit constructor inputs.  The getattr
        # fallback temporarily supports older combined CoSkillConfig objects.
        legacy_max_steps = getattr(self.config, "max_steps", None)
        legacy_enable_evolution = getattr(self.config, "enable_evolution", None)
        legacy_artifact_dir = getattr(self.config, "artifact_dir", None)
        self.max_steps = max_steps if max_steps is not None else legacy_max_steps or 40
        self.enable_evolution = (
            enable_evolution
            if enable_evolution is not None
            else (
                legacy_enable_evolution
                if legacy_enable_evolution is not None
                else True
            )
        )
        self.artifact_dir = (
            artifact_dir if artifact_dir is not None else legacy_artifact_dir
        )
        self.edge_runtime = edge_runtime
        self.cloud_runtime = cloud_runtime

        # Keep the historical attributes available to downstream callers.
        self.edge = self.edge_runtime.agent
        self.pool = self.edge_runtime.pool
        self.cloud_client = getattr(cloud_runtime, "client", None)
        self.recorder = (
            RunArtifactRecorder(self.artifact_dir)
            if self.artifact_dir is not None
            else None
        )
        self._evolution_locks: dict[str, asyncio.Lock] = {}

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Stable name of the concrete framework implementation."""

        raise NotImplementedError

    # Thin delegates expose the six operators as framework extension points.
    def retrieve_skill(self, task: Task) -> SkillArtifact:
        return self.cloud_runtime.retrieve_skill(task)

    def lookup_skill(self, task_type: str) -> SkillArtifact | None:
        return self.cloud_runtime.lookup_skill(task_type)

    def commit_skill(self, skill: SkillArtifact) -> None:
        self.cloud_runtime.commit_skill(skill)

    async def execute_at_edge(
        self,
        task: Task,
        environment: Environment,
        skill: SkillArtifact,
    ) -> tuple[Trajectory, TokenUsage]:
        return await self.edge_runtime.execute(task, environment, skill)

    def schedule_collaboration(self, trajectory: Trajectory) -> TriggerDecision:
        return self.edge_runtime.schedule(trajectory)

    def compile_evidence(
        self, task_type: str, trajectories: list[Trajectory]
    ) -> Any:
        return self.cloud_runtime.compile_evidence(task_type, trajectories)

    async def evolve_skill(
        self,
        skill: SkillArtifact,
        evidence: Any,
        trigger_reasons: list[str],
    ) -> EvolutionDraft:
        return await self.cloud_runtime.evolve_skill(
            skill,
            evidence,
            trigger_reasons,
            framework_name=self.framework_name,
        )

    def validate_evolution(
        self,
        current: SkillArtifact,
        draft: EvolutionDraft,
    ) -> ValidationResult:
        return self.cloud_runtime.validate_evolution(current, draft)

    async def run_episode(
        self,
        task: Task,
        environment: Environment,
        *,
        episode_index: int | None = None,
    ) -> EpisodeResult:
        """Run the six-step episode pipeline documented at module level."""

        # Step 1 [Cloud]: retrieve or initialize the task's reusable skill.
        skill = self.retrieve_skill(task)

        # Step 2 [Edge]: execute the task and initialize its episode result.
        trajectory, edge_usage = await self.execute_at_edge(task, environment, skill)
        result = EpisodeResult(trajectory=trajectory, token_usage=edge_usage)

        # Step 3 [Edge -> Cloud]: buffer the trajectory and, when triggered,
        # collect a consistent batch for cloud evolution.
        if not self.enable_evolution:
            self._record_episode(skill, result, episode_index)
            return result

        decision = self.schedule_collaboration(trajectory)
        if not decision.triggered:
            self._record_episode(skill, result, episode_index)
            return result

        lock = self._evolution_locks.setdefault(task.task_type, asyncio.Lock())
        async with lock:
            trajectories = self.edge_runtime.drain(task.task_type)
            if not trajectories:
                self._record_episode(skill, result, episode_index)
                return result
            current = self.lookup_skill(task.task_type) or skill

            # Step 4 [Cloud]: compile evidence and evolve a candidate skill.
            evidence = self.compile_evidence(task.task_type, trajectories)
            draft = await self.evolve_skill(current, evidence, decision.reasons)

            # Step 5 [Cloud]: validate the candidate before committing it.
            validation = self.validate_evolution(current, draft)
            committed = current
            if validation.accepted:
                self.commit_skill(draft.candidate)
                committed = draft.candidate
            result.evolution_triggered = True
            result.evolution_summary = self._evolution_summary(draft, validation)
            result.token_usage = result.token_usage + draft.usage

        # Step 6 [Edge + Cloud]: record artifacts and return the combined result.
        self._record_episode(committed, result, episode_index)
        return result

    async def flush(self, task_type: str, reason: str = "manual flush") -> str:
        """Force the shared evolution loop for buffered trajectories."""

        trajectories = self.edge_runtime.drain(task_type)
        if not trajectories:
            return "No pending trajectories"
        current = self.lookup_skill(task_type)
        if current is None:
            raise KeyError(task_type)
        evidence = self.compile_evidence(task_type, trajectories)
        draft = await self.evolve_skill(current, evidence, [reason])
        validation = self.validate_evolution(current, draft)
        committed = current
        if validation.accepted:
            self.commit_skill(draft.candidate)
            committed = draft.candidate
        if self.recorder is not None:
            self.recorder.record_tree(committed, event="flush")
        return self._evolution_summary(draft, validation)

    def _record_episode(
        self,
        skill: SkillArtifact,
        result: EpisodeResult,
        episode_index: int | None,
    ) -> None:
        if self.recorder is not None:
            self.recorder.record(
                result.trajectory,
                skill,
                episode_index=episode_index,
                evolution_triggered=result.evolution_triggered,
                evolution_summary=result.evolution_summary,
            )

    @staticmethod
    def _evolution_summary(
        draft: EvolutionDraft, validation: ValidationResult
    ) -> str:
        if validation.accepted:
            return draft.summary
        return f"{draft.summary} [rejected: {validation.reason}]"


class BaseSkillEvolutionFramework(SkillEvolutionFramework):
    """Runnable default framework and base composition for specialized methods."""

    def __init__(
        self,
        *,
        device_client: LLMClient | None = None,
        cloud_client: LLMClient | None = None,
        config: SkillEvolutionConfig | None = None,
        edge_runtime: EdgeRuntime | None = None,
        cloud_runtime: CloudRuntime | None = None,
        max_steps: int | None = None,
        enable_evolution: bool | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        effective_config = config or SkillEvolutionConfig()
        resolved_max_steps = max_steps if max_steps is not None else 40
        if edge_runtime is None:
            if device_client is None:
                raise TypeError("device_client is required without edge_runtime")
            edge_runtime = BaseEdgeRuntime(
                device_client,
                max_steps=resolved_max_steps,
                action_retries=effective_config.action_retries,
                trigger=effective_config.trigger,
            )
        if cloud_runtime is None:
            if cloud_client is None:
                raise TypeError("cloud_client is required without cloud_runtime")
            cloud_runtime = BaseCloudRuntime(cloud_client)
        super().__init__(
            edge_runtime=edge_runtime,
            cloud_runtime=cloud_runtime,
            config=effective_config,
            max_steps=resolved_max_steps,
            enable_evolution=enable_evolution,
            artifact_dir=artifact_dir,
        )

    @property
    def framework_name(self) -> str:
        return "base-edge-cloud-skill-evolution"


__all__ = [
    "BaseSkillEvolutionFramework",
    "EvolutionDraft",
    "SkillArtifact",
    "SkillEvolutionConfig",
    "SkillEvolutionFramework",
    "ValidationResult",
]
