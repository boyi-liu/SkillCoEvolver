"""Method-independent contracts for skill-evolution runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from ..utils.models import Task, TokenUsage, Trajectory


class SkillGuidance(Protocol):
    """Minimal skill view required by an edge executor."""

    def guidance(self, *, max_depth: int | None = None) -> str:
        """Render instructions that can be inserted into an edge prompt."""


@runtime_checkable
class SkillArtifact(SkillGuidance, Protocol):
    """Serializable skill shared by framework and method implementations."""

    task_type: str
    goal: str
    revision: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the skill for storage and artifact recording."""


@dataclass(slots=True)
class StepResult:
    observation: str
    reward: float = 0.0
    done: bool = False
    success: bool = False
    info: dict[str, Any] | None = None


class Environment(ABC):
    """Environment interface consumed by all edge runtimes."""

    @abstractmethod
    async def reset(self, task: Task) -> str:
        raise NotImplementedError

    @abstractmethod
    async def step(self, action: str) -> StepResult:
        raise NotImplementedError

    def state_key(self, observation: str) -> str:
        return observation

    def admissible_actions(self) -> Sequence[str] | None:
        """Return the current closed action set, when one is available."""

        return None


@dataclass(slots=True)
class TriggerConfig:
    """Thresholds controlling when buffered trajectories are uploaded."""

    recent_window: int = 16
    failure_rate_threshold: float = 0.60
    fixed_batch_size: int | None = None


@dataclass(slots=True)
class TriggerDecision:
    """Decision and evidence returned by collaboration scheduling."""

    triggered: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvolutionDraft:
    """A candidate skill revision produced before validation and persistence."""

    candidate: SkillArtifact
    summary: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationResult:
    """Decision made by a method's evolution guard."""

    accepted: bool
    reason: str = ""


class EdgeRuntime(Protocol):
    """Edge-side operators required by the shared control loop."""

    agent: Any
    pool: Any

    async def execute(
        self,
        task: Task,
        environment: Environment,
        skill: SkillArtifact,
    ) -> tuple[Trajectory, TokenUsage]: ...

    def schedule(self, trajectory: Trajectory) -> TriggerDecision: ...

    def drain(self, task_type: str) -> list[Trajectory]: ...


class TrajectoryScheduler(Protocol):
    """Trajectory buffering and collaboration-trigger contract."""

    config: TriggerConfig

    def add(self, trajectory: Trajectory) -> TriggerDecision: ...

    def drain(self, task_type: str) -> list[Trajectory]: ...

    def pending_count(self, task_type: str) -> int: ...


class CloudRuntime(Protocol):
    """Cloud-side operators required by the shared control loop."""

    def retrieve_skill(self, task: Task) -> SkillArtifact: ...

    def lookup_skill(self, task_type: str) -> SkillArtifact | None: ...

    def commit_skill(self, skill: SkillArtifact) -> None: ...

    def compile_evidence(
        self,
        task_type: str,
        trajectories: list[Trajectory],
    ) -> Any: ...

    async def evolve_skill(
        self,
        skill: SkillArtifact,
        evidence: Any,
        trigger_reasons: list[str],
        *,
        framework_name: str,
    ) -> EvolutionDraft: ...

    def validate_evolution(
        self,
        current: SkillArtifact,
        draft: EvolutionDraft,
    ) -> ValidationResult: ...


__all__ = [
    "CloudRuntime",
    "EdgeRuntime",
    "Environment",
    "EvolutionDraft",
    "SkillArtifact",
    "SkillGuidance",
    "StepResult",
    "TrajectoryScheduler",
    "TriggerConfig",
    "TriggerDecision",
    "ValidationResult",
]
