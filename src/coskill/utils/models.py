"""Serializable data models shared across edge, cloud, and storage layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class Task:
    """Description of a task episode and its reusable skill template."""

    task_type: str
    goal: str
    task_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)
    skill_template: str | None = None


@dataclass(slots=True)
class TrajectoryStep:
    """One state, action, and resulting observation from an episode."""

    state: str
    action: str
    observation: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the step into a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrajectoryStep":
        """Rebuild a trajectory step from serialized data."""
        return cls(**value)


@dataclass(slots=True)
class Trajectory:
    """Complete episode trace together with its success outcome."""

    task_id: str
    task_type: str
    goal: str
    steps: list[TrajectoryStep]
    success: bool
    final_observation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    trajectory_id: str = field(default_factory=lambda: uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trajectory, including all nested steps."""
        value = asdict(self)
        value["steps"] = [step.to_dict() for step in self.steps]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Trajectory":
        """Rebuild a trajectory and its steps from serialized data."""
        value = dict(value)
        value["steps"] = [TrajectoryStep.from_dict(step) for step in value["steps"]]
        return cls(**value)


@dataclass(slots=True)
class TokenUsage:
    """Prompt and completion token counters accumulated across calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass(slots=True)
class EpisodeResult:
    """Result returned after an episode, including optional cloud evolution."""

    trajectory: Trajectory
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    evolution_triggered: bool = False
    evolution_summary: str = ""
