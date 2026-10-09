"""Baseline edge contracts, execution, and collaboration scheduling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from ..llm import LLMClient, LLMError, Message
from ..utils.models import Task, TokenUsage, Trajectory, TrajectoryStep
from .utils.prompt import (
    EDGE_CLOSED_SYSTEM_PROMPT,
    EDGE_OPEN_SYSTEM_PROMPT,
    edge_prompt,
)


class SkillGuidance(Protocol):
    """Minimal skill view required by an edge executor."""

    def guidance(self, *, max_depth: int | None = None) -> str:
        """Render instructions that can be inserted into an edge prompt."""


@dataclass(slots=True)
class StepResult:
    observation: str
    reward: float = 0.0
    done: bool = False
    success: bool = False
    info: dict[str, Any] | None = None


class Environment(ABC):
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


class BaseEdge:
    """Runnable baseline for edge execution and collaboration scheduling."""

    def __init__(
        self,
        client: LLMClient,
        *,
        max_steps: int = 40,
        action_retries: int = 2,
        trigger: TriggerConfig | None = None,
    ) -> None:
        if action_retries < 0:
            raise ValueError("action_retries must be non-negative")
        self.client = client
        self.max_steps = max_steps
        self.action_retries = action_retries
        self.config = trigger or TriggerConfig()
        if self.config.recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if self.config.fixed_batch_size is not None and self.config.fixed_batch_size <= 0:
            raise ValueError("fixed_batch_size must be positive")
        self._pending: dict[str, list[Trajectory]] = defaultdict(list)
        self._recent: dict[str, deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.config.recent_window)
        )
        # Compatibility handles; specialized runtimes replace either component.
        self.agent: Any = self
        self.pool: Any = self

    async def execute(
        self,
        task: Task,
        environment: Environment,
        skill: SkillGuidance,
    ) -> tuple[Trajectory, TokenUsage]:
        return await self.agent.run(task, environment, skill)

    async def run(
        self,
        task: Task,
        environment: Environment,
        skills: SkillGuidance,
    ) -> tuple[Trajectory, TokenUsage]:
        """Run an episode with a single-attempt baseline action selector."""

        observation = await environment.reset(task)
        steps: list[TrajectoryStep] = []
        usage = TokenUsage()
        success = False
        for _ in range(self.max_steps):
            admissible_actions = environment.admissible_actions()
            if admissible_actions is not None and not admissible_actions:
                raise LLMError("Environment exposed an empty admissible action set")
            system_prompt = (
                EDGE_CLOSED_SYSTEM_PROMPT
                if admissible_actions is not None
                else EDGE_OPEN_SYSTEM_PROMPT
            )
            messages = [
                Message("system", system_prompt),
                Message(
                    "user",
                    edge_prompt(
                        task.goal,
                        observation,
                        skills.guidance(),
                        self._history(steps),
                        admissible_actions,
                    ),
                ),
            ]
            payload, response = await self._complete_action(
                messages, admissible_actions=admissible_actions
            )
            usage = usage + response.usage
            action = self._resolved_action(payload, admissible_actions)
            if not action:
                raise LLMError("Device model returned an invalid action")
            state = environment.state_key(observation)
            outcome = await environment.step(action)
            steps.append(TrajectoryStep(state, action, outcome.observation))
            observation = outcome.observation
            if outcome.done:
                success = outcome.success
                break
        return (
            Trajectory(
                task_id=task.task_id,
                task_type=task.task_type,
                goal=task.goal,
                steps=steps,
                success=success,
                final_observation=observation,
                metadata=task.metadata,
            ),
            usage,
        )

    async def _complete_action(
        self,
        messages: list[Message],
        *,
        admissible_actions: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], Any]:
        """Request one action without specialized recovery behavior."""

        payload, response = await self.client.complete_json(messages)
        if not self._resolved_action(payload, admissible_actions):
            raise LLMError("Device model returned an invalid action")
        return payload, response

    def schedule(self, trajectory: Trajectory) -> TriggerDecision:
        return self.pool.add(trajectory)

    def add(self, trajectory: Trajectory) -> TriggerDecision:
        """Buffer experience and apply the elementary threshold trigger."""

        task_type = trajectory.task_type
        self._pending[task_type].append(trajectory)
        recent = self._recent[task_type]
        recent.append(trajectory.success)
        reasons: list[str] = []
        if (
            self.config.fixed_batch_size is not None
            and len(self._pending[task_type]) >= self.config.fixed_batch_size
        ):
            reasons.append(
                "fixed update batch reached: "
                f"{len(self._pending[task_type])} trajectories"
            )
            return TriggerDecision(True, reasons)
        if self.config.fixed_batch_size is not None:
            return TriggerDecision(False, reasons)
        if len(self._pending[task_type]) >= self.config.recent_window:
            failure_rate = 1.0 - sum(recent) / len(recent)
            if failure_rate >= self.config.failure_rate_threshold:
                reasons.append(
                    f"recent failure rate {failure_rate:.2f} >= "
                    f"{self.config.failure_rate_threshold:.2f}"
                )
        return TriggerDecision(bool(reasons), reasons)

    def drain(self, task_type: str) -> list[Trajectory]:
        if self.pool is not self:
            return self.pool.drain(task_type)
        return self._pending.pop(task_type, [])

    def pending_count(self, task_type: str) -> int:
        if self.pool is not self:
            return self.pool.pending_count(task_type)
        return len(self._pending.get(task_type, []))

    @staticmethod
    def _resolved_action(
        payload: dict[str, Any], admissible_actions: Sequence[str] | None
    ) -> str:
        if admissible_actions is None:
            if set(payload) != {"action"}:
                return ""
            raw_action = payload.get("action")
            return raw_action.strip() if isinstance(raw_action, str) else ""
        if set(payload) != {"action_index"}:
            return ""
        raw_index = payload.get("action_index")
        if (
            isinstance(raw_index, int)
            and not isinstance(raw_index, bool)
            and 0 <= raw_index < len(admissible_actions)
        ):
            return str(admissible_actions[raw_index])
        return ""

    @staticmethod
    def _history(steps: list[TrajectoryStep], limit: int = 6) -> str:
        return "\n".join(
            f"action={step.action!r} -> {step.observation[:240]}"
            for step in steps[-limit:]
        )


__all__ = [
    "BaseEdge",
    "Environment",
    "SkillGuidance",
    "StepResult",
    "TriggerConfig",
    "TriggerDecision",
]
