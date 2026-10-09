"""Reusable edge execution and trajectory-scheduling building blocks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Sequence

from ..llm import LLMClient, LLMError, Message
from ..utils.models import Task, TokenUsage, Trajectory, TrajectoryStep
from .contracts import (
    Environment,
    SkillGuidance,
    TrajectoryScheduler,
    TriggerConfig,
    TriggerDecision,
)
from .edge_prompts import (
    EDGE_CLOSED_SYSTEM_PROMPT,
    EDGE_OPEN_SYSTEM_PROMPT,
    edge_prompt,
)


class BaseEdgeExecutor(ABC):
    """Shared episode loop with method-specific action completion."""

    def __init__(
        self,
        client: LLMClient,
        *,
        max_steps: int = 40,
        action_retries: int = 2,
    ) -> None:
        if action_retries < 0:
            raise ValueError("action_retries must be non-negative")
        self.client = client
        self.max_steps = max_steps
        self.action_retries = action_retries

    async def run(
        self,
        task: Task,
        environment: Environment,
        skills: SkillGuidance,
    ) -> tuple[Trajectory, TokenUsage]:
        """Run an episode until the environment terminates or the budget ends."""

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

    @abstractmethod
    async def _complete_action(
        self,
        messages: list[Message],
        *,
        admissible_actions: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], Any]:
        """Return one method-specific action completion."""

        raise NotImplementedError

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


class BaseEdgeAgent(BaseEdgeExecutor):
    """Default single-attempt action selector."""

    async def _complete_action(
        self,
        messages: list[Message],
        *,
        admissible_actions: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], Any]:
        payload, response = await self.client.complete_json(messages)
        if not self._resolved_action(payload, admissible_actions):
            raise LLMError("Device model returned an invalid action")
        return payload, response


class FailureRateTrajectoryPool:
    """Reusable fixed-batch and recent-failure trajectory scheduler."""

    def __init__(self, config: TriggerConfig | None = None) -> None:
        self.config = config or TriggerConfig()
        if self.config.recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if self.config.fixed_batch_size is not None and self.config.fixed_batch_size <= 0:
            raise ValueError("fixed_batch_size must be positive")
        self._pending: dict[str, list[Trajectory]] = defaultdict(list)
        self._recent: dict[str, deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.config.recent_window)
        )

    def add(self, trajectory: Trajectory) -> TriggerDecision:
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
        return self._pending.pop(task_type, [])

    def pending_count(self, task_type: str) -> int:
        return len(self._pending.get(task_type, []))


class ComposedEdgeRuntime:
    """Compose an action executor with a trajectory scheduler."""

    def __init__(
        self,
        agent: BaseEdgeExecutor,
        pool: TrajectoryScheduler,
    ) -> None:
        self.agent = agent
        self.pool = pool
        self.client = agent.client
        self.max_steps = agent.max_steps
        self.action_retries = agent.action_retries
        self.config = pool.config

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
        skill: SkillGuidance,
    ) -> tuple[Trajectory, TokenUsage]:
        return await self.execute(task, environment, skill)

    def schedule(self, trajectory: Trajectory) -> TriggerDecision:
        return self.pool.add(trajectory)

    def add(self, trajectory: Trajectory) -> TriggerDecision:
        return self.pool.add(trajectory)

    def drain(self, task_type: str) -> list[Trajectory]:
        return self.pool.drain(task_type)

    def pending_count(self, task_type: str) -> int:
        return self.pool.pending_count(task_type)


class BaseEdgeRuntime(ComposedEdgeRuntime):
    """Default edge runtime, customizable through agent and pool class hooks."""

    agent_class: type[BaseEdgeExecutor] = BaseEdgeAgent
    pool_class: type[FailureRateTrajectoryPool] = FailureRateTrajectoryPool

    def __init__(
        self,
        client: LLMClient,
        *,
        max_steps: int = 40,
        action_retries: int = 2,
        trigger: TriggerConfig | None = None,
    ) -> None:
        super().__init__(
            self.agent_class(
                client,
                max_steps=max_steps,
                action_retries=action_retries,
            ),
            self.pool_class(trigger),
        )


__all__ = [
    "BaseEdgeAgent",
    "BaseEdgeExecutor",
    "BaseEdgeRuntime",
    "ComposedEdgeRuntime",
    "FailureRateTrajectoryPool",
]
