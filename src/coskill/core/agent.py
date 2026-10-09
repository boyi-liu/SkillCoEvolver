"""Environment protocols and the edge-side agent execution loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from ..llm import LLMClient, LLMError, Message
from ..llm.prompts import (
    EDGE_CLOSED_SYSTEM_PROMPT,
    EDGE_OPEN_SYSTEM_PROMPT,
    edge_prompt,
)
from ..skills import ProgressiveSkillTree
from .models import Task, TokenUsage, Trajectory, TrajectoryStep


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
        """Return the current closed action set, when the environment exposes one."""
        return None


class EdgeAgent:
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
        skills: ProgressiveSkillTree,
    ) -> tuple[Trajectory, TokenUsage]:
        """Run one task episode and return its trajectory and token usage."""
        observation = await environment.reset(task)
        steps: list[TrajectoryStep] = []
        usage = TokenUsage()
        success = False
        for _ in range(self.max_steps):
            history = self._history(steps)
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
                        history,
                        admissible_actions,
                    ),
                ),
            ]
            payload, response = await self._complete_action(
                messages,
                admissible_actions=admissible_actions,
            )
            usage = usage + response.usage
            action = self._resolved_action(payload, admissible_actions)
            if not action:
                raise LLMError("Device model returned an empty action before completion")
            state = environment.state_key(observation)
            outcome = await environment.step(action)
            steps.append(
                TrajectoryStep(
                    state=state,
                    action=action,
                    observation=outcome.observation,
                )
            )
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
        """Retry incomplete responses and invalid closed-set action selections."""
        total_usage = TokenUsage()
        last_payload: dict[str, Any] = {}
        for attempt in range(self.action_retries + 1):
            attempt_messages = messages
            if attempt:
                if admissible_actions is not None:
                    valid_labels = ", ".join(
                        str(index) for index in range(len(admissible_actions))
                    )
                    correction = (
                        f"Your previous response {last_payload!r} did not copy a valid "
                        f"menu label. The only valid labels are [{valid_labels}]. "
                        "Copy one integer printed before an action; do not count the "
                        "actions yourself. Return only "
                        '{"action_index": <one listed integer>}.'
                    )
                else:
                    correction = (
                        f"Your previous response {last_payload!r} did not contain exactly "
                        "one usable action. Return only "
                        '{"action": "one non-empty environment action"}.'
                    )
                attempt_messages = [
                    *messages,
                    Message("user", correction),
                ]
            payload, response = await self.client.complete_json(attempt_messages)
            last_payload = payload
            total_usage = total_usage + response.usage
            action = self._resolved_action(payload, admissible_actions)
            if action:
                response.usage = total_usage
                return payload, response
        raise LLMError(
            f"Device model returned an invalid action after "
            f"{self.action_retries} retries; payload={last_payload!r}; "
            f"admissible_count={len(admissible_actions) if admissible_actions is not None else 'n/a'}"
        )

    @staticmethod
    def _resolved_action(
        payload: dict[str, Any], admissible_actions: Sequence[str] | None
    ) -> str:
        """Validate a model payload and resolve it to an environment action."""
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
