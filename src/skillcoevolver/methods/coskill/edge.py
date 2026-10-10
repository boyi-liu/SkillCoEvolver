"""CoSkill edge composition: robust actions and adaptive upload triggers."""

from __future__ import annotations

from typing import Any, Sequence

from ...framework import (
    BaseEdgeAgent,
    BaseEdgeRuntime,
    FailureRateTrajectoryPool,
)
from ...llm import LLMError, Message
from ...utils.models import TokenUsage


class EdgeAgent(BaseEdgeAgent):
    """Add corrective retries to the shared edge execution template."""

    async def _complete_action(
        self,
        messages: list[Message],
        *,
        admissible_actions: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], Any]:
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
                attempt_messages = [*messages, Message("user", correction)]
            payload, response = await self.client.complete_json(attempt_messages)
            last_payload = payload
            total_usage = total_usage + response.usage
            if self._resolved_action(payload, admissible_actions):
                response.usage = total_usage
                return payload, response
        raise LLMError(
            "Device model returned an invalid action after "
            f"{self.action_retries} retries; payload={last_payload!r}; "
            f"admissible_count={len(admissible_actions) if admissible_actions is not None else 'n/a'}"
        )


class TrajectoryPool(FailureRateTrajectoryPool):
    """CoSkill name for the shared failure-aware trajectory scheduler."""


class CoSkillEdge(BaseEdgeRuntime):
    """CoSkill runtime composed from its robust agent and trajectory pool."""

    agent_class = EdgeAgent
    pool_class = TrajectoryPool


__all__ = ["CoSkillEdge", "EdgeAgent", "TrajectoryPool"]
