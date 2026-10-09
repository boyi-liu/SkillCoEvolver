"""Cloud-side parsing and application of evidence-backed skill updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..llm import LLMClient, Message
from ..skills import ProgressiveSkillTree
from ..utils.models import TokenUsage
from .prompts import CLOUD_SYSTEM_PROMPT, cloud_prompt
from .trie import TrajectoryTrie


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

    def __init__(self, client: LLMClient) -> None:
        self.client = client

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
                        tree.guidance(), trie.to_markdown(), trigger_reasons
                    ),
                ),
            ]
        )
        operations = _object_list(payload.get("operations"), limit=3)
        for operation in operations:
            operation["children"] = _object_list(
                operation.get("children"), limit=3
            )
        applied = tree.apply_operations(operations)
        return EvolutionResult(
            summary=str(payload.get("summary", "Skill tree updated")),
            diagnoses=_object_list(payload.get("diagnoses"), limit=3),
            contrasts=_object_list(payload.get("contrasts"), limit=3),
            applied_operations=applied,
            usage=response.usage,
        )
