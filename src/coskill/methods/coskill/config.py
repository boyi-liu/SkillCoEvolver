"""Configuration owned by the CoSkill method."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...framework import SkillEvolutionConfig


@dataclass(slots=True)
class CoSkillConfig(SkillEvolutionConfig):
    """CoSkill-specific hyperparameters.

    The final three fields are deprecated compatibility inputs. New callers
    should pass global run controls directly to :class:`CoSkill`.
    """

    trie_observation_limit: int = 300
    max_evolution_operations: int = 3
    max_steps: int | None = None
    enable_evolution: bool | None = None
    artifact_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.trie_observation_limit <= 0:
            raise ValueError("trie_observation_limit must be positive")
        if not 1 <= self.max_evolution_operations <= 3:
            raise ValueError("max_evolution_operations must be between 1 and 3")


__all__ = ["CoSkillConfig"]
