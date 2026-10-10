"""CoSkill method: trie evidence and progressive skill evolution."""

from .cloud import (
    CloudSkillEvolver,
    CoSkillCloud,
    EvolutionResult,
    ObservationCodec,
    TrajectoryTrie,
    TrieNode,
)
from .config import CoSkillConfig
from .edge import CoSkillEdge, EdgeAgent, TrajectoryPool
from .library import LibraryEntry, SkillLibrary
from .pipeline import CoSkill
from .tree import ProgressiveSkillTree, SkillNode

__all__ = [
    "CloudSkillEvolver",
    "CoSkill",
    "CoSkillCloud",
    "CoSkillConfig",
    "CoSkillEdge",
    "EdgeAgent",
    "EvolutionResult",
    "LibraryEntry",
    "ObservationCodec",
    "ProgressiveSkillTree",
    "SkillLibrary",
    "SkillNode",
    "TrajectoryPool",
    "TrajectoryTrie",
    "TrieNode",
]
