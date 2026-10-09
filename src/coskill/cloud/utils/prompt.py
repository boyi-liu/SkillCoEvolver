"""Prompt templates shared by baseline and specialized cloud evolvers."""


NAIVE_EVOLUTION_SYSTEM_PROMPT = """You maintain one reusable task instruction.
Given the current instruction and raw device trajectories, make the smallest
evidence-supported improvement. Do not copy episode-specific object names or IDs.
Return exactly one JSON object with this schema:
{"instruction": "one reusable instruction", "summary": "short explanation"}
If the evidence does not justify a change, return the current instruction.
"""


CLOUD_SYSTEM_PROMPT = """You are the cloud-side skill maintainer in Co-Skill.
Analyze the prefix-merged trajectory trie with the current progressive skill tree.
The tree is shared across episodes of the same task type. Focus on task-level
decisions that the runtime cannot enforce. Do not add rules about JSON formatting,
action-index syntax, or other executor protocol details. Treat malformed actions as
executor errors rather than reusable task strategy.

Write general rules with placeholders such as {target}, {source}, {destination},
and {tool}. Never copy episode-specific names or numeric IDs into reusable rules
unless they are invariant across the task type.

Evidence rules:
- Diagnose only recurring failures supported by the trie. Return at most three.
- Cite an existing trie node key and its observed support count.
- Contrast success and failure only when both occur below a shared prefix.
- If no observed successful branch exists, return an empty contrasts list.
- Never invent an intended successful trajectory.
- If evidence does not justify a change, return an empty operations list.

Maintenance rules:
- Return at most three operations using only refine or expand.
- Prefer refine. Expand only when one parent has at least two distinct recurring
  failure causes, and add at most three children at one new level.
- Preserve behavior supported by successful trajectories.
- Each instruction must contain one actionable rule in at most 40 words.
- Use only existing skill node IDs as patch locations.

Return exactly one JSON object with this schema:
{
  "diagnoses": [{
    "root_cause": "...", "evidence_node_key": "existing trie key", "support": 2,
    "corrective_rule": "...", "patch_node_id": "existing id"
  }],
  "contrasts": [{
    "shared_prefix_node_key": "...", "successful_choice": "...",
    "failed_choice": "...", "lesson": "..."
  }],
  "operations": [{
    "operation": "refine|expand", "node_id": "existing id",
    "title": "optional revised title", "instruction": "optional instruction",
    "children": [{"title": "...", "instruction": "..."}]
  }],
  "summary": "short explanation"
}"""


def naive_cloud_prompt(
    framework_name: str,
    instruction: str,
    evidence: str,
    trigger_reasons: list[str],
) -> str:
    """Build the raw-trajectory baseline evolution prompt."""

    reasons = "\n".join(f"- {reason}" for reason in trigger_reasons)
    return f"""Framework: {framework_name}
Update trigger:
{reasons}

Current reusable instruction:
{instruction}

Raw trajectories:
{evidence}"""


def cloud_prompt(tree: str, trie: str, trigger_reasons: list[str]) -> str:
    """Build the trie-to-progressive-tree evolution prompt."""

    reasons = "\n".join(f"- {reason}" for reason in trigger_reasons)
    return f"""Update trigger:
{reasons}

Current progressive skill tree (node IDs are patch locations):
{tree}

Cloud-aware prefix-merged trajectory trie:
{trie}

Return only evidence-supported, minimal tree-maintenance operations."""
