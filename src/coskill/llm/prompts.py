"""Prompt templates for edge action selection and cloud skill maintenance."""

from typing import Sequence


EDGE_CLOSED_SYSTEM_PROMPT = """You select exactly one next environment action.
Use the episode goal to bind placeholders in the reusable skill rules. Prefer an
action that makes measurable progress. Do not repeat an action when recent history
shows it produced no state change. The environment decides when the episode ends.

Return exactly one JSON object: {"action_index": <zero-based integer>}.
The integer must identify an entry in the numbered action menu. Never return action
text, reasoning, done, Markdown, or additional fields."""


EDGE_OPEN_SYSTEM_PROMPT = """You select exactly one next environment action.
Use the episode goal to bind placeholders in the reusable skill rules. Prefer a
short action that makes measurable progress. Do not repeat an action when recent
history shows it produced no state change. The environment decides when the episode
ends.

Return exactly one JSON object: {"action": "one environment action"}.
Never return reasoning, done, Markdown, or additional fields."""


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


def edge_prompt(
    goal: str,
    observation: str,
    guidance: str,
    history: str,
    admissible_actions: Sequence[str] | None = None,
) -> str:
    """Build the edge prompt containing task context and any closed action menu."""
    action_menu = (
        "\n".join(f"{index}: {action}" for index, action in enumerate(admissible_actions))
        if admissible_actions is not None
        else "(open action space)"
    )
    response_schema = (
        '{"action_index": <integer>}'
        if admissible_actions is not None
        else '{"action": "one environment action"}'
    )
    return f"""Episode goal:
{goal}

Current observation:
{observation}

Recent actions and outcomes:
{history or '(none)'}

Reusable skill rules:
{guidance}

Numbered actions:
{action_menu}

Choose the single action that best advances the episode goal.
Return only {response_schema}."""


def cloud_prompt(tree: str, trie: str, trigger_reasons: list[str]) -> str:
    """Build the cloud prompt from the current tree, evidence trie, and triggers."""
    reasons = "\n".join(f"- {reason}" for reason in trigger_reasons)
    return f"""Update trigger:
{reasons}

Current progressive skill tree (node IDs are patch locations):
{tree}

Cloud-aware prefix-merged trajectory trie:
{trie}

Return only evidence-supported, minimal tree-maintenance operations."""
