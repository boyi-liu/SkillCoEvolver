"""Method-independent prompts for environment action selection."""

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


__all__ = ["EDGE_CLOSED_SYSTEM_PROMPT", "EDGE_OPEN_SYSTEM_PROMPT", "edge_prompt"]
