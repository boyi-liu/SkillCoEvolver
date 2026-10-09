"""Default prompt templates for flat-skill cloud evolution."""


BASE_EVOLUTION_SYSTEM_PROMPT = """You maintain one reusable task instruction.
Given the current instruction and raw device trajectories, make the smallest
evidence-supported improvement. Do not copy episode-specific object names or IDs.
Return exactly one JSON object with this schema:
{"instruction": "one reusable instruction", "summary": "short explanation"}
If the evidence does not justify a change, return the current instruction.
"""


def base_cloud_prompt(
    framework_name: str,
    instruction: str,
    evidence: str,
    trigger_reasons: list[str],
) -> str:
    """Build the raw-trajectory default evolution prompt."""

    reasons = "\n".join(f"- {reason}" for reason in trigger_reasons)
    return f"""Framework: {framework_name}
Update trigger:
{reasons}

Current reusable instruction:
{instruction}

Raw trajectories:
{evidence}"""


__all__ = ["BASE_EVOLUTION_SYSTEM_PROMPT", "base_cloud_prompt"]
