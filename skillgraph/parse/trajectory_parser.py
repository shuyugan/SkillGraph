"""Parse raw trajectory.json files (ATIF-v1.2 format) into TrajectoryRecord objects.

Trajectory JSON layout:
  step_id=1  source=system   -> system prompt (ignored)
  step_id=2  source=user     -> task description
  step_id=3+ source=agent    -> agent turns (THOUGHT + ```bash``` + observation)

Each agent step contains:
  - message: full text response (THOUGHT + ```bash ...```)
  - reasoning_content: (step_id=3 only) initial reasoning
  - tool_calls: [{tool_call_id, function_name, arguments: {command}}]
  - observation: {results: [{content}]}
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from skillgraph.config import TRAJECTORIES_DIR
from skillgraph.parse.action_classifier import classify_action
from skillgraph.parse.models import TrajectoryRecord, TrajectoryStep

logger = logging.getLogger("skillgraph.parse")


def parse_all_trajectories(traj_dir: Path | None = None) -> list[TrajectoryRecord]:
    """Parse all trajectories from the given directory."""
    traj_dir = traj_dir or TRAJECTORIES_DIR
    records = []
    for subdir in sorted(traj_dir.iterdir()):
        if not subdir.is_dir() or "__" not in subdir.name:
            continue
        try:
            record = parse_single_trajectory(subdir)
            records.append(record)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", subdir.name, e)
    logger.info("Parsed %d trajectories", len(records))
    return records


def parse_single_trajectory(attempt_dir: Path) -> TrajectoryRecord:
    """Parse a single attempt directory into a TrajectoryRecord."""
    trajectory_path = attempt_dir / "agent" / "trajectory.json"
    reward_path = attempt_dir / "verifier" / "reward.txt"

    if not trajectory_path.exists():
        raise FileNotFoundError(f"No trajectory.json in {attempt_dir}/agent/")

    with open(trajectory_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Metadata
    trajectory_id = attempt_dir.name
    task_id = _infer_task_id(trajectory_id)

    # Resolved status
    resolved = False
    if reward_path.exists():
        resolved = reward_path.read_text(encoding="utf-8").strip() == "1"

    # Extract task description from step_id=2 (source=user)
    steps_raw = raw.get("steps", [])
    task_description = _extract_task_description(steps_raw)

    # Parse agent steps
    normalized_steps = []
    step_idx = 0

    for step in steps_raw:
        if step.get("source") != "agent":
            continue

        step_idx += 1
        thought = _extract_thought(step)
        action = _extract_action(step)
        observation = _extract_observation(step)
        action_type = classify_action(action)

        normalized_steps.append(TrajectoryStep(
            step_num=step_idx,
            thought=thought,
            action=action,
            action_type=action_type,
            observation=observation,
        ))

    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        task_id=task_id,
        task_description=task_description,
        resolved=resolved,
        steps=normalized_steps,
    )


# ── Internal helpers ──

def _infer_task_id(trajectory_id: str) -> str:
    """Extract task_id from directory name.

    e.g. "django__django-11119__6vgXR7E" -> "django__django-11119"
    """
    parts = trajectory_id.rsplit("__", 1)
    if len(parts) == 2:
        return parts[0]
    return trajectory_id


def _extract_task_description(steps: list[dict]) -> str:
    """Extract the task description from the user step (step_id=2)."""
    for step in steps:
        if step.get("step_id") == 2 and step.get("source") == "user":
            msg = step.get("message", "")
            # Strip boilerplate suffix (Recommended Workflow, Important Rules, etc.)
            match = re.search(
                r"Please solve this issue:\s*(.*?)(?=\n## Recommended Workflow|\n## Important Rules|\n## Formatting)",
                msg,
                re.DOTALL,
            )
            if match:
                return match.group(1).strip()
            return msg.strip()
    return ""


def _extract_thought(step: dict) -> str:
    """Extract the thought/reasoning from an agent step."""
    step_id = step.get("step_id", 0)

    # First agent step: prefer reasoning_content if available
    if step_id == 3 and step.get("reasoning_content"):
        return step["reasoning_content"].strip()

    message = step.get("message", "")

    # Extract text before the code block
    match = re.match(r"(.*?)```bash", message, re.DOTALL)
    if match:
        thought_text = match.group(1).strip()
        thought_text = re.sub(r"^THOUGHT:\s*", "", thought_text, flags=re.IGNORECASE)
        if thought_text:
            return thought_text

    # Fallback: return message minus code blocks
    cleaned = re.sub(r"```.*?```", "", message, flags=re.DOTALL).strip()
    return cleaned if cleaned else message.strip()


def _extract_action(step: dict) -> str:
    """Extract the bash command from an agent step.

    Tries tool_calls first, then falls back to parsing ```bash``` from message.
    """
    # Try structured tool_calls
    tool_calls = step.get("tool_calls", [])
    if tool_calls:
        commands = []
        for tc in tool_calls:
            args = tc.get("arguments", {})
            cmd = args.get("command", "")
            if cmd:
                commands.append(cmd)
        if commands:
            return " && ".join(commands)

    # Fallback: parse ```bash``` from message
    message = step.get("message", "")
    match = re.search(r"```bash\s*\n(.*?)\n```", message, re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""


def _extract_observation(step: dict) -> str:
    """Extract observation text from a step's observation results."""
    obs = step.get("observation", {})
    if isinstance(obs, str):
        return obs

    results = obs.get("results", [])
    parts = []
    for r in results:
        content = r.get("content", "")
        parts.append(content)
    return "\n".join(parts)
