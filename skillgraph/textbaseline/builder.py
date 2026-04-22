"""Builder for the text-based markdown skill baseline."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from skillgraph.llm import call_llm, call_llm_json
from skillgraph.parse.models import TrajectoryRecord
from skillgraph.textbaseline.prompts import (
    DOMAIN_FILE_INIT_SYSTEM,
    DOMAIN_FILE_INIT_USER,
    DOMAIN_FILE_REFINE_SYSTEM,
    DOMAIN_FILE_REFINE_USER,
    DOMAIN_FILE_UPDATE_SYSTEM,
    DOMAIN_FILE_UPDATE_USER,
    TASK_MEMO_INIT_SYSTEM,
    TASK_MEMO_INIT_USER,
    TASK_MEMO_PITFALL_SYSTEM,
    TASK_MEMO_PITFALL_USER,
    TASK_MEMO_REFINE_SYSTEM,
    TASK_MEMO_REFINE_USER,
    TASK_MEMO_UPDATE_SYSTEM,
    TASK_MEMO_UPDATE_USER,
    TRAJECTORY_ANALYZER_SYSTEM,
    TRAJECTORY_ANALYZER_USER,
)

logger = logging.getLogger("skillgraph.textbaseline")


@dataclass
class TextBaselineTrace:
    """Trace of how task memos and domain files were built."""

    task_memos: dict[str, dict] = field(default_factory=dict)
    domains: dict[str, dict] = field(default_factory=dict)

    def record_task_memo(self, task_id: str, payload: dict) -> None:
        self.task_memos[task_id] = payload

    def record_domain(self, domain: str, payload: dict) -> None:
        self.domains[domain] = payload

    def to_dict(self) -> dict:
        return {
            "task_memos": self.task_memos,
            "domains": self.domains,
        }


def build_trajectory_text(steps: list) -> str:
    """Render parsed trajectory steps for prompt consumption."""
    lines = []
    for step in steps:
        thought = step.thought if step.thought else "(no thought)"
        action = step.action if step.action else "(no action)"
        obs = step.observation if step.observation else "(no output)"
        lines.append(
            f"### Step {step.step_num} [{step.action_type}]\n"
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Observation: {obs}\n"
        )
    return "\n".join(lines)


def analyze_trajectory(record: TrajectoryRecord) -> tuple[dict, float]:
    """Analyze one trajectory into structured reusable insights."""
    prompt = TRAJECTORY_ANALYZER_USER.format(
        task_description=record.task_description,
        trajectory_steps=build_trajectory_text(record.steps),
    )
    data, cost = call_llm_json(prompt, system=TRAJECTORY_ANALYZER_SYSTEM)
    if not isinstance(data, dict):
        logger.warning("Trajectory analyzer expected dict, got %s", type(data).__name__)
        data = {
            "trajectory_summary": "",
            "problem_signals": [],
            "localization_patterns": [],
            "fix_patterns": [],
            "verification_patterns": [],
            "pitfalls": [],
            "notable_snippets": [],
        }
    return data, cost


def initialize_task_memo(task_id: str, domain: str, trajectory_insights: dict) -> tuple[str, float]:
    """Create the initial task memo from the first successful trajectory insights."""
    prompt = TASK_MEMO_INIT_USER.format(
        task_id=task_id,
        domain=domain,
        trajectory_insights_json=json.dumps(trajectory_insights, indent=2, ensure_ascii=False),
    )
    response = call_llm(prompt, system=TASK_MEMO_INIT_SYSTEM)
    return _coerce_markdown_response(response.text.strip()), response.cost


def update_task_memo(current_memo: str, trajectory_insights: dict) -> tuple[str, float]:
    """Update an existing task memo with successful trajectory insights."""
    prompt = TASK_MEMO_UPDATE_USER.format(
        current_memo=current_memo,
        trajectory_insights_json=json.dumps(trajectory_insights, indent=2, ensure_ascii=False),
    )
    response = call_llm(prompt, system=TASK_MEMO_UPDATE_SYSTEM)
    return _coerce_markdown_response(response.text.strip(), fallback=current_memo), response.cost


def add_task_pitfalls(current_memo: str, trajectory_insights: dict) -> tuple[str, float]:
    """Update an existing task memo with failed-trajectory insights."""
    prompt = TASK_MEMO_PITFALL_USER.format(
        current_memo=current_memo,
        trajectory_insights_json=json.dumps(trajectory_insights, indent=2, ensure_ascii=False),
    )
    response = call_llm(prompt, system=TASK_MEMO_PITFALL_SYSTEM)
    return _coerce_markdown_response(response.text.strip(), fallback=current_memo), response.cost


def refine_task_memo(current_memo: str) -> tuple[str, float]:
    """Lightly refine a task memo."""
    prompt = TASK_MEMO_REFINE_USER.format(current_memo=current_memo)
    response = call_llm(prompt, system=TASK_MEMO_REFINE_SYSTEM)
    return _coerce_markdown_response(response.text.strip(), fallback=current_memo), response.cost


def initialize_domain_file(domain: str, task_memo: str) -> tuple[str, float]:
    """Create the initial domain markdown skill file from one task memo."""
    prompt = DOMAIN_FILE_INIT_USER.format(domain=domain, task_memo=task_memo)
    response = call_llm(prompt, system=DOMAIN_FILE_INIT_SYSTEM)
    return _coerce_markdown_response(response.text.strip()), response.cost


def update_domain_file(current_domain_markdown: str, task_memo: str) -> tuple[str, float]:
    """Update a domain markdown skill file with a new task memo."""
    prompt = DOMAIN_FILE_UPDATE_USER.format(
        current_domain_markdown=current_domain_markdown,
        task_memo=task_memo,
    )
    response = call_llm(prompt, system=DOMAIN_FILE_UPDATE_SYSTEM)
    return _coerce_markdown_response(response.text.strip(), fallback=current_domain_markdown), response.cost


def refine_domain_file(current_domain_markdown: str) -> tuple[str, float]:
    """Consolidate a domain markdown skill file."""
    prompt = DOMAIN_FILE_REFINE_USER.format(current_domain_markdown=current_domain_markdown)
    response = call_llm(prompt, system=DOMAIN_FILE_REFINE_SYSTEM)
    return _coerce_markdown_response(response.text.strip(), fallback=current_domain_markdown), response.cost


def build_task_memo(task_id: str, records: list[TrajectoryRecord]) -> tuple[str | None, dict]:
    """Build one task memo from all trajectories for the same task."""
    resolved = sorted((r for r in records if r.resolved), key=lambda r: r.trajectory_id)
    failed = sorted((r for r in records if not r.resolved), key=lambda r: r.trajectory_id)

    trace = {
        "task_id": task_id,
        "domain": task_id.split("__", 1)[0],
        "resolved_trajectories": [r.trajectory_id for r in resolved],
        "failed_trajectories": [r.trajectory_id for r in failed],
        "analyzer_cost": 0.0,
        "memo_cost": 0.0,
        "steps": [],
    }

    if not resolved:
        trace["status"] = "skipped_no_resolved"
        return None, trace

    domain = task_id.split("__", 1)[0]
    first_insights, cost = analyze_trajectory(resolved[0])
    trace["analyzer_cost"] += cost
    memo, cost = initialize_task_memo(task_id, domain, first_insights)
    trace["memo_cost"] += cost
    trace["steps"].append({
        "step": "initialize_task_memo",
        "trajectory_id": resolved[0].trajectory_id,
        "insights": first_insights,
    })

    for record in resolved[1:]:
        insights, cost = analyze_trajectory(record)
        trace["analyzer_cost"] += cost
        memo, cost = update_task_memo(memo, insights)
        trace["memo_cost"] += cost
        trace["steps"].append({
            "step": "update_task_memo",
            "trajectory_id": record.trajectory_id,
            "insights": insights,
        })

    for record in failed:
        insights, cost = analyze_trajectory(record)
        trace["analyzer_cost"] += cost
        memo, cost = add_task_pitfalls(memo, insights)
        trace["memo_cost"] += cost
        trace["steps"].append({
            "step": "add_task_pitfalls",
            "trajectory_id": record.trajectory_id,
            "insights": insights,
        })

    memo, cost = refine_task_memo(memo)
    trace["memo_cost"] += cost
    trace["steps"].append({"step": "refine_task_memo"})
    trace["status"] = "built"
    return memo, trace


def build_domain_markdown(domain: str, task_memos: list[tuple[str, str]]) -> tuple[str | None, dict]:
    """Build one repo/domain markdown baseline from task memos."""
    trace = {
        "domain": domain,
        "source_tasks": [task_id for task_id, _ in task_memos],
        "cost": 0.0,
        "steps": [],
    }

    if not task_memos:
        trace["status"] = "skipped_empty"
        return None, trace

    first_task_id, first_memo = task_memos[0]
    domain_md, cost = initialize_domain_file(domain, first_memo)
    trace["cost"] += cost
    trace["steps"].append({"step": "initialize_domain_file", "task_id": first_task_id})

    for task_id, memo in task_memos[1:]:
        domain_md, cost = update_domain_file(domain_md, memo)
        trace["cost"] += cost
        trace["steps"].append({"step": "update_domain_file", "task_id": task_id})

    domain_md, cost = refine_domain_file(domain_md)
    trace["cost"] += cost
    trace["steps"].append({"step": "refine_domain_file"})
    trace["status"] = "built"
    return domain_md, trace


def save_task_memo(output_root: Path, task_id: str, memo: str, trace: dict) -> None:
    """Persist one task memo and its trace."""
    task_dir = output_root / "task_memos" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "memo.md").write_text(memo, encoding="utf-8")
    (task_dir / "trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_domain_markdown(output_root: Path, domain: str, markdown: str, trace: dict) -> None:
    """Persist one domain markdown baseline and its trace."""
    domain_dir = output_root / "domains" / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    (domain_dir / f"{domain}.md").write_text(markdown, encoding="utf-8")
    (domain_dir / "trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _coerce_markdown_response(result: str, fallback: str | None = None) -> str:
    """Strip fences and keep a safe markdown fallback."""
    if result.startswith("```"):
        match = re.search(r"```(?:markdown)?\s*\n(.*?)\n```", result, re.DOTALL)
        if match:
            result = match.group(1).strip()

    if not result.strip():
        if fallback is not None:
            logger.warning("Text baseline executor returned empty markdown; keeping previous version")
            return fallback
        raise ValueError("Text baseline executor returned empty markdown")

    return result.strip() + "\n"
