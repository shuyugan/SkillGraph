"""Graph builders and post-processors for trajectory-local and task-local graphs."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from skillgraph.graph.prompts import (
    INITIALIZER_SYSTEM,
    INITIALIZER_USER,
    SUCCESS_UPDATER_SYSTEM,
    SUCCESS_UPDATER_USER,
    FAILURE_UPDATER_SYSTEM,
    FAILURE_UPDATER_USER,
    TASK_RECONCILE_SYSTEM,
    TASK_RECONCILE_USER,
    EXECUTOR_UPDATE_SYSTEM,
    EXECUTOR_UPDATE_USER,
    EXECUTOR_PITFALL_SYSTEM,
    EXECUTOR_PITFALL_USER,
    EXECUTOR_REFINE_SYSTEM,
    EXECUTOR_REFINE_USER,
    LINK_EMBED_SYSTEM,
    LINK_EMBED_USER,
    assemble_node_markdown,
)
from skillgraph.llm import call_llm, call_llm_json
from skillgraph.models import SkillGraph
from skillgraph.parse.models import TrajectoryRecord

logger = logging.getLogger("skillgraph.graph.builder")

_OBSERVATION_TRUNCATION_LIMIT = 200
_CONTEXT_LENGTH_ERROR_MARKERS = (
    "context_length_exceeded",
    "Input tokens exceed the configured limit",
    "maximum context length",
    "prompt is too long",
)


# ═══════════════════════════════════════════════════════════════════════
# Trace
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BuildTrace:
    """Records every step of the accumulative build process."""

    steps: list[dict] = field(default_factory=list)
    total_action_counts: Counter[str] = field(default_factory=Counter)
    step_type_action_counts: dict[str, Counter[str]] = field(default_factory=dict)

    def record(self, step_type: str, trajectory_id: str, llm_output: dict | list,
               graph: SkillGraph) -> None:
        action_counts = _summarize_step_actions(step_type, llm_output)
        self.steps.append({
            "step": step_type,
            "trajectory_id": trajectory_id,
            "llm_output": llm_output,
            "action_counts": dict(sorted(action_counts.items())),
                "graph_snapshot": {
                    "n_nodes": len(graph.nodes),
                    "n_edges": len(graph.edges),
                    "node_ids": graph.node_ids(),
                    "edges": [
                        {"source": e.source, "target": e.target, "condition": e.condition}
                        for e in graph.edges
                    ],
                },
        })
        self.total_action_counts.update(action_counts)
        self.step_type_action_counts.setdefault(step_type, Counter()).update(action_counts)

    def operation_summary(self) -> dict:
        """Return aggregated action counts across the full build trace."""
        return {
            "total": dict(sorted(self.total_action_counts.items())),
            "by_step_type": {
                step_type: dict(sorted(counter.items()))
                for step_type, counter in sorted(self.step_type_action_counts.items())
            },
        }

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "operation_summary": self.operation_summary(),
        }


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def build_trajectory_text(steps: list, *, observation_limit: int | None = None) -> str:
    """Render parsed trajectory steps as text for LLM prompts.

    When ``observation_limit`` is set, long observations are trimmed to keep
    a single oversized shell/test output from blowing up the whole request.
    """
    lines = []
    for step in steps:
        thought = step.thought if step.thought else "(no thought)"
        action = step.action if step.action else "(no action)"
        obs = step.observation if step.observation else "(no output)"
        if observation_limit is not None and len(obs) > observation_limit:
            obs = obs[:observation_limit].rstrip() + " ... [observation truncated]"
        lines.append(
            f"### Step {step.step_num} [{step.action_type}]\n"
            f"**Thought:** {thought}\n"
            f"**Action:** {action}\n"
            f"**Observation:** {obs}\n"
        )
    return "\n".join(lines)


def _is_context_length_exceeded(exc: Exception) -> bool:
    """Return whether an exception looks like a model context-length failure."""
    message = str(exc)
    return any(marker in message for marker in _CONTEXT_LENGTH_ERROR_MARKERS)


def _call_llm_json_with_observation_fallback(
    prompt_factory,
    *,
    system: str,
    record: TrajectoryRecord,
    stage: str,
) -> tuple[dict | list, float]:
    """Call the model once normally, then retry with observation truncation if needed."""
    try:
        prompt = prompt_factory(build_trajectory_text(record.steps))
        return call_llm_json(prompt, system=system)
    except Exception as exc:
        if not _is_context_length_exceeded(exc):
            raise

        logger.warning(
            "%s: context_length_exceeded for %s; retrying with observation truncation (%d chars)",
            stage,
            record.trajectory_id,
            _OBSERVATION_TRUNCATION_LIMIT,
        )
        prompt = prompt_factory(
            build_trajectory_text(
                record.steps,
                observation_limit=_OBSERVATION_TRUNCATION_LIMIT,
            )
        )
        return call_llm_json(prompt, system=system)


def graph_snapshot(graph: SkillGraph) -> dict:
    """Return a compact serializable snapshot of the graph."""
    return {
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "node_ids": graph.node_ids(),
        "edges": [
            {"source": e.source, "target": e.target, "condition": e.condition}
            for e in graph.edges
        ],
    }


def _summarize_step_actions(step_type: str, llm_output: dict | list) -> Counter[str]:
    """Summarize structured actions for one build step."""
    counts: Counter[str] = Counter()

    if step_type == "initializer" and isinstance(llm_output, dict):
        counts["add_node"] += len(llm_output.get("nodes", []))
        counts["add_edge"] += len(llm_output.get("edges", []))
        return counts

    if step_type == "success_updater" and isinstance(llm_output, dict):
        for op in llm_output.get("operations", []):
            op_type = op.get("op")
            if op_type:
                counts[op_type] += 1
        for report in llm_output.get("post_refine", []):
            counts["post_refine_checked"] += 1
            status = report.get("status")
            if status:
                counts[f"post_refine_status_{status}"] += 1
            if report.get("refined"):
                counts["post_refine_refined"] += 1
        return counts

    if step_type == "failure_updater" and isinstance(llm_output, dict):
        pitfalls = llm_output.get("pitfalls", [])
        counts["add_pitfall"] += len(pitfalls)
        counts["failure_nodes_touched"] += len({
            pitfall.get("node_id") for pitfall in pitfalls if pitfall.get("node_id")
        })
        return counts

    if step_type == "task_reconciler" and isinstance(llm_output, dict):
        counts["update_node_card"] += len(llm_output.get("node_cards", []))
        for op in llm_output.get("edge_operations", []):
            op_type = op.get("op")
            if op_type:
                counts[op_type] += 1
        return counts

    if step_type == "granularity_normalizer" and isinstance(llm_output, dict):
        reports = llm_output.get("nodes", [])
        counts["normalize_checked"] += len(reports)
        for report in reports:
            status = report.get("status")
            if status:
                counts[f"normalize_status_{status}"] += 1
            if report.get("refined"):
                counts["normalize_refined"] += 1
        return counts

    if step_type == "link_embedding" and isinstance(llm_output, dict):
        counts["embed_links"] += len(llm_output.get("nodes_updated", []))
        return counts

    return counts


def _extract_section(markdown: str, heading: str | tuple[str, ...] | list[str]) -> str:
    """Extract a markdown section body by heading or heading aliases."""
    headings = [heading] if isinstance(heading, str) else list(heading)
    for name in headings:
        pattern = rf"## {re.escape(name)}\n\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, markdown, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _count_action_bullets(markdown: str) -> int:
    """Count action bullets in the Actions section."""
    actions = _extract_section(markdown, ("Actions", "Execution", "Procedure"))
    return sum(1 for line in actions.splitlines() if line.lstrip().startswith("- "))


def _strip_fenced_code_blocks(markdown: str) -> str:
    """Remove fenced code blocks so granularity checks do not overcount code templates."""
    return re.sub(r"```[^\n]*\n.*?\n```", "", markdown, flags=re.DOTALL)


def _count_fenced_code_lines(markdown: str) -> int:
    """Count non-empty lines inside fenced code blocks."""
    total = 0
    for match in re.finditer(r"```[^\n]*\n(.*?)\n```", markdown, flags=re.DOTALL):
        total += sum(1 for line in match.group(1).splitlines() if line.strip())
    return total


def _assess_node_granularity(markdown: str) -> dict:
    """Heuristic assessment of whether a node is too thin or too fat."""
    prose_only = _strip_fenced_code_blocks(markdown)
    non_empty_lines = sum(1 for line in prose_only.splitlines() if line.strip())
    action_bullets = _count_action_bullets(markdown)
    inline_code_spans = prose_only.count("`") // 2
    code_block_lines = _count_fenced_code_lines(markdown)

    fat_reasons: list[str] = []
    if non_empty_lines > 44:
        fat_reasons.append("long_markdown")
    if action_bullets > 6:
        fat_reasons.append("too_many_actions")
    if inline_code_spans > 8:
        fat_reasons.append("too_many_command_literals")
    if code_block_lines > 12:
        fat_reasons.append("large_embedded_code")

    if fat_reasons:
        status = "fat"
    elif non_empty_lines < 24 or action_bullets <= 1:
        status = "thin"
    else:
        status = "ok"

    return {
        "status": status,
        "non_empty_lines": non_empty_lines,
        "action_bullets": action_bullets,
        "inline_code_spans": inline_code_spans,
        "code_block_lines": code_block_lines,
        "reasons": fat_reasons,
    }


def _generalize_free_text(text: str, task_id: str | None = None) -> str:
    """Lightly scrub obviously task-specific leakage from free-form text."""
    cleaned = text.strip()
    cleaned = re.sub(r"`?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+`?", "<repo_path>", cleaned)
    cleaned = re.sub(
        r"\b(?:django|sympy|react|flask|pandas|numpy|scikit-learn|sklearn)\b",
        "<project>",
        cleaned,
        flags=re.IGNORECASE,
    )

    if task_id:
        repo_name = task_id.split("__", 1)[0]
        variants = {
            repo_name,
            repo_name.replace("-", ""),
            repo_name.replace("-", " "),
        }
        for variant in sorted((v for v in variants if v), key=len, reverse=True):
            cleaned = re.sub(re.escape(variant), "<project>", cleaned, flags=re.IGNORECASE)

    return cleaned


def _split_frontmatter(markdown: str) -> tuple[dict, str]:
    """Parse the simple YAML frontmatter used by skill nodes."""
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", markdown, flags=re.DOTALL)
    if not match:
        return {}, markdown.strip()

    frontmatter = match.group(1)
    body = match.group(2).lstrip("\n")
    fields: dict[str, object] = {}
    lines = frontmatter.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if stripped == "triggers:":
            triggers: list[str] = []
            i += 1
            while i < len(lines):
                item = lines[i].strip()
                if item.startswith("- "):
                    triggers.append(_unquote_yaml_scalar(item[2:].strip()))
                    i += 1
                    continue
                break
            fields["triggers"] = triggers
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = _unquote_yaml_scalar(value.strip())
        i += 1

    return fields, body.strip()


def _unquote_yaml_scalar(value: str) -> str:
    """Unquote a simple YAML scalar written as a single line string."""
    if len(value) >= 2 and value[0] == value[-1] == '"':
        inner = value[1:-1]
        return inner.replace('\\"', '"').replace("\\\\", "\\")
    return value


def _yaml_quote(value: str) -> str:
    """Quote a single-line YAML scalar conservatively."""
    value = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'


def _extract_first_section_line(markdown: str, headings: tuple[str, ...]) -> str:
    """Extract the first non-empty line from one of several markdown headings."""
    for heading in headings:
        pattern = rf"## {re.escape(heading)}\n\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, markdown, flags=re.DOTALL)
        if not match:
            continue
        for line in match.group(1).splitlines():
            line = line.strip()
            if line:
                return re.sub(r"\s+", " ", line)
    return ""


def _rewrite_frontmatter(markdown: str, *, fallback_node_id: str | None = None, updates: dict | None = None) -> str:
    """Rewrite node frontmatter while preserving the body."""
    current_fields, body = _split_frontmatter(markdown)
    updates = updates or {}
    merged = dict(current_fields)
    for key, value in updates.items():
        if value is None:
            continue
        merged[key] = value

    node_id = str(merged.get("name") or fallback_node_id or "unnamed")
    description = str(merged.get("description") or "").strip()
    summary = str(merged.get("summary") or description or "").strip()
    triggers = merged.get("triggers") or []
    if not isinstance(triggers, list):
        triggers = []
    triggers = [str(item).strip() for item in triggers if str(item).strip()]
    if not description:
        description = summary
    if not summary:
        summary = description or _extract_first_section_line(markdown, ("When to Use", "Situation", "Strategy", "Approach"))

    frontmatter_lines = [
        "---",
        f"name: {node_id}",
        f"summary: {_yaml_quote(summary)}",
        f"description: {_yaml_quote(description)}",
        "triggers:",
    ]
    if triggers:
        frontmatter_lines.extend(f"  - {_yaml_quote(trigger)}" for trigger in triggers)
    else:
        frontmatter_lines.append('  - "(none)"')
    frontmatter_lines.append("---")

    body = body.strip()
    return "\n".join(frontmatter_lines) + ("\n\n" + body if body else "") + "\n"


def _ensure_navigation_card(markdown: str, *, fallback_markdown: str | None = None, fallback_node_id: str | None = None) -> str:
    """Ensure a node carries the scan-card fields needed for navigation."""
    current_fields, _ = _split_frontmatter(markdown)
    fallback_fields, _ = _split_frontmatter(fallback_markdown or "")

    updates = {
        "name": current_fields.get("name") or fallback_fields.get("name") or fallback_node_id,
        "summary": current_fields.get("summary") or fallback_fields.get("summary") or current_fields.get("description") or fallback_fields.get("description"),
        "description": current_fields.get("description") or fallback_fields.get("description") or current_fields.get("summary") or fallback_fields.get("summary"),
        "triggers": current_fields.get("triggers") or fallback_fields.get("triggers") or [],
    }
    return _rewrite_frontmatter(markdown, fallback_node_id=fallback_node_id, updates=updates)


def _apply_node_card(markdown: str, *, node_id: str, card: dict, fallback_markdown: str | None = None) -> str:
    """Apply a structured scan-card update while preserving the body."""
    updates = {
        "name": node_id,
        "summary": (card.get("summary") or "").strip() or None,
        "description": (card.get("description") or "").strip() or None,
        "triggers": [str(t).strip() for t in (card.get("triggers") or []) if str(t).strip()] or None,
    }
    rewritten = _rewrite_frontmatter(markdown, fallback_node_id=node_id, updates=updates)
    return _ensure_navigation_card(rewritten, fallback_markdown=fallback_markdown, fallback_node_id=node_id)


# ═══════════════════════════════════════════════════════════════════════
# Initializer
# ═══════════════════════════════════════════════════════════════════════


def initialize_graph(record: TrajectoryRecord, trace: BuildTrace) -> SkillGraph:
    """Create initial graph from the first resolved trajectory."""
    assert record.resolved, "Initializer requires a resolved trajectory"

    data, cost = _call_llm_json_with_observation_fallback(
        lambda steps_text: INITIALIZER_USER.format(
            task_description=record.task_description,
            trajectory_steps=steps_text,
        ),
        system=INITIALIZER_SYSTEM,
        record=record,
        stage="Initializer",
    )

    if not isinstance(data, dict):
        logger.warning("Initializer: expected dict, got %s", type(data).__name__)
        return SkillGraph()

    graph = SkillGraph()

    # Create nodes (JSON → markdown via assembler)
    for node_data in data.get("nodes", []):
        node_id = node_data.get("node_id", "")
        if not node_id:
            continue
        md = _ensure_navigation_card(
            assemble_node_markdown(node_data),
            fallback_node_id=node_id,
        )
        graph.add_node(node_id, md)
        logger.info("  Initialized node: %s", node_id)

    # Create edges
    valid_ids = set(graph.node_ids())
    for edge_data in data.get("edges", []):
        src = edge_data.get("source", "")
        tgt = edge_data.get("target", "")
        if src in valid_ids and tgt in valid_ids and src != tgt:
            graph.add_edge(src, tgt, edge_data.get("condition", ""))
            logger.info("  Initialized edge: %s → %s", src, tgt)

    trace.record("initializer", record.trajectory_id, data, graph)
    logger.info("Initializer done: %d nodes, %d edges (cost=$%.4f)", len(graph.nodes), len(graph.edges), cost)
    return graph


# ═══════════════════════════════════════════════════════════════════════
# Success Updater
# ═══════════════════════════════════════════════════════════════════════


def success_update(graph: SkillGraph, record: TrajectoryRecord, trace: BuildTrace) -> SkillGraph:
    """Update graph with a resolved trajectory."""
    assert record.resolved, "Success updater requires a resolved trajectory"

    data, cost = _call_llm_json_with_observation_fallback(
        lambda steps_text: SUCCESS_UPDATER_USER.format(
            graph_content=graph.summary(),
            task_description=record.task_description,
            trajectory_steps=steps_text,
        ),
        system=SUCCESS_UPDATER_SYSTEM,
        record=record,
        stage="Success updater",
    )

    if not isinstance(data, dict):
        logger.warning("Success updater: expected dict, got %s", type(data).__name__)
        return graph

    operations = data.get("operations", [])
    logger.info("Success updater: %d operations (cost=$%.4f)", len(operations), cost)
    touched_node_ids: set[str] = set()

    for op in operations:
        op_type = op.get("op", "")

        if op_type == "update_node_card":
            node_id = op.get("node_id", "")
            card = op.get("card", {})
            if node_id and node_id in graph.nodes and isinstance(card, dict):
                current_md = graph.get_node(node_id) or ""
                updated_md = _apply_node_card(current_md, node_id=node_id, card=card, fallback_markdown=current_md)
                graph.update_node(node_id, updated_md)
                touched_node_ids.add(node_id)
                logger.info("  Updated node card: %s", node_id)

        elif op_type == "update_node":
            node_id = op.get("node_id", "")
            new_info = op.get("new_info", "")
            if node_id and new_info and node_id in graph.nodes:
                current_md = graph.get_node(node_id) or ""
                updated_md = _executor_update_node(current_md, new_info)
                updated_md = _ensure_navigation_card(updated_md, fallback_markdown=current_md, fallback_node_id=node_id)
                graph.update_node(node_id, updated_md)
                touched_node_ids.add(node_id)
                logger.info("  Updated node: %s", node_id)

        elif op_type == "add_node":
            node_data = op.get("node", {})
            node_id = node_data.get("node_id", "")
            if node_id and node_id not in graph.nodes:
                md = _ensure_navigation_card(
                    assemble_node_markdown(node_data),
                    fallback_node_id=node_id,
                )
                graph.add_node(node_id, md)
                touched_node_ids.add(node_id)
                logger.info("  Added node: %s", node_id)

        elif op_type == "split_node":
            old_id = op.get("node_id", "")
            new_nodes = op.get("new_nodes", [])
            if old_id and old_id in graph.nodes and len(new_nodes) == 2:
                # Remove old node
                old_md = graph.nodes.pop(old_id)
                logger.info("  Split node: %s →", old_id)

                # Add two new nodes
                new_ids = []
                for nd in new_nodes:
                    nid = nd.get("node_id", "")
                    if nid:
                        md = _ensure_navigation_card(
                            assemble_node_markdown(nd),
                            fallback_node_id=nid,
                        )
                        graph.add_node(nid, md)
                        new_ids.append(nid)
                        touched_node_ids.add(nid)
                        logger.info("    + %s", nid)

                # Redirect edges from old node to first new node (best-effort)
                if new_ids:
                    for edge in graph.edges:
                        if edge.source == old_id:
                            edge.source = new_ids[-1]  # last node inherits outgoing
                        if edge.target == old_id:
                            edge.target = new_ids[0]  # first node inherits incoming

        elif op_type == "add_edge":
            edge_data = op.get("edge", {})
            src = edge_data.get("source", "")
            tgt = edge_data.get("target", "")
            if src in graph.nodes and tgt in graph.nodes and src != tgt:
                graph.add_edge(src, tgt, edge_data.get("condition", ""))
                logger.info("  Added edge: %s → %s", src, tgt)

        elif op_type == "update_edge":
            edge_data = op.get("edge", {})
            src = edge_data.get("source", "")
            tgt = edge_data.get("target", "")
            condition = edge_data.get("condition")

            if src not in graph.nodes or tgt not in graph.nodes or src == tgt:
                continue

            updated = graph.update_edge(
                src,
                tgt,
                condition=condition,
            )
            if updated:
                logger.info("  Updated edge: %s → %s", src, tgt)
            else:
                logger.warning("  update_edge references missing edge: %s → %s", src, tgt)

    post_refine = _refine_nodes_if_needed(graph, touched_node_ids, scope="post_success_update")
    if post_refine:
        data = dict(data)
        data["post_refine"] = post_refine

    trace.record("success_updater", record.trajectory_id, data, graph)
    return graph


# ═══════════════════════════════════════════════════════════════════════
# Failure Updater
# ═══════════════════════════════════════════════════════════════════════


def failure_update(graph: SkillGraph, record: TrajectoryRecord, trace: BuildTrace) -> SkillGraph:
    """Add pitfalls from a failed trajectory to existing nodes."""
    assert not record.resolved, "Failure updater requires a failed trajectory"

    data, cost = _call_llm_json_with_observation_fallback(
        lambda steps_text: FAILURE_UPDATER_USER.format(
            graph_content=graph.summary(),
            task_description=record.task_description,
            trajectory_steps=steps_text,
        ),
        system=FAILURE_UPDATER_SYSTEM,
        record=record,
        stage="Failure updater",
    )

    if not isinstance(data, dict):
        logger.warning("Failure updater: expected dict, got %s", type(data).__name__)
        data = {"pitfalls": []}

    pitfalls = data.get("pitfalls", [])
    logger.info("Failure updater: %d pitfalls (cost=$%.4f)", len(pitfalls), cost)

    for pf in pitfalls:
        node_id = pf.get("node_id", "")
        pitfall_text = _generalize_free_text(pf.get("pitfall", ""), task_id=record.task_id)
        if not node_id or not pitfall_text:
            continue
        if node_id not in graph.nodes:
            logger.warning("  Pitfall references unknown node: %s", node_id)
            continue

        current_md = graph.get_node(node_id) or ""
        updated_md = _executor_add_pitfalls(current_md, [pitfall_text.strip()])
        updated_md = _ensure_navigation_card(updated_md, fallback_markdown=current_md, fallback_node_id=node_id)
        graph.update_node(node_id, updated_md)
        logger.info("  Added pitfall to: %s", node_id)

    trace.record("failure_updater", record.trajectory_id, data, graph)
    return graph


def normalize_task_granularity(graph: SkillGraph, trace: BuildTrace) -> tuple[SkillGraph, list[dict]]:
    """Run a light end-of-task granularity pass without changing graph topology."""
    reports = _refine_nodes_if_needed(graph, graph.node_ids(), scope="task_end")
    trace.record("granularity_normalizer", "task_end", {"nodes": reports}, graph)
    return graph, reports


def reconcile_task_graph(graph: SkillGraph, trace: BuildTrace) -> tuple[SkillGraph, dict]:
    """Lightly reconcile scan cards and branch semantics at task end."""
    prompt = TASK_RECONCILE_USER.format(graph_content=graph.summary())
    data, cost = call_llm_json(prompt, system=TASK_RECONCILE_SYSTEM)

    if not isinstance(data, dict):
        logger.warning("Task reconciler: expected dict, got %s", type(data).__name__)
        data = {"node_cards": [], "edge_operations": []}

    node_cards = data.get("node_cards", [])
    edge_operations = data.get("edge_operations", [])
    logger.info(
        "Task reconciler: %d node card update(s), %d edge op(s) (cost=$%.4f)",
        len(node_cards), len(edge_operations), cost,
    )

    for card in node_cards:
        if not isinstance(card, dict):
            continue
        node_id = card.get("node_id", "")
        if not node_id or node_id not in graph.nodes:
            continue
        current_md = graph.get_node(node_id) or ""
        updated_md = _apply_node_card(current_md, node_id=node_id, card=card, fallback_markdown=current_md)
        graph.update_node(node_id, updated_md)
        logger.info("  Reconciled node card: %s", node_id)

    for op in edge_operations:
        if not isinstance(op, dict):
            continue
        op_type = op.get("op", "")
        edge_data = op.get("edge", {})
        src = edge_data.get("source", "")
        tgt = edge_data.get("target", "")

        if src not in graph.nodes or tgt not in graph.nodes or src == tgt:
            continue

        if op_type == "add_edge":
            graph.add_edge(src, tgt, edge_data.get("condition", ""))
            logger.info("  Reconciled add edge: %s → %s", src, tgt)
        elif op_type == "update_edge":
            condition = edge_data.get("condition")
            updated = graph.update_edge(src, tgt, condition=condition)
            if updated:
                logger.info("  Reconciled edge: %s → %s", src, tgt)

    trace.record("task_reconciler", "task_end", data, graph)
    return graph, data


# ═══════════════════════════════════════════════════════════════════════
# Executor (LLM-based markdown merging)
# ═══════════════════════════════════════════════════════════════════════


def _executor_update_node(current_md: str, new_info: str) -> str:
    """Use LLM to integrate new information into an existing node's markdown."""
    prompt = EXECUTOR_UPDATE_USER.format(
        current_markdown=current_md,
        new_info=new_info,
    )
    resp = call_llm(prompt, system=EXECUTOR_UPDATE_SYSTEM)
    result = _coerce_markdown_response(resp.text.strip(), current_md, "Executor update")
    return _ensure_navigation_card(result, fallback_markdown=current_md)


def _executor_add_pitfall(current_md: str, pitfall: str) -> str:
    """Use LLM to add a pitfall to an existing node's markdown."""
    return _executor_add_pitfalls(current_md, [pitfall])


def _executor_add_pitfalls(current_md: str, pitfalls: list[str]) -> str:
    """Use LLM to add one or more pitfalls to an existing node's markdown."""
    pitfall_text = "\n".join(f"- {pitfall}" for pitfall in pitfalls if pitfall.strip())
    prompt = EXECUTOR_PITFALL_USER.format(
        current_markdown=current_md,
        pitfall=pitfall_text,
    )
    resp = call_llm(prompt, system=EXECUTOR_PITFALL_SYSTEM)
    result = _coerce_markdown_response(resp.text.strip(), current_md, "Executor pitfall")
    return _ensure_navigation_card(result, fallback_markdown=current_md)


def _executor_refine_node(current_md: str, refine_reason: str) -> str:
    """Use LLM to lightly refine node granularity without changing the skill."""
    prompt = EXECUTOR_REFINE_USER.format(
        current_markdown=current_md,
        refine_reason=refine_reason,
    )
    resp = call_llm(prompt, system=EXECUTOR_REFINE_SYSTEM)
    result = _coerce_markdown_response(resp.text.strip(), current_md, "Executor refine")
    return _ensure_navigation_card(result, fallback_markdown=current_md)


def _coerce_markdown_response(result: str, fallback: str, label: str) -> str:
    """Strip wrappers and validate markdown executor output."""
    if result.startswith("```"):
        match = re.search(r"```(?:markdown)?\s*\n(.*?)\n```", result, re.DOTALL)
        if match:
            result = match.group(1).strip()

    if not result.startswith("---"):
        logger.warning("%s: output missing frontmatter, keeping original", label)
        return fallback

    return result


def _refine_nodes_if_needed(graph: SkillGraph, node_ids: set[str] | list[str], *, scope: str) -> list[dict]:
    """Lightly refine flagged nodes while leaving topology unchanged."""
    reports: list[dict] = []
    for node_id in sorted(set(node_ids)):
        if node_id not in graph.nodes:
            continue

        current_md = graph.get_node(node_id) or ""
        assessment = _assess_node_granularity(current_md)
        report = {"node_id": node_id, **assessment, "refined": False}

        if assessment["status"] == "fat":
            reason = (
                f"{scope}: node is too detailed "
                f"(lines={assessment['non_empty_lines']}, "
                f"actions={assessment['action_bullets']}, "
                f"reasons={', '.join(assessment['reasons'])})"
            )
            refined_md = _executor_refine_node(current_md, reason)
            if refined_md != current_md:
                graph.update_node(node_id, refined_md)
                report["refined"] = True
                report["after"] = _assess_node_granularity(refined_md)
                logger.info("  Refined node granularity: %s", node_id)
        elif assessment["status"] == "thin":
            logger.info("  Granularity flag (thin): %s", node_id)

        reports.append(report)

    return reports


# ═══════════════════════════════════════════════════════════════════════
# Link Embedding
# ═══════════════════════════════════════════════════════════════════════


def embed_links(graph: SkillGraph, trace: BuildTrace) -> SkillGraph:
    """Embed wikilinks into node markdown content based on graph edges."""
    if not graph.edges:
        return graph

    for node_id in graph.node_ids():
        outgoing = [e for e in graph.edges if e.source == node_id]
        incoming = [e for e in graph.edges if e.target == node_id]

        if not outgoing and not incoming:
            continue

        # Build links description for the LLM
        links_lines = []
        for e in outgoing:
            links_lines.append(
                f"- Outgoing: → [[{e.target}]]\n"
                f"  Condition: {e.condition}"
            )
        for e in incoming:
            links_lines.append(
                f"- Incoming: ← [[{e.source}]]\n"
                f"  Condition: {e.condition}"
            )
        links_text = "\n".join(links_lines)

        current_md = graph.get_node(node_id)
        prompt = LINK_EMBED_USER.format(
            current_markdown=current_md,
            links_text=links_text,
        )

        resp = call_llm(prompt, system=LINK_EMBED_SYSTEM)
        result = resp.text.strip()

        # Strip code fences
        if result.startswith("```"):
            match = re.search(r"```(?:markdown)?\s*\n(.*?)\n```", result, re.DOTALL)
            if match:
                result = match.group(1).strip()

        if not result.startswith("---"):
            logger.warning("Link embedding: output missing frontmatter for %s, keeping original", node_id)
            continue

        graph.update_node(node_id, result)
        logger.info("  Embedded links in: %s (%d outgoing, %d incoming)", node_id, len(outgoing), len(incoming))

    trace.record("link_embedding", "all", {"nodes_updated": graph.node_ids()}, graph)
    return graph
