"""Step 3: Match new draft nodes against existing graph and merge if matching."""

from __future__ import annotations

import logging
import re

from skillgraph.compare.prompts import (
    MATCH_SYSTEM,
    MATCH_USER,
    MERGE_SYSTEM,
    MERGE_USER,
    format_existing_nodes_for_match,
)
from skillgraph.llm import call_llm, call_llm_json
from skillgraph.models import DraftNode, SkillNode

logger = logging.getLogger("skillgraph.compare")


def compare_and_merge(
    draft: DraftNode,
    existing_nodes: list[SkillNode],
) -> tuple[SkillNode, SkillNode | None]:
    """Compare a draft node against existing graph and merge if matching.

    Sends ALL existing nodes of the same type in one prompt.
    LLM picks the match (or "none").

    Returns:
        (result_node, merged_with):
        - result_node: the new or merged SkillNode
        - merged_with: the existing node it was merged with (or None)
    """
    # Build summary of all existing nodes of the same type
    existing_summary = format_existing_nodes_for_match(existing_nodes, draft.type)

    if existing_summary == "(no existing nodes of this type)":
        # No candidates → new node
        return _draft_to_node(draft), None

    # One LLM call: find match
    prompt = MATCH_USER.format(
        new_node_markdown=draft.markdown[:3000],
        existing_nodes_summary=existing_summary,
    )

    try:
        data, cost = call_llm_json(prompt, system=MATCH_SYSTEM)
        match_id = data.get("match_id", "none") if isinstance(data, dict) else "none"
        reason = data.get("reason", "") if isinstance(data, dict) else ""
    except Exception as e:
        logger.warning("Match finding failed: %s", e)
        return _draft_to_node(draft), None

    if match_id == "none" or not match_id:
        logger.info("No match for '%s': %s", draft.node_id, reason)
        return _draft_to_node(draft), None

    # Find the matched existing node
    matched_node = next(
        (n for n in existing_nodes if n.node_id == match_id), None
    )
    if not matched_node:
        logger.warning(
            "LLM returned match_id '%s' but node not found in graph", match_id
        )
        return _draft_to_node(draft), None

    # Merge
    merged = _merge_nodes(draft, matched_node)
    logger.info(
        "Matched '%s' with '%s': %s", draft.node_id, match_id, reason
    )
    return merged, matched_node


def _merge_nodes(draft: DraftNode, existing: SkillNode) -> SkillNode:
    """Merge a draft node with an existing node via LLM generalization."""
    prompt = MERGE_USER.format(
        node_a_markdown=existing.markdown,
        node_b_markdown=draft.markdown,
    )

    resp = call_llm(prompt, system=MERGE_SYSTEM)
    merged_markdown = resp.text.strip()

    # Clean up code block wrapping if present
    if merged_markdown.startswith("```"):
        match = re.search(
            r"```(?:markdown)?\s*\n(.*?)\n```", merged_markdown, re.DOTALL
        )
        if match:
            merged_markdown = match.group(1).strip()

    # Extract slug from merged markdown
    slug = _extract_slug(merged_markdown, existing.node_id)

    source_tasks = list(set(existing.source_tasks + [draft.task_id]))

    return SkillNode(
        node_id=slug,
        type=existing.type,
        markdown=merged_markdown,
        source_tasks=source_tasks,
    )


def _draft_to_node(draft: DraftNode) -> SkillNode:
    """Convert a DraftNode to a SkillNode (no merge, keep task-specific)."""
    return SkillNode(
        node_id=draft.node_id,
        type=draft.type,
        markdown=draft.markdown,
        source_tasks=[draft.task_id],
    )


def _extract_slug(markdown: str, fallback: str) -> str:
    match = re.search(r"^name:\s*(.+)$", markdown, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback
