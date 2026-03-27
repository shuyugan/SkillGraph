"""Step 4: Build links between skill nodes based on trajectory relationships."""

from __future__ import annotations

import logging
import re

from skillgraph.link.prompts import (
    LINK_BUILD_SYSTEM,
    LINK_BUILD_USER,
    format_available_nodes_text,
    format_relationships_text,
)
from skillgraph.llm import call_llm
from skillgraph.models import Relationship, SkillNode, TaskSummary

logger = logging.getLogger("skillgraph.link")


def build_links(
    nodes: list[SkillNode],
    task_summaries: list[TaskSummary],
    item_to_node_map: dict[str, str],
) -> list[SkillNode]:
    """Build wikilinks between nodes based on relationships from task summaries.

    Args:
        nodes: current skill graph nodes
        task_summaries: all processed TaskSummary objects
        item_to_node_map: mapping from aggregated item_id to final node_id

    Returns:
        updated nodes with wikilinks embedded
    """
    # Collect all relationships, mapped to node IDs
    node_relationships: dict[str, list[dict]] = {n.node_id: [] for n in nodes}

    for summary in task_summaries:
        for rel in summary.relationships:
            source_node = item_to_node_map.get(rel.source_item_id)
            target_node = item_to_node_map.get(rel.target_item_id)

            if not source_node or not target_node:
                continue
            if source_node == target_node:
                continue
            if source_node not in node_relationships:
                continue

            node_relationships[source_node].append(
                {
                    "rel_type": rel.rel_type,
                    "target_slug": target_node,
                    "description": rel.description,
                }
            )

    # For each node that has relationships, embed links via LLM
    updated = []
    for node in nodes:
        rels = node_relationships.get(node.node_id, [])

        # Deduplicate relationships
        seen = set()
        unique_rels = []
        for r in rels:
            key = (r["rel_type"], r["target_slug"])
            if key not in seen:
                seen.add(key)
                unique_rels.append(r)

        if not unique_rels:
            updated.append(node)
            continue

        updated_node = _embed_links(node, unique_rels, nodes)
        updated.append(updated_node)

    logger.info("Built links for %d nodes", sum(1 for n in nodes if node_relationships.get(n.node_id)))
    return updated


def _embed_links(
    node: SkillNode,
    relationships: list[dict],
    all_nodes: list[SkillNode],
) -> SkillNode:
    """Embed wikilinks into a single node via LLM."""
    rels_text = format_relationships_text(relationships)
    available_text = format_available_nodes_text(all_nodes, node.node_id)

    prompt = LINK_BUILD_USER.format(
        target_node_markdown=node.markdown,
        relationships_text=rels_text,
        available_nodes_text=available_text,
    )

    resp = call_llm(prompt, system=LINK_BUILD_SYSTEM)
    updated_md = resp.text.strip()

    # Clean up if LLM wrapped in code block
    if updated_md.startswith("```"):
        match = re.search(r"```(?:markdown)?\s*\n(.*?)\n```", updated_md, re.DOTALL)
        if match:
            updated_md = match.group(1).strip()

    # Validate frontmatter exists
    if not updated_md.startswith("---"):
        logger.warning("Link embedding failed for %s, keeping original", node.node_id)
        return node

    return SkillNode(
        node_id=node.node_id,
        type=node.type,
        markdown=updated_md,
        source_tasks=node.source_tasks,
    )
