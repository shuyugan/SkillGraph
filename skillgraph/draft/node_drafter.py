"""Step 2c: Generate draft markdown nodes from aggregated items."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from skillgraph.config import CACHE_DIR
from skillgraph.draft.prompts import NODE_DRAFT_SYSTEM, NODE_DRAFT_USER
from skillgraph.llm import call_llm
from skillgraph.models import AggregatedItem, DraftNode, Relationship, TaskSummary

logger = logging.getLogger("skillgraph.draft")

STEP2C_CACHE = CACHE_DIR / "step2c"


def generate_draft_nodes(
    summary: TaskSummary, *, use_cache: bool = True
) -> list[DraftNode]:
    """Generate draft markdown nodes from a TaskSummary's aggregated items.

    Each AggregatedItem becomes one DraftNode with markdown content.
    Nodes are task-specific at this stage (generalization happens in Step 3 merge).
    """
    task_cache = STEP2C_CACHE / summary.task_id
    task_cache.mkdir(parents=True, exist_ok=True)

    drafts = []
    for item in summary.items:
        cache_file = task_cache / f"{item.item_id}.json"

        if use_cache and cache_file.exists():
            draft = _load_cache(cache_file)
            logger.debug("Loaded cached draft for %s/%s", summary.task_id, item.item_id)
        else:
            draft = _generate_single(item, summary)
            _save_cache(cache_file, draft)
            logger.info("Generated draft node: %s", draft.node_id)

        drafts.append(draft)

    return drafts


def _generate_single(item: AggregatedItem, summary: TaskSummary) -> DraftNode:
    """Generate a single draft node via LLM."""
    code_examples_text = "\n".join(f"- {ex}" for ex in item.code_examples) or "(none)"

    prompt = NODE_DRAFT_USER.format(
        type=item.type,
        title=item.title,
        description=item.description,
        code_examples=code_examples_text,
    )

    resp = call_llm(prompt, system=NODE_DRAFT_SYSTEM)
    markdown = resp.text.strip()

    # Strip outer code block wrapper if LLM wrapped the markdown
    if markdown.startswith("```"):
        match = re.search(r"```(?:markdown)?\s*\n(.*?)\n```", markdown, re.DOTALL)
        if match:
            markdown = match.group(1).strip()

    # Extract slug from the generated YAML frontmatter
    slug = _extract_slug(markdown, item.title)

    # Collect relationships involving this item
    item_rels = [
        r
        for r in summary.relationships
        if r.source_item_id == item.item_id or r.target_item_id == item.item_id
    ]

    return DraftNode(
        node_id=slug,
        task_id=summary.task_id,
        type=item.type,
        markdown=markdown,
        source_item_ids=[item.item_id],
        relationships=item_rels,
    )


def _extract_slug(markdown: str, fallback_title: str) -> str:
    """Extract the name/slug from YAML frontmatter, or derive from title."""
    match = re.search(r"^name:\s*(.+)$", markdown, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Derive slug from title
    slug = fallback_title.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug)
    return slug[:60]


# ── Cache helpers ──


def _save_cache(path: Path, draft: DraftNode) -> None:
    data = {
        "node_id": draft.node_id,
        "task_id": draft.task_id,
        "type": draft.type,
        "markdown": draft.markdown,
        "source_item_ids": draft.source_item_ids,
        "relationships": [
            {
                "rel_type": r.rel_type,
                "source_item_id": r.source_item_id,
                "target_item_id": r.target_item_id,
                "description": r.description,
            }
            for r in draft.relationships
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_cache(path: Path) -> DraftNode:
    data = json.loads(path.read_text(encoding="utf-8"))
    relationships = [
        Relationship(
            rel_type=r["rel_type"],
            source_item_id=r["source_item_id"],
            target_item_id=r["target_item_id"],
            description=r.get("description", ""),
        )
        for r in data.get("relationships", [])
    ]
    return DraftNode(
        node_id=data["node_id"],
        task_id=data["task_id"],
        type=data["type"],
        markdown=data["markdown"],
        source_item_ids=data.get("source_item_ids", []),
        relationships=relationships,
    )
