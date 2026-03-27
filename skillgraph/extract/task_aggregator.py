"""Step 2b: Cross-trajectory aggregation within a single task."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from skillgraph.config import CACHE_DIR
from skillgraph.extract.prompts import (
    TASK_AGGREGATION_SYSTEM,
    TASK_AGGREGATION_USER,
    build_trajectory_analysis_text,
)
from skillgraph.llm import call_llm_json
from skillgraph.models import (
    AggregatedItem,
    Relationship,
    TaskSummary,
    TrajectoryAnalysis,
)

logger = logging.getLogger("skillgraph.extract")

STEP2B_CACHE = CACHE_DIR / "step2b"


def aggregate_task(
    task_id: str,
    analyses: list[TrajectoryAnalysis],
    *,
    use_cache: bool = True,
) -> TaskSummary:
    """Aggregate multiple trajectory analyses for the same task.

    Compares resolved vs failed trajectories to identify common techniques,
    error-recovery patterns, and anti-patterns.
    """
    STEP2B_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = STEP2B_CACHE / f"{task_id}.json"

    if use_cache and cache_file.exists():
        summary = _load_cache(cache_file)
        logger.debug("Loaded cached task summary for %s", task_id)
        return summary

    # Build prompt
    n_resolved = sum(1 for a in analyses if a.resolved)
    n_failed = sum(1 for a in analyses if not a.resolved)

    analyses_text = "\n\n".join(
        build_trajectory_analysis_text(a) for a in analyses
    )

    prompt = TASK_AGGREGATION_USER.format(
        n_trajectories=len(analyses),
        n_resolved=n_resolved,
        n_failed=n_failed,
        trajectory_analyses=analyses_text,
    )

    # Call LLM
    data, cost = call_llm_json(prompt, system=TASK_AGGREGATION_SYSTEM)

    # Parse response
    raw_items = data.get("items", []) if isinstance(data, dict) else []
    raw_rels = data.get("relationships", []) if isinstance(data, dict) else []

    # Build item ID mapping (title → id) for relationship resolution
    items = []
    title_to_id = {}
    for raw in raw_items:
        item_id = str(uuid.uuid4())[:8]
        title = raw.get("title", "")
        title_to_id[title] = item_id

        # Collect source segment IDs from the raw data
        source_ids = raw.get("source_segment_ids", [])

        items.append(
            AggregatedItem(
                item_id=item_id,
                type=raw.get("type", "technique"),
                title=title,
                description=raw.get("description", ""),
                code_examples=raw.get("code_examples", []),
                source_segment_ids=source_ids,
            )
        )

    # Resolve relationships (title-based → id-based)
    relationships = []
    for raw_rel in raw_rels:
        source_title = raw_rel.get("source_title", "")
        target_title = raw_rel.get("target_title", "")
        source_id = title_to_id.get(source_title, "")
        target_id = title_to_id.get(target_title, "")
        if source_id and target_id:
            relationships.append(
                Relationship(
                    rel_type=raw_rel.get("rel_type", "if_fails"),
                    source_item_id=source_id,
                    target_item_id=target_id,
                    description=raw_rel.get("description", ""),
                )
            )

    summary = TaskSummary(
        task_id=task_id,
        items=items,
        relationships=relationships,
    )

    _save_cache(cache_file, summary)
    logger.info(
        "Aggregated %d items + %d relationships for %s (cost=$%.4f)",
        len(items),
        len(relationships),
        task_id,
        cost,
    )
    return summary


# ── Cache helpers ──


def _save_cache(path: Path, summary: TaskSummary) -> None:
    data = {
        "task_id": summary.task_id,
        "items": [
            {
                "item_id": item.item_id,
                "type": item.type,
                "title": item.title,
                "description": item.description,
                "code_examples": item.code_examples,
                "source_segment_ids": item.source_segment_ids,
            }
            for item in summary.items
        ],
        "relationships": [
            {
                "rel_type": r.rel_type,
                "source_item_id": r.source_item_id,
                "target_item_id": r.target_item_id,
                "description": r.description,
            }
            for r in summary.relationships
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_cache(path: Path) -> TaskSummary:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = [
        AggregatedItem(
            item_id=d["item_id"],
            type=d["type"],
            title=d["title"],
            description=d["description"],
            code_examples=d.get("code_examples", []),
            source_segment_ids=d.get("source_segment_ids", []),
        )
        for d in data["items"]
    ]
    relationships = [
        Relationship(
            rel_type=r["rel_type"],
            source_item_id=r["source_item_id"],
            target_item_id=r["target_item_id"],
            description=r.get("description", ""),
        )
        for r in data.get("relationships", [])
    ]
    return TaskSummary(
        task_id=data["task_id"],
        items=items,
        relationships=relationships,
    )
