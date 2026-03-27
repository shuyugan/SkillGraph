"""Step 2a: Per-trajectory segment extraction using LLM."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from skillgraph.config import CACHE_DIR
from skillgraph.extract.prompts import (
    SEGMENT_EXTRACTION_SYSTEM,
    SEGMENT_EXTRACTION_USER,
    build_trajectory_steps_text,
)
from skillgraph.llm import call_llm_json
from skillgraph.models import ErrorRecoveryPair, Segment, TrajectoryAnalysis
from skillgraph.parse.models import TrajectoryRecord

logger = logging.getLogger("skillgraph.extract")

STEP2A_CACHE = CACHE_DIR / "step2a"


def extract_segments(
    record: TrajectoryRecord, *, use_cache: bool = True
) -> TrajectoryAnalysis:
    """Extract knowledge segments from a single trajectory.

    Returns a TrajectoryAnalysis containing typed segments and error-recovery pairs.
    Results are cached per trajectory_id.
    """
    STEP2A_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = STEP2A_CACHE / f"{record.trajectory_id}.json"

    if use_cache and cache_file.exists():
        analysis = _load_cache(cache_file)
        logger.debug("Loaded cached analysis for %s", record.trajectory_id)
        return analysis

    # Build prompt
    steps_text = build_trajectory_steps_text(record.steps)
    result_label = "RESOLVED" if record.resolved else "FAILED"

    prompt = SEGMENT_EXTRACTION_USER.format(
        task_description=record.task_description[:2000],
        result=result_label,
        trajectory_steps=steps_text,
    )

    # Call LLM
    data, cost = call_llm_json(prompt, system=SEGMENT_EXTRACTION_SYSTEM)

    # Parse response
    raw_segments = data.get("segments", []) if isinstance(data, dict) else []
    raw_pairs = data.get("error_recovery_pairs", []) if isinstance(data, dict) else []

    segments = []
    for raw in raw_segments:
        seg = Segment(
            segment_id=str(uuid.uuid4())[:8],
            trajectory_id=record.trajectory_id,
            task_id=record.task_id,
            type=raw.get("type", "technique"),
            title=raw.get("title", ""),
            description=raw.get("description", ""),
            code_examples=raw.get("code_examples", []),
            context=raw.get("context", ""),
            outcome=raw.get("outcome", ""),
            order=raw.get("order", 0),
        )
        segments.append(seg)

    recovery_pairs = []
    for raw_pair in raw_pairs:
        # Map order numbers to segment IDs
        failed_order = raw_pair.get("failed_segment_order", 0)
        recovery_order = raw_pair.get("recovery_segment_order", 0)
        failed_seg = next((s for s in segments if s.order == failed_order), None)
        recovery_seg = next((s for s in segments if s.order == recovery_order), None)
        if failed_seg and recovery_seg:
            recovery_pairs.append(
                ErrorRecoveryPair(
                    failed_segment_id=failed_seg.segment_id,
                    recovery_segment_id=recovery_seg.segment_id,
                    error_description=raw_pair.get("error_description", ""),
                )
            )

    analysis = TrajectoryAnalysis(
        trajectory_id=record.trajectory_id,
        task_id=record.task_id,
        resolved=record.resolved,
        segments=segments,
        recovery_pairs=recovery_pairs,
    )

    # Cache
    _save_cache(cache_file, analysis)
    logger.info(
        "Extracted %d segments from %s (cost=$%.4f)",
        len(segments),
        record.trajectory_id,
        cost,
    )
    return analysis


# ── Cache helpers ──


def _save_cache(path: Path, analysis: TrajectoryAnalysis) -> None:
    data = {
        "trajectory_id": analysis.trajectory_id,
        "task_id": analysis.task_id,
        "resolved": analysis.resolved,
        "segments": [
            {
                "segment_id": s.segment_id,
                "trajectory_id": s.trajectory_id,
                "task_id": s.task_id,
                "type": s.type,
                "title": s.title,
                "description": s.description,
                "code_examples": s.code_examples,
                "context": s.context,
                "outcome": s.outcome,
                "order": s.order,
            }
            for s in analysis.segments
        ],
        "recovery_pairs": [
            {
                "failed_segment_id": p.failed_segment_id,
                "recovery_segment_id": p.recovery_segment_id,
                "error_description": p.error_description,
            }
            for p in analysis.recovery_pairs
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_cache(path: Path) -> TrajectoryAnalysis:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        Segment(
            segment_id=s["segment_id"],
            trajectory_id=s["trajectory_id"],
            task_id=s["task_id"],
            type=s["type"],
            title=s["title"],
            description=s["description"],
            code_examples=s.get("code_examples", []),
            context=s.get("context", ""),
            outcome=s.get("outcome", ""),
            order=s.get("order", 0),
        )
        for s in data["segments"]
    ]
    recovery_pairs = [
        ErrorRecoveryPair(
            failed_segment_id=p["failed_segment_id"],
            recovery_segment_id=p["recovery_segment_id"],
            error_description=p["error_description"],
        )
        for p in data.get("recovery_pairs", [])
    ]
    return TrajectoryAnalysis(
        trajectory_id=data["trajectory_id"],
        task_id=data["task_id"],
        resolved=data["resolved"],
        segments=segments,
        recovery_pairs=recovery_pairs,
    )
