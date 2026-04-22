"""Cross-task merger: level-wise clustering + direct graph synthesis."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from skillgraph.llm import call_llm, call_llm_json
from skillgraph.merge.prompts import (
    CLUSTER_LEVEL_SYSTEM,
    CLUSTER_LEVEL_USER,
    DIRECT_GRAPH_SYNTHESIS_SYSTEM,
    DIRECT_GRAPH_SYNTHESIS_USER,
    EDGE_RECONCILE_SYSTEM,
    EDGE_RECONCILE_USER,
    GRAPH_ALIGN_SYSTEM,
    GRAPH_ALIGN_USER,
    MERGE_EXECUTOR_SYSTEM,
    MERGE_EXECUTOR_USER,
    format_graph_packet,
    format_edge_reconcile_context,
    format_graph_summary,
)
from skillgraph.models import SkillGraph

logger = logging.getLogger("skillgraph.merge")


# ═══════════════════════════════════════════════════════════════════════
# Merge Trace
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MergeTrace:
    """Records every step of the cross-task merge process."""

    levels: list[dict] = field(default_factory=list)

    def record_level(
        self,
        level: int,
        clustering: dict,
        merges: list[dict],
        results: list["GraphBundle"],
    ) -> None:
        self.levels.append({
            "level": level,
            "clustering": clustering,
            "merges": merges,
            "n_subgraphs_after_level": len(results),
            "result_snapshots": [
                {
                    "index": idx,
                    "label": result.label,
                    "n_nodes": len(result.graph.nodes),
                    "n_edges": len(result.graph.edges),
                    "node_ids": result.graph.node_ids(),
                    "source_tasks": result.task_ids,
                }
                for idx, result in enumerate(results)
            ],
        })

    def to_dict(self) -> dict:
        return {"levels": self.levels}


# ═══════════════════════════════════════════════════════════════════════
# Level-wise graph bundles
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class GraphBundle:
    """A graph plus the source-task context needed for higher-level clustering."""

    graph: SkillGraph
    task_ids: list[str]
    task_descriptions: list[str]
    label: str


def merge_subgraphs(
    graph_a: SkillGraph,
    graph_b: SkillGraph,
) -> tuple[SkillGraph, dict]:
    """Merge two subgraphs using graph-pair alignment and node merge execution."""
    merge_log = {
        "graph_a_nodes": graph_a.node_ids(),
        "graph_b_nodes": graph_b.node_ids(),
        "alignment": {},
        "validated_merge_pairs": [],
        "skipped_merge_pairs": [],
        "resolution": [],
        "node_id_map": {"A": {}, "B": {}},
        "edge_reconciliation": [],
    }
    id_map_a = merge_log["node_id_map"]["A"]
    id_map_b = merge_log["node_id_map"]["B"]

    # Step 1: Graph-pair alignment
    logger.info("  Step 1: Graph-pair alignment judge (%d nodes vs %d nodes)",
                len(graph_a.nodes), len(graph_b.nodes))
    alignment = _align_graph_pair(graph_a, graph_b)
    merge_log["alignment"] = alignment

    validated_pairs, skipped_pairs = _validate_merge_pairs(alignment, graph_a, graph_b)
    merge_log["validated_merge_pairs"] = validated_pairs
    merge_log["skipped_merge_pairs"] = skipped_pairs

    # Step 2: Execute validated merge pairs
    logger.info("  Step 2: Execute %d validated merge pair(s)", len(validated_pairs))
    merged_graph = SkillGraph()
    merged_a = set()
    merged_b = set()

    for pair in validated_pairs:
        a_id = pair["a_node"]
        b_id = pair["b_node"]
        new_md = _execute_merge(graph_a.get_node(a_id), graph_b.get_node(b_id))
        new_id = _extract_node_id(new_md, fallback=a_id)
        new_id = _add_node_with_unique_id(merged_graph, new_id, new_md)
        merged_a.add(a_id)
        merged_b.add(b_id)
        id_map_a[a_id] = new_id
        id_map_b[b_id] = new_id
        merge_log["resolution"].append({
            "type": "merge",
            "a_node": a_id,
            "b_node": b_id,
            "result": new_id,
            "reason": pair.get("reason", ""),
        })
        logger.info("    MERGE: %s + %s → %s", a_id, b_id, new_id)

    # Step 3: Add unmatched nodes
    logger.info("  Step 3: Add unmatched nodes")
    for a_id in graph_a.node_ids():
        if a_id in merged_a:
            continue
        new_id = _add_node_with_unique_id(merged_graph, a_id, graph_a.get_node(a_id))
        id_map_a[a_id] = new_id
        merge_log["resolution"].append({"type": "add", "node": a_id, "side": "A", "result": new_id})
        logger.info("    ADD (A): %s%s", a_id, f" → {new_id}" if new_id != a_id else "")

    for b_id in graph_b.node_ids():
        if b_id in merged_b:
            continue
        new_id = _add_node_with_unique_id(merged_graph, b_id, graph_b.get_node(b_id))
        id_map_b[b_id] = new_id
        merge_log["resolution"].append({"type": "add", "node": b_id, "side": "B", "result": new_id})
        logger.info("    ADD (B): %s%s", b_id, f" → {new_id}" if new_id != b_id else "")

    # Step 4: Edge remapping
    logger.info("  Step 4: Edge remapping")
    valid_ids = set(merged_graph.node_ids())
    remapped_edges: list[dict] = []

    for edge in graph_a.edges:
        new_src = id_map_a.get(edge.source, edge.source)
        new_tgt = id_map_a.get(edge.target, edge.target)
        if new_src in valid_ids and new_tgt in valid_ids and new_src != new_tgt:
            remapped_edges.append({
                "source": new_src,
                "target": new_tgt,
                "condition": edge.condition,
                "origin": "A",
            })

    for edge in graph_b.edges:
        new_src = id_map_b.get(edge.source, edge.source)
        new_tgt = id_map_b.get(edge.target, edge.target)
        if new_src in valid_ids and new_tgt in valid_ids and new_src != new_tgt:
            remapped_edges.append({
                "source": new_src,
                "target": new_tgt,
                "condition": edge.condition,
                "origin": "B",
            })

    # Step 5: Semantic edge reconcile
    logger.info("  Step 5: Reconcile %d remapped edge candidate(s)", len(remapped_edges))
    merge_log["edge_reconciliation"] = _reconcile_cross_task_edges(merged_graph, remapped_edges)

    logger.info(
        "  Merge complete: %d+%d nodes → %d nodes, %d edges",
        len(graph_a.nodes), len(graph_b.nodes),
        len(merged_graph.nodes), len(merged_graph.edges),
    )

    return merged_graph, merge_log


def _judge_node(
    node_id: str,
    source_graph: SkillGraph,
    top_k_candidates: list[tuple[str, float]],
    candidate_graph: SkillGraph,
    side_label: str,
) -> dict:
    """Legacy stub retained only for compatibility."""
    raise NotImplementedError("Per-node candidate judge has been replaced by graph-pair alignment.")

# ═══════════════════════════════════════════════════════════════════════
# Tree merge orchestrator
# ═══════════════════════════════════════════════════════════════════════


def tree_merge(
    subgraphs: list[SkillGraph],
    output_dir: Path,
    *,
    source_task_ids: list[str] | None = None,
    task_descriptions: dict[str, str] | None = None,
    cluster_capacity: int = 4,
    merge_workers: int = 4,
    initial_bundles: list[GraphBundle] | None = None,
    start_level: int = 0,
    trace: MergeTrace | None = None,
) -> tuple[SkillGraph, MergeTrace]:
    """Merge multiple subgraphs using level-wise clustering + direct synthesis."""
    trace = trace or MergeTrace()

    if initial_bundles is None and not subgraphs:
        return SkillGraph(), trace

    source_task_ids = source_task_ids or [f"graph_{i}" for i in range(len(subgraphs))]
    task_descriptions = task_descriptions or {}
    cluster_capacity = max(2, min(int(cluster_capacity or 4), 4))
    merge_workers = max(1, int(merge_workers or 1))

    if initial_bundles is not None:
        current_level = initial_bundles
        level_num = max(0, int(start_level or 0))
    else:
        current_level = [
            GraphBundle(
                graph=graph,
                task_ids=[task_id],
                task_descriptions=[task_descriptions.get(task_id, "")] if task_descriptions.get(task_id, "") else [],
                label=task_id,
            )
            for task_id, graph in zip(source_task_ids, subgraphs)
        ]
        level_num = 0

    if len(current_level) == 1:
        return current_level[0].graph, trace

    while len(current_level) > 1:
        logger.info("=" * 60)
        logger.info("Clustering-guided merge Level %d: %d graph bundle(s)", level_num, len(current_level))

        if 1 < len(current_level) <= cluster_capacity:
            logger.info(
                "Remaining %d graph bundle(s) fit within cluster capacity %d; "
                "skipping clustering and synthesizing a single merged graph",
                len(current_level),
                cluster_capacity,
            )
            output_label = f"level_{level_num + 1}_group_0"
            merged_bundle, merge_log = _synthesize_group(
                current_level,
                output_label=output_label,
            )

            clustering_log = {
                "mode": "forced_within_capacity",
                "cost": 0.0,
                "raw_output": {
                    "groups": [
                        {
                            "members": list(range(len(current_level))),
                            "reason": "Remaining graphs fit within cluster capacity and are merged directly.",
                        }
                    ]
                },
                "validated_groups": [list(range(len(current_level)))],
                "validation": [],
            }
            finalized_next_level = [merged_bundle]
            finalized_level_merges = [{
                "mode": "direct_graph_synthesis",
                "group_index": 0,
                "members": list(range(len(current_level))),
                **merge_log,
            }]

            level_dir = output_dir / f"level_{level_num}"
            level_dir.mkdir(parents=True, exist_ok=True)
            merged_bundle.graph.save(level_dir / "subgraph_0")
            (level_dir / "merge_logs.json").write_text(
                json.dumps(finalized_level_merges, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (level_dir / "clustering.json").write_text(
                json.dumps(clustering_log, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (level_dir / "subgraph_metadata.json").write_text(
                json.dumps([
                    {
                        "index": 0,
                        "label": merged_bundle.label,
                        "source_tasks": merged_bundle.task_ids,
                        "n_nodes": len(merged_bundle.graph.nodes),
                        "n_edges": len(merged_bundle.graph.edges),
                    }
                ], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            trace.record_level(level_num, clustering_log, finalized_level_merges, finalized_next_level)
            current_level = finalized_next_level
            level_num += 1
            continue

        groups, clustering_log = _cluster_level(current_level, cluster_capacity=cluster_capacity)
        next_level: list[GraphBundle | None] = [None] * len(groups)
        level_merges: list[dict | None] = [None] * len(groups)
        synthesis_jobs: list[tuple[int, list[int], list[GraphBundle], str]] = []

        for group_idx, group in enumerate(groups):
            members = [current_level[i] for i in group]
            if len(members) == 1:
                bundle = members[0]
                logger.info(
                    "  Pass-through group %d: %s (%d nodes)",
                    group_idx,
                    bundle.label,
                    len(bundle.graph.nodes),
                )
                next_level[group_idx] = bundle
                level_merges[group_idx] = {
                    "mode": "pass_through",
                    "group_index": group_idx,
                    "members": group,
                    "source_tasks": bundle.task_ids,
                    "label": bundle.label,
                    "n_input_nodes": len(bundle.graph.nodes),
                    "n_input_edges": len(bundle.graph.edges),
                }
                continue

            synthesis_jobs.append(
                (
                    group_idx,
                    group,
                    members,
                    f"level_{level_num + 1}_group_{group_idx}",
                )
            )

        if synthesis_jobs:
            effective_workers = min(merge_workers, len(synthesis_jobs))
            logger.info(
                "  Running %d cluster synthesis job(s) with %d parallel worker(s)",
                len(synthesis_jobs),
                effective_workers,
            )

            if effective_workers == 1:
                for group_idx, group, members, output_label in synthesis_jobs:
                    logger.info(
                        "  Synthesizing group %d: %d graph(s), %d total input node(s)",
                        group_idx,
                        len(members),
                        sum(len(b.graph.nodes) for b in members),
                    )
                    merged_bundle, merge_log = _synthesize_group(
                        members,
                        output_label=output_label,
                    )
                    next_level[group_idx] = merged_bundle
                    level_merges[group_idx] = {
                        "mode": "direct_graph_synthesis",
                        "group_index": group_idx,
                        "members": group,
                        **merge_log,
                    }
            else:
                with ThreadPoolExecutor(
                    max_workers=effective_workers,
                    thread_name_prefix=f"merge-l{level_num}",
                ) as executor:
                    future_map = {}
                    for group_idx, group, members, output_label in synthesis_jobs:
                        logger.info(
                            "  Queue synth group %d: %d graph(s), %d total input node(s)",
                            group_idx,
                            len(members),
                            sum(len(b.graph.nodes) for b in members),
                        )
                        future = executor.submit(
                            _synthesize_group,
                            members,
                            output_label=output_label,
                        )
                        future_map[future] = (group_idx, group)

                    for future in as_completed(future_map):
                        group_idx, group = future_map[future]
                        merged_bundle, merge_log = future.result()
                        next_level[group_idx] = merged_bundle
                        level_merges[group_idx] = {
                            "mode": "direct_graph_synthesis",
                            "group_index": group_idx,
                            "members": group,
                            **merge_log,
                        }
                        logger.info(
                            "  Completed group %d → %d nodes, %d edges",
                            group_idx,
                            len(merged_bundle.graph.nodes),
                            len(merged_bundle.graph.edges),
                        )

        if any(bundle is None for bundle in next_level) or any(log is None for log in level_merges):
            raise RuntimeError(f"Level {level_num} finished with incomplete synthesis results.")

        finalized_next_level = [bundle for bundle in next_level if bundle is not None]
        finalized_level_merges = [log for log in level_merges if log is not None]

        # Save this level's results
        level_dir = output_dir / f"level_{level_num}"
        level_dir.mkdir(parents=True, exist_ok=True)

        for idx, bundle in enumerate(finalized_next_level):
            bundle.graph.save(level_dir / f"subgraph_{idx}")

        (level_dir / "merge_logs.json").write_text(
            json.dumps(finalized_level_merges, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (level_dir / "clustering.json").write_text(
            json.dumps(clustering_log, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (level_dir / "subgraph_metadata.json").write_text(
            json.dumps([
                {
                    "index": idx,
                    "label": bundle.label,
                    "source_tasks": bundle.task_ids,
                    "n_nodes": len(bundle.graph.nodes),
                    "n_edges": len(bundle.graph.edges),
                }
                for idx, bundle in enumerate(finalized_next_level)
            ], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        trace.record_level(level_num, clustering_log, finalized_level_merges, finalized_next_level)

        current_level = finalized_next_level
        level_num += 1

    final_graph = current_level[0].graph
    logger.info("Tree merge complete: %d nodes, %d edges after %d levels",
                 len(final_graph.nodes), len(final_graph.edges), level_num)

    return final_graph, trace


def _cluster_level(
    bundles: list[GraphBundle],
    *,
    cluster_capacity: int,
) -> tuple[list[list[int]], dict]:
    """Cluster current-level graph bundles into merge groups."""
    prompt = CLUSTER_LEVEL_USER.format(
        graph_packets="\n\n".join(
            format_graph_packet(
                index=i,
                task_descriptions=bundle.task_descriptions,
                graph_summary=bundle.graph.summary(),
            )
            for i, bundle in enumerate(bundles)
        )
    )

    data, cost = call_llm_json(prompt, system=CLUSTER_LEVEL_SYSTEM)
    groups, validation_log = _validate_cluster_groups(data, len(bundles), cluster_capacity)
    return groups, {
        "mode": "global",
        "cost": cost,
        "raw_output": data,
        "validated_groups": groups,
        "validation": validation_log,
    }


def _validate_cluster_groups(
    raw_output: dict | list | None,
    n_graphs: int,
    cluster_capacity: int,
) -> tuple[list[list[int]], list[dict]]:
    """Validate and repair LLM cluster output into a full partition."""
    validation_log: list[dict] = []
    raw_groups: list = []
    if isinstance(raw_output, dict):
        raw_groups = raw_output.get("groups", [])
    elif isinstance(raw_output, list):
        raw_groups = raw_output

    groups: list[list[int]] = []
    used: set[int] = set()

    for raw_group in raw_groups if isinstance(raw_groups, list) else []:
        members = raw_group.get("members") if isinstance(raw_group, dict) else raw_group
        if not isinstance(members, list):
            validation_log.append({"type": "skip_group", "reason": "members_not_list", "raw": raw_group})
            continue
        cleaned: list[int] = []
        seen_local: set[int] = set()
        for idx in members:
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= n_graphs:
                continue
            if idx in used or idx in seen_local:
                continue
            seen_local.add(idx)
            cleaned.append(idx)
        if not cleaned:
            validation_log.append({"type": "skip_group", "reason": "empty_after_clean", "raw": raw_group})
            continue
        if len(cleaned) > cluster_capacity:
            for offset in range(0, len(cleaned), cluster_capacity):
                chunk = cleaned[offset:offset + cluster_capacity]
                groups.append(chunk)
                used.update(chunk)
                validation_log.append({
                    "type": "split_oversized_group",
                    "original": cleaned,
                    "chunk": chunk,
                })
            continue
        groups.append(cleaned)
        used.update(cleaned)

    missing = [idx for idx in range(n_graphs) if idx not in used]
    for idx in missing:
        groups.append([idx])
        validation_log.append({"type": "add_missing_singleton", "index": idx})

    if all(len(group) == 1 for group in groups) and n_graphs > 1:
        raise ValueError(
            "Clustering produced only singleton groups; no merge progress is possible at this level."
        )

    return groups, validation_log


def _synthesize_group(
    members: list[GraphBundle],
    *,
    output_label: str,
) -> tuple[GraphBundle, dict]:
    """Directly synthesize one merged graph from a small cluster of graphs."""
    prompt = DIRECT_GRAPH_SYNTHESIS_USER.format(
        graph_packets="\n\n".join(
            format_graph_packet(
                index=i,
                task_descriptions=bundle.task_descriptions,
                graph_summary=bundle.graph.summary(),
            )
            for i, bundle in enumerate(members)
        )
    )

    total_cost = 0.0
    last_data: dict | list | None = None
    last_validation: list[dict] = []
    synth_retries = 2
    current_prompt = prompt

    for attempt in range(synth_retries + 1):
        data, cost = call_llm_json(current_prompt, system=DIRECT_GRAPH_SYNTHESIS_SYSTEM)
        total_cost += cost
        last_data = data
        merged_graph, validation = _build_graph_from_synthesis_output(data)
        last_validation = validation
        node_validation_errors = _has_invalid_node_validation(validation)
        empty_graph = _is_empty_graph_output(merged_graph, validation)
        if merged_graph.nodes and not node_validation_errors:
            break
        if attempt < synth_retries:
            if empty_graph:
                logger.warning(
                    "Direct graph synthesis for %s returned an empty graph (attempt %d/%d), retrying with the same prompt",
                    output_label,
                    attempt + 1,
                    synth_retries + 1,
                )
                current_prompt = _build_empty_graph_retry_prompt(prompt)
            else:
                logger.warning(
                    "Direct graph synthesis for %s failed node schema validation (attempt %d/%d), retrying with the same prompt",
                    output_label,
                    attempt + 1,
                    synth_retries + 1,
                )
        else:
            raw_output_preview = _preview_raw_output(last_data)
            if empty_graph:
                logger.error(
                    "Direct graph synthesis for %s returned an empty graph after %d attempts. Raw output: %s",
                    output_label,
                    synth_retries + 1,
                    raw_output_preview,
                )
                raise ValueError(
                    f"Direct graph synthesis returned an empty graph for {output_label}. "
                    f"Validation: {validation[:6]} Raw output: {raw_output_preview}"
                )
            logger.error(
                "Direct graph synthesis for %s returned invalid node schema after %d attempts. "
                "Validation: %s Raw output: %s",
                output_label,
                synth_retries + 1,
                validation[:6],
                raw_output_preview,
            )
            raise ValueError(
                f"Direct graph synthesis returned invalid node schema for {output_label}. "
                f"Validation: {validation[:6]} Raw output: {raw_output_preview}"
            )

    merged_bundle = GraphBundle(
        graph=merged_graph,
        task_ids=_merge_task_ids(members),
        task_descriptions=_merge_task_descriptions(members),
        label=output_label,
    )
    return merged_bundle, {
        "source_tasks": _merge_task_ids(members),
        "input_snapshots": [_bundle_snapshot(bundle) for bundle in members],
        "cost": total_cost,
        "raw_output": last_data,
        "validation": last_validation,
        "result_snapshot": _bundle_snapshot(merged_bundle),
        "fallback": None,
    }


def _has_invalid_node_validation(validation_log: list[dict]) -> bool:
    """Return True if synthesis output had any node-level schema failures."""
    node_error_types = {"invalid_payload", "invalid_nodes", "skip_node"}
    return any(
        isinstance(entry, dict) and entry.get("type") in node_error_types
        for entry in validation_log
    )


def _is_empty_graph_output(graph: SkillGraph, validation_log: list[dict]) -> bool:
    """Return True when synthesis produced an empty graph without node schema errors."""
    return not graph.nodes and not _has_invalid_node_validation(validation_log)


def _build_empty_graph_retry_prompt(base_prompt: str) -> str:
    """Add a targeted correction when a synthesis attempt returns an empty graph."""
    return (
        base_prompt
        + "\n\nIMPORTANT RETRY INSTRUCTION:\n"
        + "Your previous output was an empty graph. This is invalid and unreasonable.\n"
        + "Please try again.\n"
        + "If strong merges are not possible, preserve the relevant source nodes and "
          "their important structure instead of collapsing to zero nodes.\n"
        + "Return a NON-EMPTY merged graph JSON.\n"
    )


def _preview_raw_output(data: dict | list | None, limit: int = 1200) -> str:
    """Render a compact preview of raw synthesized output for debugging."""
    if data is None:
        return "<none>"
    try:
        text = json.dumps(data, ensure_ascii=False)
    except Exception:
        text = repr(data)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _build_graph_from_synthesis_output(data: dict | list | None) -> tuple[SkillGraph, list[dict]]:
    """Validate LLM synthesis output and materialize a SkillGraph."""
    graph = SkillGraph()
    validation_log: list[dict] = []
    if not isinstance(data, dict):
        validation_log.append({"type": "invalid_payload", "reason": "not_a_dict"})
        return graph, validation_log

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    node_id_map: dict[str, str] = {}

    if not isinstance(nodes, list):
        validation_log.append({"type": "invalid_nodes", "reason": "nodes_not_list"})
        return graph, validation_log

    for raw_node in nodes:
        if not isinstance(raw_node, dict):
            validation_log.append({"type": "skip_node", "reason": "not_an_object"})
            continue
        raw_id = str(raw_node.get("node_id") or "").strip()
        summary = str(raw_node.get("summary") or "").strip()
        description = str(raw_node.get("description") or "").strip()
        triggers = raw_node.get("triggers") or []
        body = str(raw_node.get("body") or "").strip()
        if not raw_id:
            validation_log.append({"type": "skip_node", "reason": "missing_node_id", "raw": raw_node})
            continue
        node_id = _slugify_node_id(raw_id)
        if not summary:
            summary = description or raw_id.replace("-", " ")
        if not description:
            description = summary
        if not isinstance(triggers, list):
            triggers = []
        cleaned_triggers = [str(item).strip() for item in triggers if str(item).strip()]
        markdown = _build_node_markdown(
            node_id=node_id,
            summary=summary,
            description=description,
            triggers=cleaned_triggers,
            body=body,
        )
        unique_id = _add_node_with_unique_id(graph, node_id, markdown)
        node_id_map[raw_id] = unique_id
        node_id_map[node_id] = unique_id

    if not isinstance(edges, list):
        validation_log.append({"type": "invalid_edges", "reason": "edges_not_list"})
        edges = []

    valid_ids = set(graph.node_ids())
    for raw_edge in edges:
        if not isinstance(raw_edge, dict):
            validation_log.append({"type": "skip_edge", "reason": "not_an_object"})
            continue
        raw_source = str(raw_edge.get("source") or "").strip()
        raw_target = str(raw_edge.get("target") or "").strip()
        condition = " ".join(str(raw_edge.get("condition") or "").split()).strip()
        source = node_id_map.get(raw_source, raw_source)
        target = node_id_map.get(raw_target, raw_target)
        if not source or not target or source == target or not condition:
            validation_log.append({"type": "skip_edge", "reason": "invalid_edge", "raw": raw_edge})
            continue
        if source not in valid_ids or target not in valid_ids:
            validation_log.append({"type": "skip_edge", "reason": "unknown_node_ref", "raw": raw_edge})
            continue
        graph.add_edge(source, target, condition)

    return graph, validation_log


def _merge_task_ids(members: list[GraphBundle]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for bundle in members:
        for task_id in bundle.task_ids:
            if task_id in seen:
                continue
            seen.add(task_id)
            merged.append(task_id)
    return merged


def _merge_task_descriptions(members: list[GraphBundle]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for bundle in members:
        for desc in bundle.task_descriptions:
            text = " ".join(desc.split()).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(text)
    return merged


def _bundle_snapshot(bundle: GraphBundle) -> dict:
    return {
        "label": bundle.label,
        "source_tasks": bundle.task_ids,
        "n_nodes": len(bundle.graph.nodes),
        "n_edges": len(bundle.graph.edges),
        "node_ids": bundle.graph.node_ids(),
    }


def _build_node_markdown(
    *,
    node_id: str,
    summary: str,
    description: str,
    triggers: list[str],
    body: str,
) -> str:
    body = body.strip()
    if not body:
        body = f"## When to Use\n\n{description}\n\n## Strategy\n\n{summary}\n"
    trigger_block = "\n".join(f'  - {_yaml_quote(trigger)}' for trigger in triggers) if triggers else '  - "(none)"'
    return f"""\
---
name: {node_id}
summary: {_yaml_quote(summary)}
description: {_yaml_quote(description)}
triggers:
{trigger_block}
---

# {node_id}

{body}
""".strip() + "\n"


def _yaml_quote(value: str) -> str:
    value = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'


def _slugify_node_id(raw: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", raw.strip().lower()).strip("-")
    return slug or "merged-node"


def _align_graph_pair(graph_a: SkillGraph, graph_b: SkillGraph) -> dict:
    """Ask the LLM to align two graphs and return merge pairs."""
    prompt = GRAPH_ALIGN_USER.format(
        graph_a_summary=format_graph_summary("Graph A", graph_a.nodes, graph_a.edges),
        graph_b_summary=format_graph_summary("Graph B", graph_b.nodes, graph_b.edges),
    )

    data, cost = call_llm_json(prompt, system=GRAPH_ALIGN_SYSTEM)
    if not isinstance(data, dict):
        logger.warning("Graph alignment: expected dict, got %s", type(data).__name__)
        return {"merge_pairs": []}

    pairs = data.get("merge_pairs", [])
    if not isinstance(pairs, list):
        logger.warning("Graph alignment: merge_pairs is not a list")
        return {"merge_pairs": []}

    logger.info("  Alignment judge returned %d merge pair(s) (cost=$%.4f)", len(pairs), cost)
    return {"merge_pairs": pairs}


def _validate_merge_pairs(alignment: dict, graph_a: SkillGraph, graph_b: SkillGraph) -> tuple[list[dict], list[dict]]:
    """Validate graph alignment output and enforce one-to-one merge pairs."""
    validated: list[dict] = []
    skipped: list[dict] = []
    used_a: set[str] = set()
    used_b: set[str] = set()

    for raw_pair in alignment.get("merge_pairs", []):
        if not isinstance(raw_pair, dict):
            skipped.append({"raw": raw_pair, "reason": "not_an_object"})
            continue

        a_id = raw_pair.get("a_node")
        b_id = raw_pair.get("b_node")
        reason = raw_pair.get("reason", "")

        if not a_id or not b_id:
            skipped.append({"raw": raw_pair, "reason": "missing_node_id"})
            continue
        if a_id not in graph_a.nodes or b_id not in graph_b.nodes:
            skipped.append({"raw": raw_pair, "reason": "unknown_node_id"})
            continue
        if a_id in used_a or b_id in used_b:
            skipped.append({"raw": raw_pair, "reason": "violates_one_to_one"})
            continue

        used_a.add(a_id)
        used_b.add(b_id)
        validated.append({"a_node": a_id, "b_node": b_id, "reason": reason})

    return validated, skipped


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════


def _execute_merge(md_a: str, md_b: str) -> str:
    """Use LLM to merge two node markdowns into a more general version."""
    prompt = MERGE_EXECUTOR_USER.format(
        node_a_markdown=md_a,
        node_b_markdown=md_b,
    )

    for attempt in range(2):
        resp = call_llm(prompt, system=MERGE_EXECUTOR_SYSTEM)
        result = resp.text.strip()

        if result.startswith("```"):
            match = re.search(r"```(?:markdown)?\s*\n(.*?)\n```", result, re.DOTALL)
            if match:
                result = match.group(1).strip()

        if result.startswith("---"):
            return result

        logger.warning(
            "Merge executor attempt %d/2 returned invalid markdown; retrying",
            attempt + 1,
        )

    logger.warning("Merge executor failed twice; using deterministic fallback merge")
    return _fallback_merge_markdown(md_a, md_b)


def _reconcile_cross_task_edges(graph: SkillGraph, remapped_edges: list[dict]) -> list[dict]:
    """Reconcile remapped cross-task edges into a minimal semantically coherent set."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for edge in remapped_edges:
        key = (edge["source"], edge["target"])
        grouped[key].append(edge)

    reconciliation_log: list[dict] = []

    for (source, target), candidates in sorted(grouped.items()):
        unique_candidates = _dedupe_edge_candidates(candidates)
        if len(unique_candidates) == 1:
            edge = unique_candidates[0]
            graph.add_edge(source, target, edge["condition"])
            reconciliation_log.append({
                "source": source,
                "target": target,
                "candidates": unique_candidates,
                "final_edges": [{
                    "condition": edge["condition"],
                    "reason": "single_candidate",
                }],
            })
            continue

        context = format_edge_reconcile_context(
            source,
            graph.get_node(source) or "",
            target,
            graph.get_node(target) or "",
            unique_candidates,
        )
        final_edges = _call_edge_reconcile_with_retry(
            context=context,
            source=source,
            target=target,
        )
        if not final_edges:
            final_edges = _fallback_edge_reconcile(unique_candidates)

        for edge in final_edges:
            graph.add_edge(source, target, edge["condition"])

        reconciliation_log.append({
            "source": source,
            "target": target,
            "candidates": unique_candidates,
            "final_edges": final_edges,
        })
        logger.info(
            "    EDGE RECONCILE: %s -> %s from %d candidate(s) to %d final edge(s)",
            source, target, len(unique_candidates), len(final_edges),
        )

    return reconciliation_log


def _dedupe_edge_candidates(candidates: list[dict]) -> list[dict]:
    """Remove exact duplicate edge candidates before reconciliation."""
    deduped: list[dict] = []
    seen: set[str] = set()
    for edge in candidates:
        key = " ".join((edge.get("condition") or "").split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append({
            "source": edge["source"],
            "target": edge["target"],
            "condition": key,
            "origin": edge.get("origin", "?"),
        })
    return deduped


def _fallback_edge_reconcile(candidates: list[dict]) -> list[dict]:
    """Conservative fallback when semantic edge reconciliation fails."""
    ranked = sorted(
        candidates,
        key=lambda edge: len(edge.get("condition", "")),
    )
    best = ranked[0]
    return [{
        "condition": best["condition"],
        "reason": "fallback_best_candidate",
    }]


def _call_edge_reconcile_with_retry(
    *,
    context: str,
    source: str,
    target: str,
    max_attempts: int = 2,
) -> list[dict]:
    """Call the edge reconcile judge with retries before falling back."""
    prompt = EDGE_RECONCILE_USER.format(edge_context=context)

    for attempt in range(max_attempts):
        data, cost = call_llm_json(prompt, system=EDGE_RECONCILE_SYSTEM)

        if isinstance(data, dict) and isinstance(data.get("edges", []), list):
            final_edges: list[dict] = []
            for raw in data.get("edges", []):
                if not isinstance(raw, dict):
                    continue
                condition = (raw.get("condition") or "").strip()
                reason = (raw.get("reason") or "").strip()
                if not condition:
                    continue
                final_edges.append({
                    "condition": condition,
                    "reason": reason,
                })
            if final_edges:
                return final_edges

        logger.warning(
            "Edge reconcile attempt %d/%d returned invalid payload for %s -> %s (cost=$%.4f); retrying",
            attempt + 1,
            max_attempts,
            source,
            target,
            cost,
        )

    logger.warning(
        "Edge reconcile failed after %d attempt(s) for %s -> %s; using conservative fallback",
        max_attempts,
        source,
        target,
    )
    return []


def _fallback_merge_markdown(md_a: str, md_b: str) -> str:
    """Deterministically merge two nodes when the LLM executor fails twice.

    This fallback is intentionally symmetric: it preserves information from both
    inputs instead of biasing toward one side.
    """
    fields_a, _ = _split_frontmatter(md_a)
    fields_b, _ = _split_frontmatter(md_b)

    name_a = fields_a.get("name") or "node-a"
    name_b = fields_b.get("name") or "node-b"
    summary_a = fields_a.get("summary") or fields_a.get("description") or ""
    summary_b = fields_b.get("summary") or fields_b.get("description") or ""
    description_a = fields_a.get("description") or summary_a
    description_b = fields_b.get("description") or summary_b
    triggers_a = fields_a.get("triggers") or []
    triggers_b = fields_b.get("triggers") or []

    merged_name = _fallback_node_id(str(name_a), str(name_b))
    merged_summary = _merge_scalar_text(str(summary_a), str(summary_b))
    merged_description = _merge_scalar_text(str(description_a), str(description_b))
    merged_triggers = _merge_string_lists(triggers_a, triggers_b)

    when_text = _merge_section_text(
        _extract_section(md_a, ("When to Use", "Situation", "Context")),
        _extract_section(md_b, ("When to Use", "Situation", "Context")),
    )
    strategy_text = _merge_section_text(
        _extract_section(md_a, ("Strategy", "Approach", "Method")),
        _extract_section(md_b, ("Strategy", "Approach", "Method")),
    )
    actions_text = _merge_action_sections(
        _extract_section(md_a, ("Actions",)),
        _extract_section(md_b, ("Actions",)),
    )
    checks_text = _merge_section_text(
        _extract_section(md_a, ("Checks", "Verification")),
        _extract_section(md_b, ("Checks", "Verification")),
    )
    pitfalls_text = _merge_action_sections(
        _extract_section(md_a, ("Pitfalls",)),
        _extract_section(md_b, ("Pitfalls",)),
    )

    def yaml_quote(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    triggers_yaml = "\n".join(f"  - {yaml_quote(t)}" for t in merged_triggers) if merged_triggers else '  - "(none)"'

    body_sections = []
    if when_text:
        body_sections.append(("When to Use", when_text))
    if strategy_text:
        body_sections.append(("Strategy", strategy_text))
    if actions_text:
        body_sections.append(("Actions", actions_text))
    if checks_text:
        body_sections.append(("Checks", checks_text))
    if pitfalls_text:
        body_sections.append(("Pitfalls", pitfalls_text))

    body_text = "\n\n".join(
        f"## {heading}\n\n{content}".strip()
        for heading, content in body_sections
        if content.strip()
    )

    return f"""\
---
name: {merged_name}
summary: {yaml_quote(merged_summary or merged_description or merged_name)}
description: {yaml_quote(merged_description or merged_summary or merged_name)}
triggers:
{triggers_yaml}
---

# {merged_name}

{body_text}
""".strip() + "\n"


def _split_frontmatter(markdown: str) -> tuple[dict[str, object], str]:
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
    """Unquote a simple YAML scalar written as a single-line string."""
    if len(value) >= 2 and value[0] == value[-1] == '"':
        inner = value[1:-1]
        return inner.replace('\\"', '"').replace("\\\\", "\\")
    return value


def _extract_section(markdown: str, headings: tuple[str, ...]) -> str:
    """Extract a section body by heading name."""
    for heading in headings:
        pattern = rf"## {re.escape(heading)}\s*\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, markdown, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _merge_scalar_text(a: str, b: str) -> str:
    """Merge two short scalar strings conservatively."""
    a = " ".join(a.split()).strip()
    b = " ".join(b.split()).strip()
    if not a:
        return b
    if not b:
        return a
    if a.casefold() == b.casefold():
        return a
    if a.casefold() in b.casefold():
        return b
    if b.casefold() in a.casefold():
        return a
    return f"{a} / {b}"


def _merge_string_lists(a: list, b: list) -> list[str]:
    """Merge two string lists while preserving order and deduplicating loosely."""
    merged: list[str] = []
    seen: set[str] = set()
    for item in list(a) + list(b):
        text = " ".join(str(item).split()).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(text)
    return merged


def _merge_section_text(a: str, b: str) -> str:
    """Merge two prose sections without discarding either side."""
    parts = []
    for text in (a, b):
        text = text.strip()
        if not text:
            continue
        if any(text.casefold() == existing.casefold() for existing in parts):
            continue
        parts.append(text)
    return "\n\n".join(parts)


def _merge_action_sections(a: str, b: str) -> str:
    """Merge sections that are usually bullet lists."""
    bullets = _extract_bullets(a) + _extract_bullets(b)
    merged_bullets = _merge_string_lists(bullets, [])
    if merged_bullets:
        return "\n".join(f"- {bullet}" for bullet in merged_bullets)
    return _merge_section_text(a, b)


def _extract_bullets(section_text: str) -> list[str]:
    """Extract top-level markdown bullets from a section."""
    bullets: list[str] = []
    for line in section_text.splitlines():
        match = re.match(r"^\s*-\s+(.*)$", line)
        if match:
            bullets.append(match.group(1).strip())
    return bullets


def _fallback_node_id(name_a: str, name_b: str) -> str:
    """Create a symmetric fallback node id from two source node ids."""
    tokens_a = [tok for tok in name_a.split("-") if tok]
    tokens_b = [tok for tok in name_b.split("-") if tok]
    common = [tok for tok in tokens_a if tok in set(tokens_b)]
    if len(common) >= 2:
        return "-".join(common)
    ordered = sorted({name_a, name_b})
    base = "-".join(ordered)
    base = re.sub(r"[^a-zA-Z0-9-]+", "-", base).strip("-").lower()
    return base[:80] if base else "merged-skill"


def _extract_node_id(markdown: str, fallback: str) -> str:
    """Extract the node name/slug from YAML frontmatter."""
    match = re.search(r"^name:\s*(.+)$", markdown, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback


def _add_node_with_unique_id(graph: SkillGraph, node_id: str, markdown: str) -> str:
    """Add a node to the graph while avoiding silent overwrites on slug collisions."""
    unique_id = _make_unique_node_id(node_id, set(graph.node_ids()))
    if unique_id != node_id:
        markdown = _rename_node_markdown(markdown, unique_id)
    graph.add_node(unique_id, markdown)
    return unique_id


def _make_unique_node_id(base_id: str, existing_ids: set[str]) -> str:
    """Return a collision-free node id."""
    if base_id not in existing_ids:
        return base_id

    suffix = 2
    while f"{base_id}-{suffix}" in existing_ids:
        suffix += 1
    return f"{base_id}-{suffix}"


def _rename_node_markdown(markdown: str, node_id: str) -> str:
    """Keep markdown frontmatter/title aligned with the graph's internal node id."""
    markdown = re.sub(r"^name:\s*.+$", f"name: {node_id}", markdown, count=1, flags=re.MULTILINE)
    markdown = re.sub(r"^#\s+.+$", f"# {node_id}", markdown, count=1, flags=re.MULTILINE)
    return markdown
