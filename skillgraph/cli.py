"""CLI entry point for the Skill Graph pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from skillgraph.config import TRAJECTORIES_DIR, SKILLS_DIR, CACHE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill Graph Construction Pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # ── build ──
    p_build = sub.add_parser("build", help="Build skill graph from trajectories")
    p_build.add_argument(
        "--task", action="append", default=None,
        help="Process specific task(s). Can be repeated.",
    )
    p_build.add_argument("-t", "--trajectories", type=Path, default=TRAJECTORIES_DIR)
    p_build.add_argument("-o", "--output", type=Path, default=SKILLS_DIR)
    p_build.add_argument("-m", "--model", type=str, default=None, help="LLM model (default: from config)")
    p_build.add_argument("--no-cache", action="store_true")

    # ── inspect ──
    p_inspect = sub.add_parser("inspect", help="Inspect intermediate results")
    p_inspect.add_argument("--task", type=str)
    p_inspect.add_argument(
        "--step", choices=["2a", "2b", "2c", "graph"], default="graph",
    )

    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "build":
        _run_build(args)
    elif args.command == "inspect":
        _run_inspect(args)
    else:
        parser.print_help()


def _run_build(args):
    """Run the full pipeline: parse → extract → aggregate → draft → compare → link → assemble."""
    import skillgraph.config as cfg

    # Override model if specified
    if args.model:
        cfg.MODEL = args.model

    from skillgraph.assemble.assembler import assemble, load_existing_nodes
    from skillgraph.compare.comparator import compare_and_merge
    from skillgraph.draft.node_drafter import generate_draft_nodes
    from skillgraph.extract.segment_extractor import extract_segments
    from skillgraph.extract.task_aggregator import aggregate_task
    from skillgraph.link.link_builder import build_links
    from skillgraph.parse.trajectory_parser import parse_all_trajectories

    logger = logging.getLogger("skillgraph.cli")
    use_cache = not args.no_cache

    # Step 1: Parse
    logger.info("Step 1: Parsing trajectories from %s", args.trajectories)
    all_records = parse_all_trajectories(args.trajectories)

    # Group by task
    tasks: dict[str, list] = {}
    for record in all_records:
        tasks.setdefault(record.task_id, []).append(record)

    # Filter tasks if specified
    if args.task:
        tasks = {k: v for k, v in tasks.items() if k in args.task}

    logger.info(
        "Processing %d tasks (%d trajectories)",
        len(tasks),
        sum(len(v) for v in tasks.values()),
    )

    # Load existing graph (for incremental processing)
    existing_nodes = load_existing_nodes(args.output)
    all_task_summaries = []
    item_to_node_map: dict[str, str] = {}
    # Track node ID renames caused by merges (old_id → new_id)
    node_id_redirects: dict[str, str] = {}

    # ── Process each task ──
    for task_id, records in sorted(tasks.items()):
        logger.info("=" * 60)
        logger.info("Task: %s (%d trajectories)", task_id, len(records))

        # Step 2a: Per-trajectory segment extraction
        logger.info("  Step 2a: Extracting segments")
        analyses = []
        for record in records:
            analysis = extract_segments(record, use_cache=use_cache)
            analyses.append(analysis)
            logger.info(
                "    %s: %d segments (%s)",
                record.trajectory_id[:40],
                len(analysis.segments),
                "resolved" if record.resolved else "failed",
            )

        # Step 2b: Cross-trajectory aggregation
        logger.info("  Step 2b: Aggregating across trajectories")
        summary = aggregate_task(task_id, analyses, use_cache=use_cache)
        logger.info("    %d items, %d relationships", len(summary.items), len(summary.relationships))
        all_task_summaries.append(summary)

        # Step 2c: Generate draft nodes
        logger.info("  Step 2c: Generating draft nodes")
        drafts = generate_draft_nodes(summary, use_cache=use_cache)
        logger.info("    %d drafts", len(drafts))

        # Step 3: Compare + merge
        logger.info("  Step 3: Compare + merge (%d existing nodes)", len(existing_nodes))
        for draft in drafts:
            result_node, merged_with = compare_and_merge(draft, existing_nodes)

            # Track item → node mapping
            for item_id in draft.source_item_ids:
                item_to_node_map[item_id] = result_node.node_id

            if merged_with:
                existing_nodes = [
                    n for n in existing_nodes if n.node_id != merged_with.node_id
                ]
                existing_nodes.append(result_node)
                # Track rename so existing wikilinks can be updated
                if merged_with.node_id != result_node.node_id:
                    node_id_redirects[merged_with.node_id] = result_node.node_id
                    # Also update item_to_node_map entries that pointed to the old ID
                    for iid, nid in item_to_node_map.items():
                        if nid == merged_with.node_id:
                            item_to_node_map[iid] = result_node.node_id
                logger.info("    Merged '%s' → '%s'", draft.node_id, result_node.node_id)
            else:
                existing_nodes.append(result_node)
                logger.info("    New: '%s'", result_node.node_id)

    # Apply node ID redirects: update wikilinks in all nodes that point to renamed nodes
    if node_id_redirects:
        logger.info("Updating %d wikilink redirects", len(node_id_redirects))
        existing_nodes = _apply_redirects(existing_nodes, node_id_redirects)

    # Step 4: Build links
    logger.info("=" * 60)
    logger.info("Step 4: Building links (%d nodes)", len(existing_nodes))
    linked_nodes = build_links(existing_nodes, all_task_summaries, item_to_node_map)

    # Step 5: Assemble
    logger.info("Step 5: Assembling output")
    output_dir = assemble(linked_nodes, args.output)

    logger.info("=" * 60)
    logger.info("Done! %d nodes → %s", len(linked_nodes), output_dir)


def _apply_redirects(nodes: list, redirects: dict[str, str]) -> list:
    """Update [[wikilinks]] in all node markdown to reflect renamed node IDs."""
    import re
    updated = []
    for node in nodes:
        md = node.markdown
        for old_id, new_id in redirects.items():
            md = re.sub(
                r"\[\[" + re.escape(old_id) + r"\]\]",
                f"[[{new_id}]]",
                md,
            )
        if md != node.markdown:
            from skillgraph.models import SkillNode
            node = SkillNode(
                node_id=node.node_id,
                type=node.type,
                markdown=md,
                source_tasks=node.source_tasks,
            )
        updated.append(node)
    return updated


def _run_inspect(args):
    """Inspect intermediate results."""
    if args.step == "2a" and args.task:
        cache_dir = CACHE_DIR / "step2a"
        for f in sorted(cache_dir.glob(f"{args.task}*.json")):
            data = json.loads(f.read_text())
            status = "RESOLVED" if data["resolved"] else "FAILED"
            print(f"\n{'=' * 60}")
            print(f"Trajectory: {data['trajectory_id']} ({status})")
            for seg in data["segments"]:
                print(f"  [{seg['type']}] {seg['title']} (order: {seg['order']})")
                print(f"    {seg['description'][:150]}")

    elif args.step == "2b" and args.task:
        cache_file = CACHE_DIR / "step2b" / f"{args.task}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            print(f"Task: {data['task_id']}")
            for item in data["items"]:
                print(f"  [{item['type']}] {item['title']}")
                print(f"    {item['description'][:150]}")
            for rel in data.get("relationships", []):
                print(f"  REL: {rel['rel_type']}: {rel['source_item_id']} → {rel['target_item_id']}")
        else:
            print(f"No cache found for {args.task}")

    elif args.step == "graph":
        state_file = CACHE_DIR / "graph_state.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            nodes = data.get("nodes", [])
            print(f"Graph: {len(nodes)} nodes")
            for n in nodes:
                tasks_str = ", ".join(n.get("source_tasks", []))
                print(f"  [{n['type']}] {n['node_id']} (tasks: {tasks_str})")
        else:
            print("No graph state. Run 'build' first.")

    else:
        print("Usage: skillgraph inspect --task <task_id> --step <2a|2b|2c|graph>")


if __name__ == "__main__":
    main()
