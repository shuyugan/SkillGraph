"""CLI entry point for the Skill Graph pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path

from skillgraph.config import TRAJECTORIES_DIR, DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill Graph Construction Pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # ── build (intra-task) ──
    p_build = sub.add_parser("build", help="Build per-task skill graphs from trajectories")
    p_build.add_argument(
        "--task", action="append", default=None,
        help="Process specific task(s). Can be repeated.",
    )
    p_build.add_argument("-t", "--trajectories", type=Path, default=TRAJECTORIES_DIR)
    p_build.add_argument("-m", "--model", type=str, default=None)
    p_build.add_argument("--clean", action="store_true", help="Clear output before building.")
    p_build.add_argument(
        "--task-workers",
        type=int,
        default=1,
        help="Number of tasks to build in parallel. Default: 1 (sequential).",
    )

    # ── merge (cross-task) ──
    p_merge = sub.add_parser("merge", help="Merge per-task graphs into repo-level domain graphs")
    p_merge.add_argument("-m", "--model", type=str, default=None)
    p_merge.add_argument(
        "-t", "--trajectories", type=Path, default=None,
        help="Optional trajectory directory for loading task descriptions used during clustering.",
    )
    p_merge.add_argument(
        "--domain", action="append", default=None,
        help="Merge only specific repo-domain(s), e.g. django or sympy. Can be repeated.",
    )
    p_merge.add_argument(
        "--cluster-capacity",
        type=int,
        default=4,
        help="Maximum graphs per merge group during level-wise clustering (recommended: 2-4).",
    )
    p_merge.add_argument(
        "--merge-workers",
        type=int,
        default=4,
        help="Number of cluster synthesis jobs to run in parallel per merge level. Default: 4.",
    )
    p_merge.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the highest completed merge level in data/domain_graphs/<domain>/ if present.",
    )
    p_merge.add_argument("--clean", action="store_true", help="Clear merge output for the selected domain(s) before running.")

    # ── text-baseline ──
    p_text = sub.add_parser("text-baseline", help="Build file-based markdown skill baseline from trajectories")
    p_text.add_argument(
        "--task", action="append", default=None,
        help="Process specific task(s). Can be repeated.",
    )
    p_text.add_argument(
        "--domain", action="append", default=None,
        help="Restrict to specific repo-domain(s), e.g. django or sympy. Can be repeated.",
    )
    p_text.add_argument("-t", "--trajectories", type=Path, default=TRAJECTORIES_DIR)
    p_text.add_argument("-m", "--model", type=str, default=None)
    p_text.add_argument("--clean", action="store_true", help="Clear text-baseline output before building.")
    p_text.add_argument(
        "--task-workers",
        type=int,
        default=1,
        help="Number of task memos to build in parallel. Default: 1 (sequential).",
    )

    # ── inspect ──
    p_inspect = sub.add_parser("inspect", help="Inspect built graph")
    p_inspect.add_argument("--task", type=str, help="Task ID (omit for global graph)")
    p_inspect.add_argument("--domain", type=str, help="Repo-domain ID (e.g. django)")
    p_inspect.add_argument("--global", dest="show_global", action="store_true", help="Legacy alias; lists domain graphs")

    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "build":
        _run_build(args)
    elif args.command == "merge":
        _run_merge(args)
    elif args.command == "text-baseline":
        _run_text_baseline(args)
    elif args.command == "inspect":
        _run_inspect(args)
    else:
        parser.print_help()


def _run_build(args):
    """Run the intra-task pipeline: initializer → success updates → failure updates."""
    import shutil

    import skillgraph.config as cfg

    if args.model:
        cfg.MODEL = args.model

    from skillgraph.graph.builder import (
        BuildTrace,
        failure_update,
        normalize_task_granularity,
        reconcile_task_graph,
        initialize_graph,
        graph_snapshot,
        success_update,
    )
    from skillgraph.parse.trajectory_parser import parse_all_trajectories

    logger = logging.getLogger("skillgraph.cli")

    output_dir = DATA_DIR / "graphs"

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("Cleaned output: %s", output_dir)

    # ── Step 1: Parse ──
    logger.info("Step 1: Parsing trajectories from %s", args.trajectories)
    all_records = parse_all_trajectories(args.trajectories)

    # Group by task
    tasks: dict[str, list] = {}
    for record in all_records:
        tasks.setdefault(record.task_id, []).append(record)
    if args.task:
        tasks = {k: v for k, v in tasks.items() if k in args.task}

    logger.info(
        "Processing %d tasks (%d trajectories)",
        len(tasks), sum(len(v) for v in tasks.values()),
    )

    task_workers = max(1, int(args.task_workers or 1))

    def build_single_task(task_id: str, records: list) -> tuple[str, bool, str | None]:
        logger.info("=" * 60)
        logger.info("Task: %s (%d trajectories)", task_id, len(records))

        resolved = sorted((r for r in records if r.resolved), key=lambda r: r.trajectory_id)
        failed = sorted((r for r in records if not r.resolved), key=lambda r: r.trajectory_id)
        logger.info("  %d resolved, %d failed", len(resolved), len(failed))

        if not resolved:
            logger.info("  No resolved trajectories — skipping task")
            return task_id, True, None

        task_output = output_dir / task_id
        if task_output.exists():
            shutil.rmtree(task_output)
        task_output.mkdir(parents=True, exist_ok=True)

        trace = BuildTrace()

        logger.info("  2a: Initialize from first resolved trajectory: %s", resolved[0].trajectory_id)
        graph = initialize_graph(resolved[0], trace)
        if not graph.nodes:
            logger.warning("  Initializer produced an empty graph — skipping task")
            return task_id, True, None

        if len(resolved) > 1:
            logger.info("  2b: Apply %d success updates", len(resolved) - 1)
            for record in resolved[1:]:
                graph = success_update(graph, record, trace)
        else:
            logger.info("  2b: No additional resolved trajectories to update")

        if failed:
            logger.info("  2c: Apply %d failure updates", len(failed))
            for record in failed:
                graph = failure_update(graph, record, trace)
        else:
            logger.info("  2c: No failed trajectories to update")

        logger.info("  2d: Reconcile node cards and branch semantics")
        graph, reconciliation_report = reconcile_task_graph(graph, trace)

        logger.info("  2e: Run light granularity normalize pass")
        graph, granularity_report = normalize_task_granularity(graph, trace)

        # Save
        graph.save(task_output)

        trace_path = task_output / "trace.json"
        trace_path.write_text(
            json.dumps({
                "mode": "incremental_build",
                "task_id": task_id,
                "task_description": resolved[0].task_description,
                "resolved_trajectories": [r.trajectory_id for r in resolved],
                "failed_trajectories": [r.trajectory_id for r in failed],
                "initializer_trajectory": resolved[0].trajectory_id,
                "build_trace": trace.to_dict(),
                "operation_summary": trace.operation_summary(),
                "reconciliation_report": reconciliation_report,
                "granularity_report": granularity_report,
                "final_graph_snapshot": graph_snapshot(graph),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("  Saved → %s (%d nodes, %d edges)",
                     task_output, len(graph.nodes), len(graph.edges))
        return task_id, True, None

    # ── Step 2: Per-task graph construction (incremental) ──
    task_items = sorted(tasks.items())
    failures: list[tuple[str, str]] = []

    if task_workers == 1 or len(task_items) <= 1:
        for task_id, records in task_items:
            try:
                build_single_task(task_id, records)
            except Exception as exc:  # pragma: no cover - defensive logging path
                logger.exception("Task failed: %s", task_id)
                failures.append((task_id, str(exc)))
    else:
        logger.info("Running intra-task build with %d parallel task worker(s)", task_workers)
        with ThreadPoolExecutor(max_workers=task_workers, thread_name_prefix="intra-task") as executor:
            futures = {
                executor.submit(build_single_task, task_id, records): task_id
                for task_id, records in task_items
            }
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    future.result()
                except Exception as exc:  # pragma: no cover - defensive logging path
                    logger.exception("Task failed: %s", task_id)
                    failures.append((task_id, str(exc)))

    logger.info("=" * 60)
    if failures:
        logger.warning("Done with %d failed task(s)", len(failures))
        for task_id, error in failures:
            logger.warning("  %s: %s", task_id, error)
    else:
        logger.info("Done!")


def _run_merge(args):
    """Run cross-task clustering-guided merge per repository domain."""
    import shutil

    import skillgraph.config as cfg

    if args.model:
        cfg.MODEL = args.model

    from skillgraph.merge.merger import GraphBundle, MergeTrace, tree_merge
    from skillgraph.models import SkillGraph
    from skillgraph.parse.trajectory_parser import parse_all_trajectories

    logger = logging.getLogger("skillgraph.cli")

    graphs_dir = DATA_DIR / "graphs"
    merge_root = DATA_DIR / "domain_graphs"

    if args.clean and args.resume:
        logger.error("Cannot use --clean and --resume together.")
        return

    task_descriptions: dict[str, str] = {}
    if args.trajectories:
        logger.info("Loading task descriptions from trajectories: %s", args.trajectories)
        for record in parse_all_trajectories(args.trajectories):
            if record.task_description and record.task_id not in task_descriptions:
                task_descriptions[record.task_id] = record.task_description

    # Load all per-task subgraphs
    if not graphs_dir.exists():
        logger.error("No per-task graphs found. Run 'build' first.")
        return

    domain_buckets: dict[str, list[tuple[str, SkillGraph]]] = {}
    for task_dir in sorted(graphs_dir.iterdir()):
        if not task_dir.is_dir() or not (task_dir / "graph.json").exists():
            continue
        graph = SkillGraph.load(task_dir)
        if graph.nodes:
            if task_dir.name not in task_descriptions:
                trace_path = task_dir / "trace.json"
                if trace_path.exists():
                    try:
                        trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
                        desc = str(trace_data.get("task_description") or "").strip()
                        if desc:
                            task_descriptions[task_dir.name] = desc
                    except Exception:
                        logger.debug("Could not read task description from %s", trace_path)
            domain_id = task_dir.name.split("__", 1)[0]
            domain_buckets.setdefault(domain_id, []).append((task_dir.name, graph))
            logger.info("Loaded %s: %d nodes, %d edges", task_dir.name, len(graph.nodes), len(graph.edges))

    if args.domain:
        allowed = set(args.domain)
        domain_buckets = {k: v for k, v in domain_buckets.items() if k in allowed}

    if not domain_buckets:
        logger.error("No domain buckets found to merge.")
        return

    logger.info("Merging %d repo-domain bucket(s)", len(domain_buckets))

    merge_root.mkdir(parents=True, exist_ok=True)

    def _level_index(level_dir: Path) -> int:
        return int(level_dir.name.split("_", 1)[1])

    def _load_level_bundles(domain_output: Path, level_dir: Path) -> list[GraphBundle]:
        metadata_path = level_dir / "subgraph_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for resume: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        bundles: list[GraphBundle] = []
        for item in metadata:
            idx = int(item["index"])
            subgraph_dir = level_dir / f"subgraph_{idx}"
            graph = SkillGraph.load(subgraph_dir)
            if not graph.nodes:
                raise ValueError(f"Resume subgraph is empty: {subgraph_dir}")
            source_tasks = [str(task_id) for task_id in item.get("source_tasks", [])]
            descriptions = [
                task_descriptions[task_id]
                for task_id in source_tasks
                if task_id in task_descriptions and task_descriptions[task_id]
            ]
            bundles.append(
                GraphBundle(
                    graph=graph,
                    task_ids=source_tasks,
                    task_descriptions=descriptions,
                    label=str(item.get("label") or f"subgraph_{idx}"),
                )
            )
        return bundles

    def _load_resume_state(domain_output: Path) -> tuple[list[GraphBundle], int, MergeTrace] | None:
        if not domain_output.exists():
            return None

        level_dirs = sorted(
            [
                path
                for path in domain_output.iterdir()
                if path.is_dir() and path.name.startswith("level_") and (path / "subgraph_metadata.json").exists()
            ],
            key=_level_index,
        )
        if not level_dirs:
            return None

        trace = MergeTrace()
        latest_bundles: list[GraphBundle] = []
        for level_dir in level_dirs:
            level_num = _level_index(level_dir)
            result_bundles = _load_level_bundles(domain_output, level_dir)
            latest_bundles = result_bundles
            clustering = {}
            merges = []
            clustering_path = level_dir / "clustering.json"
            merge_logs_path = level_dir / "merge_logs.json"
            if clustering_path.exists():
                clustering = json.loads(clustering_path.read_text(encoding="utf-8"))
            if merge_logs_path.exists():
                merges = json.loads(merge_logs_path.read_text(encoding="utf-8"))
            trace.record_level(level_num, clustering, merges, result_bundles)

        return latest_bundles, _level_index(level_dirs[-1]) + 1, trace

    for domain_id, members in sorted(domain_buckets.items()):
        task_ids = [task_id for task_id, _ in members]
        subgraphs = [graph for _, graph in members]
        domain_output = merge_root / domain_id

        if args.clean and domain_output.exists():
            shutil.rmtree(domain_output)
            logger.info("Cleaned merge output for domain %s: %s", domain_id, domain_output)

        domain_output.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Domain: %s (%d task graph(s))", domain_id, len(subgraphs))

        if args.resume and (domain_output / "final" / "graph.json").exists():
            logger.info("Domain %s already has a final graph at %s; skipping resume.", domain_id, domain_output / "final")
            continue

        resume_state = _load_resume_state(domain_output) if args.resume else None
        if resume_state is not None:
            initial_bundles, start_level, existing_trace = resume_state
            logger.info(
                "Resuming %s from level_%d outputs: %d graph bundle(s)",
                domain_id,
                start_level - 1,
                len(initial_bundles),
            )
            domain_graph, merge_trace = tree_merge(
                [],
                domain_output,
                task_descriptions=task_descriptions,
                cluster_capacity=args.cluster_capacity,
                merge_workers=args.merge_workers,
                initial_bundles=initial_bundles,
                start_level=start_level,
                trace=existing_trace,
            )
        else:
            domain_graph, merge_trace = tree_merge(
                subgraphs,
                domain_output,
                source_task_ids=task_ids,
                task_descriptions=task_descriptions,
                cluster_capacity=args.cluster_capacity,
                merge_workers=args.merge_workers,
            )

        domain_graph.save(domain_output / "final")
        (domain_output / "merge_trace.json").write_text(
            json.dumps(merge_trace.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (domain_output / "provenance.json").write_text(
            json.dumps({
                "domain_id": domain_id,
                "source_tasks": task_ids,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("Domain graph saved → %s (%d nodes, %d edges)",
                    domain_output / "final", len(domain_graph.nodes), len(domain_graph.edges))


def _run_text_baseline(args):
    """Build the file-based markdown skill baseline."""
    import shutil

    import skillgraph.config as cfg

    if args.model:
        cfg.MODEL = args.model

    from skillgraph.parse.trajectory_parser import parse_all_trajectories
    from skillgraph.textbaseline.builder import (
        TextBaselineTrace,
        build_domain_markdown,
        build_task_memo,
        save_domain_markdown,
        save_task_memo,
    )

    logger = logging.getLogger("skillgraph.cli")
    output_root = DATA_DIR / "text_baseline"

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
        logger.info("Cleaned output: %s", output_root)

    logger.info("Step 1: Parsing trajectories from %s", args.trajectories)
    all_records = parse_all_trajectories(args.trajectories)

    tasks: dict[str, list] = {}
    for record in all_records:
        tasks.setdefault(record.task_id, []).append(record)

    if args.task:
        allowed_tasks = set(args.task)
        tasks = {k: v for k, v in tasks.items() if k in allowed_tasks}

    if args.domain:
        allowed_domains = set(args.domain)
        tasks = {
            k: v for k, v in tasks.items()
            if k.split("__", 1)[0] in allowed_domains
        }

    logger.info(
        "Building text baseline from %d tasks (%d trajectories)",
        len(tasks), sum(len(v) for v in tasks.values()),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    trace = TextBaselineTrace()
    task_workers = max(1, int(args.task_workers or 1))
    task_items = sorted(tasks.items())
    failures: list[tuple[str, str]] = []
    built_task_memos: dict[str, str] = {}

    def build_single_task(task_id: str, records: list) -> tuple[str, str | None, dict]:
        memo, memo_trace = build_task_memo(task_id, records)
        return task_id, memo, memo_trace

    if task_workers == 1 or len(task_items) <= 1:
        for task_id, records in task_items:
            try:
                built_task_id, memo, memo_trace = build_single_task(task_id, records)
                trace.record_task_memo(built_task_id, memo_trace)
                if memo is not None:
                    built_task_memos[built_task_id] = memo
                    save_task_memo(output_root, built_task_id, memo, memo_trace)
                    logger.info("Task memo saved → %s", output_root / "task_memos" / built_task_id / "memo.md")
                else:
                    logger.info("Task memo skipped → %s", built_task_id)
            except Exception as exc:  # pragma: no cover - defensive logging path
                logger.exception("Task memo build failed: %s", task_id)
                failures.append((task_id, str(exc)))
    else:
        logger.info("Running text-baseline task memo build with %d parallel worker(s)", task_workers)
        with ThreadPoolExecutor(max_workers=task_workers, thread_name_prefix="text-baseline") as executor:
            futures = {
                executor.submit(build_single_task, task_id, records): task_id
                for task_id, records in task_items
            }
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    built_task_id, memo, memo_trace = future.result()
                    trace.record_task_memo(built_task_id, memo_trace)
                    if memo is not None:
                        built_task_memos[built_task_id] = memo
                        save_task_memo(output_root, built_task_id, memo, memo_trace)
                        logger.info("Task memo saved → %s", output_root / "task_memos" / built_task_id / "memo.md")
                    else:
                        logger.info("Task memo skipped → %s", built_task_id)
                except Exception as exc:  # pragma: no cover - defensive logging path
                    logger.exception("Task memo build failed: %s", task_id)
                    failures.append((task_id, str(exc)))

    domain_buckets: dict[str, list[tuple[str, str]]] = {}
    for task_id, memo in sorted(built_task_memos.items()):
        domain = task_id.split("__", 1)[0]
        domain_buckets.setdefault(domain, []).append((task_id, memo))

    logger.info("Step 2: Building %d domain markdown file(s)", len(domain_buckets))
    for domain, task_memos in sorted(domain_buckets.items()):
        try:
            domain_md, domain_trace = build_domain_markdown(domain, task_memos)
            trace.record_domain(domain, domain_trace)
            if domain_md is not None:
                save_domain_markdown(output_root, domain, domain_md, domain_trace)
                logger.info("Domain file saved → %s", output_root / "domains" / domain / f"{domain}.md")
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception("Domain baseline build failed: %s", domain)
            failures.append((domain, str(exc)))

    (output_root / "trace.json").write_text(
        json.dumps(trace.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=" * 60)
    if failures:
        logger.warning("Text baseline done with %d failure(s)", len(failures))
        for item_id, error in failures:
            logger.warning("  %s: %s", item_id, error)
    else:
        logger.info("Text baseline done!")


def _run_inspect(args):
    """Inspect a built graph."""
    from skillgraph.models import SkillGraph

    if args.domain:
        graph_dir = DATA_DIR / "domain_graphs" / args.domain / "final"
    elif args.show_global:
        graph_dir = DATA_DIR / "domain_graphs"
    elif args.task:
        graph_dir = DATA_DIR / "graphs" / args.task
    else:
        # List all built graphs
        graphs_dir = DATA_DIR / "graphs"
        if graphs_dir.exists():
            print("Per-task graphs:")
            for task_dir in sorted(graphs_dir.iterdir()):
                if task_dir.is_dir() and (task_dir / "graph.json").exists():
                    data = json.loads((task_dir / "graph.json").read_text())
                    print(f"  {task_dir.name}: {len(data['node_ids'])} nodes, {len(data['edges'])} edges")

        domains_dir = DATA_DIR / "domain_graphs"
        if domains_dir.exists():
            print("\nDomain graphs:")
            for domain_dir in sorted(domains_dir.iterdir()):
                final_dir = domain_dir / "final"
                if final_dir.is_dir() and (final_dir / "graph.json").exists():
                    data = json.loads((final_dir / "graph.json").read_text())
                    print(f"  {domain_dir.name}: {len(data['node_ids'])} nodes, {len(data['edges'])} edges")
        return

    if args.show_global:
        domains_dir = graph_dir
        if not domains_dir.exists():
            print(f"No domain graphs found at {domains_dir}")
            return
        for domain_dir in sorted(domains_dir.iterdir()):
            final_dir = domain_dir / "final"
            if final_dir.is_dir() and (final_dir / "graph.json").exists():
                data = json.loads((final_dir / "graph.json").read_text())
                print(f"{domain_dir.name}: {len(data['node_ids'])} nodes, {len(data['edges'])} edges")
        return

    if not graph_dir.exists():
        print(f"No graph found at {graph_dir}")
        return

    graph = SkillGraph.load(graph_dir)

    print(f"\nGraph: {graph_dir.name}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")

    print("\n--- Nodes ---")
    for node_id in graph.node_ids():
        md = graph.get_node(node_id)
        lines = md.split("\n")
        preview = "\n".join(lines[:8])
        print(f"\n[{node_id}]")
        print(preview)
        print("...")

    print("\n--- Edges ---")
    for e in graph.edges:
        print(f"  {e.source} --> {e.target}")
        print(f"    condition: {e.condition}")


if __name__ == "__main__":
    main()
