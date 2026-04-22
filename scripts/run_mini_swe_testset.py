#!/usr/bin/env python3

"""Run mini-swe-agent on the task IDs listed in SkillGraph/test.json.

This is a thin wrapper around mini-swe-agent's existing SWE-bench batch runner.
It keeps the default mini-swe-agent config/output format, but replaces the
dataset-selection step with a fixed task-id list loaded from a local JSON file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from rich.live import Live

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLGRAPH_DIR = REPO_ROOT / "SkillGraph"
MINI_SWE_AGENT_DIR = REPO_ROOT / "mini-swe-agent"
DEFAULT_SELECTION_FILE = SKILLGRAPH_DIR / "test.json"
DEFAULT_OUTPUT_DIR = MINI_SWE_AGENT_DIR / "mini_swe_testset_runs" / "default"
MINI_SWE_AGENT_SRC = REPO_ROOT / "mini-swe-agent" / "src"
DEFAULT_CACHE_ROOT = Path(os.environ.get("MINI_SWE_CACHE_DIR", "/tmp/mini_swe_agent_cache"))
LOCAL_DATASET_FALLBACKS = {
    ("verified", "test"): Path("/tmp/swebench_verified_test.parquet"),
}

# The shared home cache on this machine may be read-only. Point HF caches to a
# writable location unless the user already overrode them.
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_ROOT / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_CACHE_ROOT / "huggingface" / "datasets"))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_CACHE_ROOT / "huggingface" / "hub"))

if str(MINI_SWE_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(MINI_SWE_AGENT_SRC))

try:
    from datasets import load_dataset
    from minisweagent.config import get_config_from_spec
    from minisweagent.run.benchmarks.swebench import (
        DATASET_MAPPING,
        DEFAULT_CONFIG_FILE,
        process_instance,
    )
    from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
    from minisweagent.utils.log import add_file_handler, logger
    from minisweagent.utils.serialize import UNSET, recursive_merge
except Exception as exc:  # pragma: no cover - import path/env issue
    raise SystemExit(
        "Failed to import mini-swe-agent dependencies. "
        "Run this script with the same Python environment as mini-swe-agent "
        f"(for example `/usr/bin/python3`). Import error: {exc}"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run mini-swe-agent on the task IDs listed in SkillGraph/test.json "
            "or another compatible selection file."
        )
    )
    parser.add_argument(
        "--selection-file",
        default=str(DEFAULT_SELECTION_FILE),
        help="Path to a JSON dict mapping repo/domain names to lists of SWE-bench instance IDs.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Restrict the run to one or more keys from the selection file. Can be repeated.",
    )
    parser.add_argument(
        "--subset",
        default="verified",
        help="SWE-bench subset or dataset path. Defaults to 'verified'.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to load. Defaults to 'test'.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for mini-swe-agent trajectories and preds.json.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads. Defaults to 1.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model name override, e.g. anthropic/claude-3-haiku-20240307.",
    )
    parser.add_argument(
        "--model-class",
        default=None,
        help="Optional mini-swe-agent model class override.",
    )
    parser.add_argument(
        "--environment-class",
        default=None,
        help="Optional environment class override, e.g. docker or singularity.",
    )
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        default=[str(DEFAULT_CONFIG_FILE)],
        help=(
            "mini-swe-agent config spec. Can be repeated. "
            "Defaults to the built-in swebench.yaml."
        ),
    )
    parser.add_argument(
        "--redo-existing",
        action="store_true",
        help="Re-run instances already present in preds.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selection and print counts without running any tasks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of selected instances after filtering.",
    )
    return parser.parse_args()


def load_selection(selection_file: Path, repo_filter: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    raw = json.loads(selection_file.read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"Selection file must be a JSON object: {selection_file}")

    selected: dict[str, list[str]] = {}
    for repo_name, task_ids in raw.items():
        if repo_filter and repo_name not in repo_filter:
            continue
        if not isinstance(task_ids, list) or not all(isinstance(x, str) for x in task_ids):
            raise SystemExit(
                f"Selection file entries must be lists of strings. Problem key: {repo_name}"
            )
        selected[repo_name] = task_ids

    if repo_filter:
        missing_repos = sorted(set(repo_filter) - set(selected))
        if missing_repos:
            raise SystemExit(
                f"Requested repo key(s) not found in {selection_file}: {', '.join(missing_repos)}"
            )

    flat_ids = [task_id for task_ids in selected.values() for task_id in task_ids]
    duplicates = [task_id for task_id, count in Counter(flat_ids).items() if count > 1]
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:10])
        raise SystemExit(f"Selection file contains duplicate task IDs: {preview}")

    return selected, flat_ids


def load_selected_instances(subset: str, split: str, selected_ids: list[str]) -> list[dict]:
    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")

    try:
        instances = list(load_dataset(dataset_path, split=split))
    except Exception as exc:
        fallback_path = LOCAL_DATASET_FALLBACKS.get((subset, split))
        if fallback_path is None or not fallback_path.exists():
            raise
        logger.warning(
            "Failed to load %s/%s via datasets (%s). Falling back to local parquet: %s",
            dataset_path,
            split,
            exc,
            fallback_path,
        )
        import pandas as pd

        instances = pd.read_parquet(fallback_path).to_dict(orient="records")

    instance_map = {instance["instance_id"]: instance for instance in instances}

    missing = sorted(set(selected_ids) - set(instance_map))
    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(
            f"{len(missing)} selected task ID(s) were not found in {dataset_path}/{split}. "
            f"Examples: {preview}"
        )

    return [instance_map[task_id] for task_id in selected_ids]


def save_run_metadata(
    output_dir: Path,
    *,
    selection_file: Path,
    selected_by_repo: dict[str, list[str]],
    selected_ids: list[str],
    subset: str,
    split: str,
) -> None:
    metadata = {
        "selection_file": str(selection_file),
        "subset": subset,
        "split": split,
        "total_instances": len(selected_ids),
        "selected_by_repo": selected_by_repo,
        "selected_ids": selected_ids,
    }
    (output_dir / "selected_instances.json").write_text(json.dumps(metadata, indent=2))


def build_config(args: argparse.Namespace) -> dict:
    logger.info(f"Building agent config from specs: {args.config}")
    configs = [get_config_from_spec(spec) for spec in args.config]
    configs.append(
        {
            "environment": {"environment_class": args.environment_class or UNSET},
            "model": {
                "model_name": args.model or UNSET,
                "model_class": args.model_class or UNSET,
            },
        }
    )
    return recursive_merge(*configs)


def run_instances(instances: list[dict], output_dir: Path, config: dict, workers: int) -> None:
    progress_manager = RunBatchProgressManager(
        len(instances),
        output_dir / f"exit_statuses_{int(time.time())}.yaml",
    )

    def process_futures(futures: dict[concurrent.futures.Future, str]) -> None:
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as exc:  # pragma: no cover - mirrors mini-swe-agent runner
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {exc}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, exc)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_dir, config, progress_manager): instance[
                    "instance_id"
                ]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:  # pragma: no cover - interactive path
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


def main() -> int:
    args = parse_args()
    selection_file = Path(args.selection_file).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)

    add_file_handler(output_dir / "minisweagent.log")
    logger.info(f"Results will be saved to {output_dir}")

    selected_by_repo, selected_ids = load_selection(selection_file, args.repo)
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]
        allowed = set(selected_ids)
        selected_by_repo = {
            repo_name: [task_id for task_id in task_ids if task_id in allowed]
            for repo_name, task_ids in selected_by_repo.items()
        }

    instances = load_selected_instances(args.subset, args.split, selected_ids)

    if not args.redo_existing and (output_dir / "preds.json").exists():
        existing_ids = set(json.loads((output_dir / "preds.json").read_text()).keys())
        before = len(instances)
        instances = [instance for instance in instances if instance["instance_id"] not in existing_ids]
        skipped = before - len(instances)
        if skipped:
            logger.info(f"Skipping {skipped} existing instance(s) already present in preds.json")

    selected_counter = {repo_name: len(task_ids) for repo_name, task_ids in selected_by_repo.items()}
    logger.info(f"Selected repo counts: {selected_counter}")
    logger.info(f"Prepared {len(instances)} instance(s) to run")

    save_run_metadata(
        output_dir,
        selection_file=selection_file,
        selected_by_repo=selected_by_repo,
        selected_ids=[instance["instance_id"] for instance in instances],
        subset=args.subset,
        split=args.split,
    )

    if args.dry_run:
        logger.info("Dry run complete. No tasks were executed.")
        return 0

    if not instances:
        logger.info("No instances left to run.")
        return 0

    config = build_config(args)
    run_instances(instances, output_dir, config, max(args.workers, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
