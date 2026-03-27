"""Step 5: Assemble final skill graph — validate links, generate index, write files."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from skillgraph.config import SKILLS_DIR, CACHE_DIR
from skillgraph.models import SkillNode

logger = logging.getLogger("skillgraph.assemble")

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def assemble(nodes: list[SkillNode], output_dir: Path | None = None) -> Path:
    """Write all skill nodes and index.md to the output directory."""
    output_dir = output_dir or SKILLS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    slugs = {node.node_id for node in nodes}

    # Validate wikilinks
    dangling = _validate_links(nodes, slugs)
    if dangling:
        logger.warning("Found %d nodes with dangling wikilinks", len(dangling))

    # Write node files
    for node in nodes:
        md = node.markdown
        # Remove dangling links (replace [[bad]] with plain text)
        if node.node_id in dangling:
            for target in dangling[node.node_id]:
                md = md.replace(f"[[{target}]]", target)

        file_path = output_dir / f"{node.node_id}.md"
        file_path.write_text(md, encoding="utf-8")

    # Generate and write index.md
    index_content = _generate_index(nodes)
    (output_dir / "index.md").write_text(index_content, encoding="utf-8")

    # Save graph state
    _save_graph_state(nodes)

    logger.info("Assembled %d skill nodes + index.md in %s", len(nodes), output_dir)
    return output_dir


def load_existing_nodes(output_dir: Path | None = None) -> list[SkillNode]:
    """Load existing skill nodes from the graph state file."""
    state_file = CACHE_DIR / "graph_state.json"
    if not state_file.exists():
        return []

    output_dir = output_dir or SKILLS_DIR
    data = json.loads(state_file.read_text(encoding="utf-8"))
    nodes = []

    for entry in data.get("nodes", []):
        md_path = output_dir / f"{entry['node_id']}.md"
        if md_path.exists():
            markdown = md_path.read_text(encoding="utf-8")
        else:
            markdown = entry.get("markdown", "")

        nodes.append(
            SkillNode(
                node_id=entry["node_id"],
                type=entry["type"],
                markdown=markdown,
                source_tasks=entry.get("source_tasks", []),
            )
        )

    logger.info("Loaded %d existing nodes from graph state", len(nodes))
    return nodes


# ── Internal helpers ──


def _validate_links(nodes: list[SkillNode], valid_slugs: set[str]) -> dict[str, list[str]]:
    dangling: dict[str, list[str]] = {}
    for node in nodes:
        targets = _WIKILINK_RE.findall(node.markdown)
        bad = [t for t in targets if t not in valid_slugs]
        if bad:
            dangling[node.node_id] = bad
    return dangling


def _generate_index(nodes: list[SkillNode]) -> str:
    groups: dict[str, list[SkillNode]] = defaultdict(list)
    for node in nodes:
        groups[node.type].append(node)

    type_order = ["fix_pattern", "technique", "recovery", "anti_pattern"]
    type_labels = {
        "fix_pattern": "Fix Patterns",
        "technique": "Techniques",
        "recovery": "Error Recovery",
        "anti_pattern": "Anti-Patterns",
    }

    lines = ["# Skill Library\n"]

    for node_type in type_order:
        if node_type not in groups:
            continue
        label = type_labels.get(node_type, node_type)
        lines.append(f"## {label}\n")
        for node in sorted(groups[node_type], key=lambda n: n.node_id):
            desc = _extract_description(node.markdown)
            lines.append(f"- [[{node.node_id}]]: {desc}")
        lines.append("")

    # Any types not in type_order
    for node_type, node_list in sorted(groups.items()):
        if node_type in type_order:
            continue
        lines.append(f"## {node_type}\n")
        for node in sorted(node_list, key=lambda n: n.node_id):
            desc = _extract_description(node.markdown)
            lines.append(f"- [[{node.node_id}]]: {desc}")
        lines.append("")

    return "\n".join(lines)


def _extract_description(markdown: str) -> str:
    parts = markdown.split("---", 2)
    body = parts[2].strip() if len(parts) >= 3 else markdown.strip()
    for line in body.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:120]
    return "(no description)"


def _save_graph_state(nodes: list[SkillNode]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "nodes": [
            {
                "node_id": n.node_id,
                "type": n.type,
                "source_tasks": n.source_tasks,
            }
            for n in nodes
        ]
    }
    state_file = CACHE_DIR / "graph_state.json"
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
