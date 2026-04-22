"""Data models for the Skill Graph pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class Edge:
    """A directed edge between two skill nodes."""

    source: str  # node_id
    target: str  # node_id
    condition: str  # when to follow this edge


class SkillGraph:
    """In-memory skill graph: markdown nodes + structured edges.

    Nodes are stored as markdown strings keyed by node_id.
    Edges are structured data (source, target, condition).
    """

    def __init__(self) -> None:
        self.nodes: dict[str, str] = {}  # node_id → markdown content
        self.edges: list[Edge] = []

    def add_node(self, node_id: str, markdown: str) -> None:
        self.nodes[node_id] = markdown

    def update_node(self, node_id: str, markdown: str) -> None:
        self.nodes[node_id] = markdown

    def get_node(self, node_id: str) -> str | None:
        return self.nodes.get(node_id)

    def add_edge(self, source: str, target: str, condition: str) -> None:
        """Insert or upsert an edge.

        If an edge with the same (source, target) already exists, merge
        the condition text instead of silently dropping the update.
        """
        for e in self.edges:
            if e.source == source and e.target == target:
                e.condition = _merge_edge_conditions(e.condition, condition)
                return
        self.edges.append(Edge(source=source, target=target, condition=condition))

    def update_edge(
        self,
        source: str,
        target: str,
        *,
        condition: str | None = None,
    ) -> bool:
        """Refine an existing edge's condition.

        Returns True if a matching edge was found and updated.
        """
        for edge in self.edges:
            if edge.source != source or edge.target != target:
                continue

            next_condition = edge.condition if condition is None else condition.strip()
            edge.condition = next_condition
            return True

        return False

    def node_ids(self) -> list[str]:
        return list(self.nodes.keys())

    def summary(self) -> str:
        """Compact text summary of the graph for LLM prompts."""
        lines = []
        for node_id, md in self.nodes.items():
            lines.append(f"### Node: {node_id}")
            lines.append(md)
            lines.append("")
        if self.edges:
            lines.append("### Edges")
            for e in self.edges:
                lines.append(f"- {e.source} --[{e.condition}]--> {e.target}")
        return "\n".join(lines)

    # ── Persistence ──

    def save(self, output_dir: Path) -> None:
        """Save graph to disk: one .md file per node + graph.json for edges."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write node markdown files
        nodes_dir = output_dir / "nodes"
        nodes_dir.mkdir(exist_ok=True)
        for node_id, md in self.nodes.items():
            (nodes_dir / f"{node_id}.md").write_text(md, encoding="utf-8")

        # Write edges + node list
        meta = {
            "node_ids": list(self.nodes.keys()),
            "edges": [
                {"source": e.source, "target": e.target, "condition": e.condition}
                for e in self.edges
            ],
        }
        (output_dir / "graph.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, output_dir: Path) -> SkillGraph:
        """Load graph from disk."""
        graph = cls()
        meta_path = output_dir / "graph.json"
        if not meta_path.exists():
            return graph

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        nodes_dir = output_dir / "nodes"

        for node_id in meta.get("node_ids", []):
            md_path = nodes_dir / f"{node_id}.md"
            if md_path.exists():
                graph.nodes[node_id] = md_path.read_text(encoding="utf-8")

        for e in meta.get("edges", []):
            graph.edges.append(Edge(
                source=e["source"], target=e["target"],
                condition=e["condition"],
            ))

        return graph


def _merge_edge_conditions(existing: str, new: str) -> str:
    """Merge two edge conditions conservatively without dropping information."""
    existing = existing.strip()
    new = new.strip()

    if not existing:
        return new
    if not new:
        return existing
    if existing.casefold() == new.casefold():
        return existing
    return f"{existing} ; OR {new}"
