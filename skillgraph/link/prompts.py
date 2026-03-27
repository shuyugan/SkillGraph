"""Prompts for Step 4: link building."""

LINK_BUILD_SYSTEM = """\
You are a knowledge graph editor. You embed wikilinks into skill nodes \
to create a navigable skill library for a debugging agent.

## Rules

1. Links use [[node-slug]] format embedded in conditional prose sentences.
2. Only add these types of links:
   - **if_fails**: "If this approach fails → [[recovery-node]]"
   - **instead_do**: "Instead of this → [[better-technique-node]]"
   - **needs**: "Requires [[dependency-node]] first"
3. Do NOT add sequential/workflow links ("after this, do that").
4. Do NOT add generic "see also" links. Every link must be actionable.
5. Do NOT change the core content of the node — only INSERT link sentences \
in natural positions within the existing text, or add a brief section at the end.
6. Maximum 5 outgoing links per node.
7. **Keep link sentences abstract** — do NOT include project-specific names, file paths, \
framework names, or class names from the relationship description. \
Describe the link condition generically (e.g., "if the invocation context is correct but \
tests still fail" instead of "if switching to Django's canonical runner works but errors remain").

Return the complete updated markdown (with frontmatter).
"""

LINK_BUILD_USER = """\
Add wikilinks to the following skill node.

## Target Node
{target_node_markdown}

## Relationships to Embed
{relationships_text}

## Available Target Nodes
{available_nodes_text}

Rewrite the target node with wikilinks inserted.
"""


def format_relationships_text(relationships: list[dict]) -> str:
    """Format relationships for the link build prompt."""
    if not relationships:
        return "(no relationships to embed)"
    lines = []
    for rel in relationships:
        lines.append(
            f"- {rel['rel_type']}: this node → [[{rel['target_slug']}]] "
            f"({rel.get('description', '')})"
        )
    return "\n".join(lines)


def format_available_nodes_text(nodes: list, exclude_slug: str) -> str:
    """Format available nodes as a summary."""
    parts = []
    for node in nodes:
        if node.node_id == exclude_slug:
            continue
        # Extract first meaningful line from markdown (skip frontmatter)
        lines = node.markdown.split("\n")
        description = ""
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if not in_frontmatter and line.strip() and not line.startswith("#"):
                description = line.strip()[:100]
                break
        parts.append(f"- [[{node.node_id}]] ({node.type}): {description}")
    return "\n".join(parts) if parts else "(no other nodes available)"
