"""Prompts for Step 3: match finding and merge."""

MATCH_SYSTEM = """\
You are matching a new skill node against an existing skill library.
Both new and existing nodes are already abstracted (use <placeholder> notation).

Given a new node and a list of existing nodes, determine if the new node \
describes the SAME underlying debugging pattern as any existing node.

Two nodes MATCH if:
- They describe the same core technique, recovery strategy, or anti-pattern
- An agent would consult BOTH nodes in the same situation
- They are duplicates or near-duplicates at the pattern level (even if phrased differently)

Two nodes DO NOT MATCH if:
- They solve fundamentally different problems
- The techniques/strategies are genuinely distinct (e.g., "scope a search" vs "fix a missing parameter")
- One is a special case of the other but with genuinely different steps

Return JSON:
{"match_id": "<node_id of the matching existing node>" | "none", "reason": "one sentence"}
"""

MATCH_USER = """\
## New Node
{new_node_markdown}

## Existing Nodes
{existing_nodes_summary}

Which existing node (if any) matches the new node? Return the node_id or "none".
"""

MERGE_SYSTEM = """\
You merge two abstract skill nodes that describe the same debugging pattern into one unified node.
Both input nodes already use <placeholder> notation.

## Rules

1. The merged node must remain FULLY ABSTRACT — no task-specific identifiers.
2. Consolidate insights from both nodes: if one has a step the other lacks, include it.
3. Unify triggers: combine both trigger lists, removing duplicates.
4. Unify code templates: keep the most informative structure; use <placeholder> consistently.
5. The description should reflect the broadest applicable scenario.
6. Keep the same markdown format (YAML frontmatter + body).
7. Target 20-40 lines.
8. Choose the most general, descriptive slug that covers the unified pattern.

Output the complete merged markdown file content (no code fences around it).
"""

MERGE_USER = """\
Merge these two skill nodes into one unified abstract node.

## Node A
{node_a_markdown}

## Node B
{node_b_markdown}

Write the merged markdown node (YAML frontmatter + body, no surrounding code block).
"""


def format_existing_nodes_for_match(nodes: list, node_type: str) -> str:
    """Format existing nodes of the same type as a summary for the match prompt."""
    candidates = [n for n in nodes if n.type == node_type]
    if not candidates:
        return "(no existing nodes of this type)"

    parts = []
    for node in candidates:
        # Extract first meaningful line after frontmatter as description
        lines = node.markdown.split("\n")
        desc = ""
        in_fm = False
        for line in lines:
            if line.strip() == "---":
                in_fm = not in_fm
                continue
            if not in_fm and line.strip() and not line.startswith("#"):
                desc = line.strip()[:150]
                break

        parts.append(f"### [{node.node_id}]\n{desc}\n")

    return "\n".join(parts)
