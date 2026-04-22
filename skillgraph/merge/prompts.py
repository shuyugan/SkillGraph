"""Prompts for cross-task merge: clustering and direct graph synthesis."""

# ═══════════════════════════════════════════════════════════════════════
# Level-wise clustering
# ═══════════════════════════════════════════════════════════════════════

CLUSTER_LEVEL_SYSTEM = """\
You organize a set of current-level skill graphs from the SAME repository domain
into merge groups for the NEXT merge level.

This is NOT a topic taxonomy task.
Your job is to decide which graphs should be merged first because they are most
likely to produce one cleaner, less redundant, still-navigable merged graph.

## Core objective

Partition the input graphs into small merge groups that maximize MERGE
COMPATIBILITY, not broad topical similarity.

## What counts as a good merge group

Graphs belong together when most of the following are true:
- they contain overlapping reusable local skills
- their nodes can likely be merged into CRISP, still-usable skill nodes
- their conditional structure is compatible enough to synthesize into one
  cleaner, less redundant graph
- combining them is likely to reduce duplication without creating broad,
  mushy engineering themes

## What NOT to do

Do NOT group graphs together just because:
- they are in the same broad repo area
- they both involve testing/searching/editing in a generic sense
- they have a few overlapping words but would likely produce an over-merged graph

## Output constraints

- Return a PARTITION: every graph index must appear exactly once
- Prefer groups of size 2-4
- Use singleton groups only for true outliers
- Be conservative about high-level abstraction drift
- Optimize for groups that could be synthesized into a cleaner, less redundant,
  still-navigable graph

## Output format

Return JSON:
{
  "groups": [
    {
      "members": [0, 3, 5],
      "reason": "one sentence"
    }
  ]
}
"""

CLUSTER_LEVEL_USER = """\
Cluster these current-level graphs into merge groups for the next merge level.

The grouping criterion is: which graphs should be merged first in order to
produce cleaner, less redundant, still-navigable graphs.

{graph_packets}

Return only the merge groups.
"""


# ═══════════════════════════════════════════════════════════════════════
# Direct graph synthesis
# ═══════════════════════════════════════════════════════════════════════

DIRECT_GRAPH_SYNTHESIS_SYSTEM = """\
You synthesize ONE merged skill graph from 2-4 input skill graphs from the SAME
repository domain.

Your job is to jointly decide:
- which skill nodes should remain
- which overlapping skill nodes should be merged
- what each merged node should say
- which edges should be kept, removed, merged, or rewritten

This is a direct GRAPH SYNTHESIS task, not pairwise node matching.

## Output goal

Produce a graph that is:
- cleaner and less redundant than the naive union
- still navigable
- made of crisp reusable skill nodes
- structured so that the edges remain meaningful
- non-empty

## Hard constraint

- Never return an empty graph
- If strong merges are not possible, preserve the relevant source nodes and
  their important structure rather than collapsing the graph
- A valid merged graph may retain multiple separate source substructures when
  that yields a cleaner result than forced compression


## Node rules

- Each node should capture ONE coherent reusable skill unit
- Merge nodes only when the resulting node stays crisp and non-mushy
- It is better to keep two separate nodes than to create one vague engineering theme
- Do not over-compress the graph just to reduce node count or edge count
- If a clean merge would lose important structure, preserve separate nodes instead
- Preserve useful engineering/domain signal; do not over-genericize into empty prose
- Keep pitfalls inside nodes when appropriate rather than creating extra broad nodes

## Edge rules

- Edges should be genuine conditional routing cues
- You may preserve, delete, merge, or rewrite source edges when needed
- Do NOT preserve every source edge by default
- Do NOT add edges just because two nodes are related
- Prefer a graph whose connectivity is interpretable at inference time
- Avoid hub-like validation nodes connected from everything unless the routing is truly justified
- Conditions should be action-guiding and semantically precise; keep them concise when possible,
  but do not oversimplify away important routing detail

## Relationship to source task descriptions

Use task descriptions as supporting context for what each source graph was about,
but let the GRAPH CONTENT drive the actual synthesis.

## Output format

Return JSON:
{
  "nodes": [
    {
      "node_id": "kebab-case-id",
      "summary": "one-line scan card summary",
      "description": "one-line retrieval description",
      "triggers": ["trigger 1", "trigger 2"],
      "body": "## When to Use\\n\\n...\\n\\n## Strategy\\n\\n...\\n\\n## Actions\\n\\n- ...\\n\\n## Checks\\n\\n- ..."
    }
  ],
  "edges": [
    {
      "source": "node-id-a",
      "target": "node-id-b",
      "condition": "routing cue with enough detail to guide the next step"
    }
  ]
}

Do not wrap the JSON in code fences.
"""

DIRECT_GRAPH_SYNTHESIS_USER = """\
Synthesize one merged graph from these source graphs.

{graph_packets}

Return only the merged graph JSON.
"""

# ═══════════════════════════════════════════════════════════════════════
# Graph-Pair Alignment Judge
# ═══════════════════════════════════════════════════════════════════════

GRAPH_ALIGN_SYSTEM = """\
You align two task-local skill graphs from the SAME repository domain.

Your job is NOT to merge markdown directly. Your job is to identify which \
node pairs are similar ENOUGH that a downstream executor can rewrite them into \
ONE coherent, more general reusable skill node.

This is a MERGEABILITY decision, not an identity test.

## What counts as mergeable

Two nodes should be aligned into one merge pair when MOST of the following are true:
- they address a similar local problem, decision point, or subgoal
- their strategies are compatible enough to be expressed as one more general pattern
- their triggers or usage situations overlap in a meaningful way, even if details differ
- their checks or intended outcomes are compatible rather than contradictory
- merging them would still produce a SMALL, coherent, useful node rather than a confused mash-up

Differences in task framing, exact commands, concrete examples, trigger surface form, \
or verification phrasing are ACCEPTABLE.

## What does NOT count as mergeable

Do NOT merge nodes just because:
- they are in the same broad area
- they occur near each other in the workflow
- they both involve testing, searching, or editing in a generic sense
- the words sound similar but the resulting merged node would be internally inconsistent

Do NOT require the nodes to be nearly identical.
If they can be cleanly unified into a more general node, you SHOULD merge them.

## Important constraints

- Output one-to-one merge pairs
- Be willing to merge when two nodes are clearly about similar reusable guidance, \
even if one is more task-specific and the other is more general
- Prefer coherent abstraction over literal similarity
- Do NOT force every node to be matched
- A node may appear in at most one merge pair
- When unsure, ask: "Could a competent executor combine these into one clean node \
without making the result confusing?"

## What to use as evidence

Judge mergeability from NODE CONTENT ONLY.
Do not rely on graph edges or graph position for alignment.
Edge handling happens later in a separate reconciliation step after node merge.

## Output format

Return JSON:
{
  "merge_pairs": [
    {
      "a_node": "<node_id_from_graph_a>",
      "b_node": "<node_id_from_graph_b>",
      "reason": "one sentence"
    }
  ]
}

If nothing should merge, return {"merge_pairs": []}.
"""

GRAPH_ALIGN_USER = """\
Align these two task-local skill graphs from the same repository domain.

## Graph A

{graph_a_summary}

## Graph B

{graph_b_summary}

Return only the one-to-one merge pairs that are sufficiently similar to be \
unified into one coherent, more general skill node.
"""


# ═══════════════════════════════════════════════════════════════════════
# Edge Reconcile
# ═══════════════════════════════════════════════════════════════════════

EDGE_RECONCILE_SYSTEM = """\
You reconcile candidate cross-task edges between the SAME source node and target \
node after node alignment.

Your job is to return the MINIMAL semantically correct final edge set for this \
single source-target pair.

## Rules

- Prefer the smallest correct result, usually ONE final edge
- Keep TWO final edges only when they are clearly non-redundant and have \
  genuinely different routing cues
- Collapse duplicate or near-duplicate evidence into one generalized condition
- Use routing cues that are semantically precise and action-guiding; concise
  when possible, but not oversimplified
- Do not invent unsupported transitions

## Output format

Return JSON:
{
  "edges": [
    {
      "condition": "generalized routing cue with enough detail",
      "reason": "one sentence"
    }
  ]
}

Return {"edges": []} only if no edge should remain.
"""

EDGE_RECONCILE_USER = """\
Reconcile the candidate edges for this single source-target pair.

{edge_context}

Return only the minimal semantically correct final edge set.
"""


# ═══════════════════════════════════════════════════════════════════════
# Executor: merge two nodes into a more general version
# ═══════════════════════════════════════════════════════════════════════

MERGE_EXECUTOR_SYSTEM = """\
You merge two skill nodes from DIFFERENT tasks into one more general node.

Both input nodes describe the same underlying decision pattern but may use \
different task-specific framing. Your job is to produce a UNIFIED node that \
captures the common pattern while being useful beyond the exact source tasks.

## Rules

1. Find the common abstraction — what is the same decision pattern underlying \
both nodes?
2. Generalize exact repo-specific identifiers, but do not erase useful \
engineering/domain signal when it helps make the merged skill clearer
3. Combine the best actions/rationale from both — prefer breadth over \
repeating the same idea twice
4. Merge triggers: combine both trigger lists, remove duplicates, keep the \
most general ones
5. If both have Pitfalls sections, combine them (deduplicate)
6. Preserve or synthesize short generalized code snippets only when they materially \
help the merged skill, and keep them embedded in the most relevant section
7. Keep YAML frontmatter, but let the body use a natural small set of headings \
instead of a rigid fixed template
8. The merged body should still clearly cover when-to-use, strategy, concrete \
actions, and checks/verification
9. Choose a general, descriptive node name (kebab-case slug)
10. Keep it concise — the merged node should not be longer than the longer \
of the two inputs

Output the complete merged markdown (no code fences around it).
"""

MERGE_EXECUTOR_USER = """\
Merge these two skill nodes into one unified, more general node.

## Node A

{node_a_markdown}

## Node B

{node_b_markdown}

Output the complete merged markdown.
"""


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def format_node_summary(node_id: str, markdown: str) -> str:
    """Extract a compact summary of a node for the judge prompt."""
    import re

    lines = [f"### [{node_id}]"]

    summary_match = re.search(r'^summary:\s*"?(.+?)"?\s*$', markdown, re.MULTILINE)
    if summary_match:
        lines.append(f"Summary: {summary_match.group(1)}")

    desc_match = re.search(r'^description:\s*"?(.+?)"?\s*$', markdown, re.MULTILINE)
    if desc_match:
        lines.append(f"Description: {desc_match.group(1)}")

    # Triggers
    triggers = []
    in_triggers = False
    for line in markdown.split("\n"):
        if line.strip() == "triggers:":
            in_triggers = True
            continue
        if in_triggers:
            t_match = re.match(r'^\s+-\s+"?(.+?)"?\s*$', line)
            if t_match:
                triggers.append(t_match.group(1))
            elif not line.strip().startswith("-"):
                break
    if triggers:
        lines.append("Triggers:")
        for t in triggers:
            lines.append(f"  - {t}")

    def first_section_text(*section_names: str) -> str | None:
        for section in section_names:
            pattern = rf"## {re.escape(section)}\s*\n(.*?)(?=\n## |\Z)"
            m = re.search(pattern, markdown, re.DOTALL)
            if m:
                text = m.group(1).strip().split("\n")[0][:200]
                return text
        return None

    when_text = first_section_text("When to Use", "Situation", "Context")
    if when_text:
        lines.append(f"When to Use: {when_text}")

    strategy_text = first_section_text("Strategy", "Approach", "Method")
    if strategy_text:
        lines.append(f"Strategy: {strategy_text}")

    return "\n".join(lines)


def format_graph_packet(
    *,
    index: int,
    task_descriptions: list[str],
    graph_summary: str,
) -> str:
    """Render a full graph packet for clustering / synthesis prompts."""
    lines = [
        f"## Graph {index}",
    ]
    if task_descriptions:
        lines.append("Task descriptions:")
        for desc in task_descriptions:
            text = " ".join(desc.split()).strip()
            if text:
                lines.append(f"- {text}")
    else:
        lines.append("Task descriptions: (not available)")

    lines.extend([
        "Graph content:",
        graph_summary.strip(),
    ])
    return "\n".join(lines)


def format_graph_summary(label: str, markdown_by_node: dict, edges: list) -> str:
    """Render a node-only graph summary for graph-pair alignment."""
    lines = [f"### {label}", "", "#### Nodes"]

    for node_id, markdown in sorted(markdown_by_node.items()):
        lines.append(f"### [{node_id}]")
        lines.append(markdown.strip())
        lines.append("")

    return "\n".join(lines).strip()


def format_edge_reconcile_context(
    source_id: str,
    source_markdown: str,
    target_id: str,
    target_markdown: str,
    candidates: list[dict],
) -> str:
    """Render a compact context block for edge reconciliation."""
    lines = [
        "## Source Node",
        format_node_summary(source_id, source_markdown),
        "",
        "## Target Node",
        format_node_summary(target_id, target_markdown),
        "",
        "## Candidate Edges",
    ]

    for idx, edge in enumerate(candidates, start=1):
        condition = " ".join((edge.get("condition") or "").split())
        if len(condition) > 220:
            condition = condition[:217] + "..."
        origin = edge.get("origin", "?")
        lines.append(
            f"{idx}. [{origin}] {source_id} --> {target_id}"
            + (f" | {condition}" if condition else "")
        )

    return "\n".join(lines).strip()
