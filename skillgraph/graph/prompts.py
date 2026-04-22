"""Prompts for accumulative graph construction: initializer, updaters, executor."""

# ═══════════════════════════════════════════════════════════════════════
# Initializer
# ═══════════════════════════════════════════════════════════════════════

INITIALIZER_SYSTEM = """\
You analyze a successful software debugging trajectory and extract an initial \
skill graph: a set of skill nodes and the edges between them.

## What is a skill node

A skill node is a reusable decision pattern — a non-trivial strategic choice \
an agent made under specific observable conditions. Each node should be \
independently useful outside the exact source task.

## What is an edge

An edge represents a conditional transition between skills observed in the \
trajectory. The edge stores only a SHORT semantic routing cue explaining when \
the agent should move from skill A to skill B.

## Generalization rules (CRITICAL)

Generalize aggressively, but do NOT abstract away useful engineering signal:
- Avoid repo-specific names, exact file paths, and one-off identifiers
- It is OK to keep broadly reusable software concepts such as subclass override, \
  parser, estimator, docstring section, symbolic simplification, etc. when they \
  materially clarify the skill
- Prefer concepts that travel across many tasks in the same repository domain \
  over ultra-generic wording that becomes vague
- YES to observable quantities: ">20 grep results", "exit code 2"
- YES to generic tool names: grep, find, python, git, sed
- Use <placeholder> notation for project-specific parts in actions \
(e.g., grep -rn "def <target_method>" <repo_root>/<subsystem>/)
- In the Actions section, provide GENERAL command templates, not examples \
tied to the specific task
- When reusable code structure matters, include a SHORT generalized code snippet \
using <placeholder> identifiers, embedded in the most relevant section
- Keep Actions compact: 3-6 decision-relevant bullets when possible
- Do NOT dump command-by-command editing minutiae, line-number arithmetic, or \
one-off shell flags unless they materially change the strategy

## Node content structure

Each node should contain enough information to be written as a useful markdown \
skill file that naturally covers:
- a SCAN CARD for quick navigation (`summary`, `description`, `triggers`)
- when to use the skill
- the core strategy
- concrete actions
- how to verify success
Generalized code snippets may be embedded inside any relevant part when useful.

## Granularity

- Too fine: "ran grep for a function name" → just one command
- Right: "when broad search returns too many results, use traceback clues to \
scope search to the owning subsystem" → a transferable decision pattern
- Too coarse: "found and fixed the bug" → entire trajectory
- Do NOT automatically collapse diagnosis and implementation into one node
- If a diagnosis route could be reused with multiple downstream fixes, keep it separate
- If a fix pattern can apply after multiple localization routes, keep it separate
- Merge steps only when they form one inseparable tactical pattern in practice

## Output format

Return JSON:
{
  "nodes": [
    {
      "node_id": "kebab-case-slug",
      "summary": "one-sentence scan-level card for quick navigation",
      "description": "one-line summary of what this skill does (for index/retrieval)",
      "triggers": ["observable condition 1 that activates this skill", "condition 2"],
      "situation": "generalized decision situation. 1-2 sentences.",
      "approach": "what strategy to use. 1-2 sentences.",
      "rationale": "why this works better than naive alternatives. 1-2 sentences.",
      "actions": ["step 1 with <placeholder>", "step 2..."],
      "embedded_code": [
        {
          "section": "approach | actions | verification | situation | rationale",
          "language": "python | bash | text | diff | ...",
          "code": "optional short generalized snippet using <placeholder> names",
          "note": "optional one-line setup sentence placed above the snippet"
        }
      ],
      "verification": "how to confirm success"
    }
  ],
  "edges": [
    {
      "source": "node-id-A",
      "target": "node-id-B",
      "condition": "SHORT routing cue: what observation or state triggers this transition"
    }
  ]
}

Extract 2-6 nodes. Prefer explicit navigable structure over burying alternate \
routes inside node prose. Only create edges with clear evidence from the trajectory.
"""

INITIALIZER_USER = """\
Analyze this successful trajectory and extract the initial skill graph.

## Task Description
{task_description}

## Trajectory (RESOLVED)
{trajectory_steps}
"""


# ═══════════════════════════════════════════════════════════════════════
# Success Updater
# ═══════════════════════════════════════════════════════════════════════

SUCCESS_UPDATER_SYSTEM = """\
You analyze a successful trajectory and decide how to update an existing \
skill graph. The graph already has nodes and edges from previous trajectories \
of the SAME task — meaning this trajectory solves the same problem, possibly \
with a different approach or additional insights.

## Your job

1. Analyze the trajectory's key decision patterns
2. For each pattern, check if it matches an existing node
3. Produce operations:
   - **update_node**: the pattern matches an existing node — output the node_id \
and what new information to add to the BODY (new actions, refined rationale, etc.)
   - **update_node_card**: the body is still right, but the scan-level card \
should be sharpened (summary, triggers, description)
   - **add_node**: the pattern is genuinely NEW — output full node content
   - **split_node**: an existing node covers two DISTINCT decision situations \
that this trajectory reveals should be separate. Output the node_id to split \
and two new node definitions. Only split when you have clear evidence from \
the trajectory that the two situations are independently useful.
   - **add_edge**: a transition between nodes (new or existing) was observed
   - **update_edge**: an existing edge is real but its routing semantics should \
be refined; use this when the condition should be more precise

## CRITICAL: Distinguish content updates from routing updates

Prefer **update_node** only when all three are still the same:
- same local goal / decision situation
- same core strategy

If the body is right but the scan card is weak or too vague, use \
**update_node_card**.

If the local goal is the same but the route differs, prefer \
**add_node + add_edge** or **update_edge** over stuffing everything into one node.

Likewise for edges: prefer **update_edge** over **add_edge** when the same \
source/target transition already exists but the new trajectory clarifies the \
condition.

If the trajectory shows another viable route for the SAME local problem, \
represent that routing difference explicitly:
- prefer **add_node + add_edge** when a distinct route deserves its own navigable node
- prefer **update_edge** when an existing transition is correct but its \
condition needs refinement
- do NOT hide routing changes only inside update_node prose

Ask yourself: "Would a developer want this represented as its own navigable \
step?" If yes, prefer adding structure over prose accretion.

## Matching rules

A trajectory pattern MATCHES an existing node when:
- They address the same decision situation
- The strategic approach is the same (even if phrased differently)
- An agent applying one would naturally also be applying the other

A pattern is NEW when:
- No existing node covers this decision situation
- The approach is fundamentally different from all existing nodes

## Generalization rules

Generalize, but do not over-sanitize:
- replace repo-specific names, exact paths, and one-off identifiers with <placeholder>
- keep broadly useful repository-domain concepts when they materially help
- avoid making the skill so generic that retrieval becomes vague

For **update_node.new_info**, include only durable decision-level additions:
- good: a stronger trigger, sharper rationale, missing verification, \
  missing branch condition, a short generalized code snippet when it changes \
  the reusable skill
- bad: repo paths, shell transcript fragments, line-number arithmetic, \
  one-off edit cleanup details

## Output format

Return JSON:
{
  "operations": [
    {
      "op": "update_node_card",
      "node_id": "existing-node-id",
      "card": {
        "summary": "sharper scan-level summary",
        "description": "updated retrieval description",
        "triggers": ["updated trigger 1", "updated trigger 2"]
      }
    },
    {
      "op": "update_node",
      "node_id": "existing-node-id",
      "new_info": "what this trajectory adds: new actions, refined rationale, \
additional verification approaches, etc."
    },
    {
      "op": "add_node",
      "node": {
        "node_id": "new-kebab-slug",
        "summary": "one-sentence scan-level card",
        "description": "one-line summary",
        "triggers": ["condition 1", "condition 2"],
        "situation": "...",
        "approach": "...",
        "rationale": "...",
        "actions": ["..."],
        "embedded_code": [
          {
            "section": "approach | actions | verification | situation | rationale",
            "language": "python | bash | text | diff | ...",
            "code": "optional short generalized snippet with <placeholder> identifiers",
            "note": "optional one-line setup sentence"
          }
        ],
        "verification": "..."
      }
    },
    {
      "op": "split_node",
      "node_id": "existing-node-to-split",
      "new_nodes": [
        {
          "node_id": "first-half-slug",
          "summary": "...",
          "description": "...",
          "triggers": ["..."],
          "situation": "...",
          "approach": "...",
          "rationale": "...",
          "actions": ["..."],
          "embedded_code": [
            {
              "section": "approach | actions | verification | situation | rationale",
              "language": "python | bash | text | diff | ...",
              "code": "optional short generalized snippet with <placeholder> identifiers",
              "note": "optional one-line setup sentence"
            }
          ],
          "verification": "..."
        },
        {
          "node_id": "second-half-slug",
          "summary": "...",
          "description": "...",
          "triggers": ["..."],
          "situation": "...",
          "approach": "...",
          "rationale": "...",
          "actions": ["..."],
          "embedded_code": [
            {
              "section": "approach | actions | verification | situation | rationale",
              "language": "python | bash | text | diff | ...",
              "code": "optional short generalized snippet with <placeholder> identifiers",
              "note": "optional one-line setup sentence"
            }
          ],
          "verification": "..."
        }
      ]
    },
    {
      "op": "add_edge",
      "edge": {
        "source": "node-A",
        "target": "node-B",
        "condition": "short routing cue"
      }
    },
    {
      "op": "update_edge",
      "edge": {
        "source": "node-A",
        "target": "node-B",
        "condition": "updated short routing cue"
      }
    }
  ]
}

If the trajectory adds nothing new, return {"operations": []}.
"""

SUCCESS_UPDATER_USER = """\
Analyze this trajectory and decide how to update the existing skill graph.

## Current Graph

{graph_content}

## New Trajectory (RESOLVED)

### Task Description
{task_description}

### Steps
{trajectory_steps}
"""


# ═══════════════════════════════════════════════════════════════════════
# Failure Updater
# ═══════════════════════════════════════════════════════════════════════

FAILURE_UPDATER_SYSTEM = """\
You analyze a FAILED trajectory and identify pitfalls to add to existing \
skill graph nodes. You do NOT add new nodes or edges — only pitfalls.

## Your job

1. Analyze what the failed agent did wrong
2. For each mistake, identify which existing node's situation it relates to
3. Describe the pitfall: what the agent did wrong and why it failed

## Pitfall quality

A good pitfall:
- Names the specific bad behavior (not vague "didn't try hard enough")
- Explains WHY it leads to failure (the causal mechanism)
- Is grounded in what actually happened in the trajectory

## Generalization rules

Pitfalls should generalize the anti-pattern, but they do not need to erase \
useful engineering context.

Hard constraints:
- Avoid exact repo paths or directory names from the task
- Avoid exact project identifiers when a more general term will do
- If specificity is needed, use placeholders such as <repo_path>, \
  <project>, <base_class>, <concrete_type>

## Output format

Return JSON:
{
  "pitfalls": [
    {
      "node_id": "existing-node-id",
      "pitfall": "Doing X instead of the node's recommended approach leads to \
failure because Y. (generalized, reusable)",
      "evidence": "what specifically happened in the failed trajectory"
    }
  ]
}

If no clear pitfalls can be identified, return {"pitfalls": []}.
"""

FAILURE_UPDATER_USER = """\
Analyze this FAILED trajectory and identify pitfalls for existing graph nodes.

## Current Graph

{graph_content}

## Failed Trajectory

### Task Description
{task_description}

### Steps
{trajectory_steps}
"""


# ═══════════════════════════════════════════════════════════════════════
# Task-End Reconciliation
# ═══════════════════════════════════════════════════════════════════════

TASK_RECONCILE_SYSTEM = """\
You review a COMPLETED task-local skill graph and lightly improve its navigation structure.

Your job is to make the graph easier for an agent to SCAN and TRAVERSE.

Focus on two things only:
1. normalize each node's SCAN CARD (`summary`, `description`, `triggers`)
2. refine edge routing cues so branches are clearer

## Important constraints

- Do NOT create new nodes
- Do NOT delete nodes
- Do NOT merge nodes
- Do NOT split nodes
- Do NOT rewrite full markdown bodies
- Prefer no-op over speculative edits

## Node card guidance

Each node should have:
- `summary`: one sentence a small model can scan quickly
- `description`: concise retrieval description
- `triggers`: short observable conditions

## Edge guidance

Edges store only SHORT routing cues, not edge categories.
Prefer adding or updating an edge only when the cue materially helps traversal.

## Output format

Return JSON:
{
  "node_cards": [
    {
      "node_id": "existing-node-id",
      "summary": "scan-level summary",
      "description": "retrieval description",
      "triggers": ["trigger 1", "trigger 2"]
    }
  ],
  "edge_operations": [
    {
      "op": "update_edge",
      "edge": {
        "source": "node-A",
        "target": "node-B",
        "condition": "updated short routing cue"
      }
    },
    {
      "op": "add_edge",
      "edge": {
        "source": "node-A",
        "target": "node-B",
        "condition": "short routing cue"
      }
    }
  ]
}

If nothing clearly needs improvement, return {"node_cards": [], "edge_operations": []}.
"""

TASK_RECONCILE_USER = """\
Review this task-local skill graph and improve only its navigation structure.

## Current Graph

{graph_content}
"""


# ═══════════════════════════════════════════════════════════════════════
# Executor (LLM-based markdown content merging)
# ═══════════════════════════════════════════════════════════════════════

EXECUTOR_UPDATE_SYSTEM = """\
You REWRITE a skill node's markdown to be the best version that combines \
existing content with new information.

## Rules

1. REWRITE, don't append. Produce the most concise version that captures \
both old and new insights. If old and new say similar things differently, \
keep the better phrasing and drop the other.
2. When steps/actions overlap, merge them into one better-worded step \
rather than listing both. "do X" + "also do X but with Y" → \
"do X (including Y when applicable)"
3. Keep YAML frontmatter and preserve/refresh the scan card fields:
   - `name`
   - `summary`
   - `description`
   - `triggers`
   Do NOT drop these fields.
4. Generalize project-specific details, but keep useful engineering language.
   - Replace exact repo-only identifiers with <placeholder> when needed
   - Do not scrub the text into vague generic prose
5. Preserve or add short generalized code snippets only when they materially help \
the skill; place them in the most relevant section instead of creating a dedicated code section
6. Keep granularity tight:
   - Actions should usually stay within 3-6 bullets
   - Keep decision-relevant steps, not command-by-command transcripts
   - Drop line-number arithmetic, sed cleanup minutiae, and shell flags \
     unless they materially change the strategy
7. If new info only adds more specific command syntax for an existing step, \
fold it into broader wording instead of appending another bullet

The body should still clearly cover:
- when to use the skill
- the strategy
- concrete actions
- verification/checks

Use whichever section headings fit best, for example:
- `## When to Use` or `## Situation`
- `## Strategy` or `## Approach`
- `## Actions` or `## Execution`
- `## Checks` or `## Verification`

Output the complete updated markdown content for the node.
"""

EXECUTOR_UPDATE_USER = """\
Update this skill node by integrating the new information.

## Current Node Content

{current_markdown}

## New Information to Integrate

{new_info}

Output the complete updated markdown (no code fences around it).
"""

EXECUTOR_PITFALL_SYSTEM = """\
You integrate a new pitfall into a skill node's markdown content.

## Rules

1. Preserve all existing content
2. If a ## Pitfalls section already exists, integrate the new pitfall into it:
   - If the new pitfall overlaps with an existing one, MERGE them into one \
better-worded pitfall rather than adding a duplicate
   - If the new pitfall is genuinely different, add it
3. If no ## Pitfalls section exists, add one at the end
4. Keep pitfalls concise and actionable
5. Generalize repo-specific details without deleting useful technical context
6. Remove task-specific leakage:
   - Avoid exact repo paths
   - Avoid exact repo-only identifiers when a broader technical term will do
   - Use <placeholder> terms if specificity is needed

Output the complete updated markdown content for the node.
"""

EXECUTOR_PITFALL_USER = """\
Add these pitfall(s) to the skill node.

## Current Node Content

{current_markdown}

## Pitfall(s) to Add

{pitfall}

Output the complete updated markdown (no code fences around it).
"""

EXECUTOR_REFINE_SYSTEM = """\
You lightly refine a skill node's markdown without changing its underlying skill.

## Your job

- Keep the SAME node identity and overall meaning
- Improve granularity and readability
- Compress overly detailed Actions into a smaller set of decision-relevant steps
- Remove task-specific leakage

## Rules

1. Preserve the YAML frontmatter and the node's semantic coverage
1a. Keep the scan card fields (`name`, `summary`, `description`, `triggers`) intact
2. Do NOT invent a new skill, split the node, or merge it with another node
3. Keep Actions compact: usually 3-6 bullets
4. Retain strategic information; remove command-by-command edit/debug minutiae
5. Preserve short embedded code snippets when they are useful; if a snippet is too \
specific, generalize it with <placeholder> identifiers rather than deleting all code
6. Generalize exact repo-specific identifiers and paths, but keep useful \
technical terms when they help retrieval and reuse
7. The refined body should still clearly cover when-to-use, strategy, actions, and checks, \
but headings do not need to follow a rigid template
8. If the node is already concise enough, keep the content very close to the original

Output the complete updated markdown content for the node.
"""

EXECUTOR_REFINE_USER = """\
Lightly refine this skill node for granularity and concision.

## Why This Node Was Flagged

{refine_reason}

## Current Node Content

{current_markdown}

Output the complete updated markdown (no code fences around it).
"""


# ═══════════════════════════════════════════════════════════════════════
# Link Embedding (LLM-based)
# ═══════════════════════════════════════════════════════════════════════

LINK_EMBED_SYSTEM = """\
You embed wikilinks into a skill node's markdown content to connect it to \
related nodes in the skill graph.

## Rules

1. Embed links NATURALLY into the existing prose — do NOT just append a \
"## Links" section at the end.
2. Use [[node-slug]] format.
3. Place links where they are most contextually relevant:
   - In Actions: "If you observe X at this step → [[other-node]]"
   - In Pitfalls: "If you made this mistake → [[recovery-node]]"
   - In Verification: "Once confirmed → [[next-node]]"
   - In Situation/Approach: "If a related condition appears → [[related-node]]"
4. Each link sentence should include the CONDITION under which to follow it.
5. Do NOT change the core content — only INSERT short link sentences.
6. Preserve the YAML frontmatter and all existing sections exactly.
7. Keep it natural and concise — 1-2 sentences per link, max.

Output the complete updated markdown (no code fences around it).
"""

LINK_EMBED_USER = """\
Embed the following links into this skill node's markdown content.

## Current Node Content

{current_markdown}

## Links to Embed

{links_text}

Output the complete updated markdown with links naturally embedded.
"""


# ═══════════════════════════════════════════════════════════════════════
# Assembler (node JSON → markdown)
# ═══════════════════════════════════════════════════════════════════════

def assemble_node_markdown(node: dict) -> str:
    """Convert a node JSON dict from initializer output to markdown."""
    node_id = node.get("node_id", "unnamed")
    summary = (node.get("summary") or node.get("description") or "").strip()
    description = node.get("description", "")
    triggers = node.get("triggers", [])
    situation = node.get("situation", "")
    approach = node.get("approach", "")
    rationale = node.get("rationale", "")
    actions = node.get("actions", [])
    embedded_code = node.get("embedded_code", []) or []
    legacy_code_template = (node.get("code_template", "") or "").strip()
    verification = node.get("verification", "")

    def yaml_quote(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    triggers_yaml = "\n".join(f"  - {yaml_quote(t)}" for t in triggers) if triggers else '  - "(none)"'
    actions_text = "\n".join(f"- {a}" for a in actions) if actions else "- (none)"

    if legacy_code_template and not embedded_code:
        embedded_code = [{
            "section": "approach",
            "language": "text",
            "code": legacy_code_template,
            "note": "Generalized code pattern:",
        }]

    def render_snippets(section: str) -> str:
        blocks = []
        for item in embedded_code:
            if not isinstance(item, dict):
                continue
            if (item.get("section") or "").strip().lower() != section.lower():
                continue
            code = (item.get("code") or "").strip()
            if not code:
                continue
            language = (item.get("language") or "text").strip() or "text"
            note = (item.get("note") or "").strip()
            block = ""
            if note:
                block += f"{note}\n\n"
            block += f"```{language}\n{code}\n```"
            blocks.append(block)
        if not blocks:
            return ""
        return "\n\n" + "\n\n".join(blocks)

    when_to_use_body = f"{situation}{render_snippets('situation')}".strip()

    strategy_parts = []
    if approach.strip():
        strategy_parts.append(approach.strip())
    rationale_text = rationale.strip()
    if rationale_text:
        prefix = "Why this works: "
        strategy_parts.append(prefix + rationale_text)
    rationale_snippets = render_snippets('rationale')
    if rationale_snippets:
        strategy_parts.append(rationale_snippets.strip())
    approach_snippets = render_snippets('approach')
    if approach_snippets:
        strategy_parts.append(approach_snippets.strip())
    strategy_body = "\n\n".join(part for part in strategy_parts if part).strip()

    actions_body = f"{actions_text}{render_snippets('actions')}".strip()
    checks_body = f"{verification}{render_snippets('verification')}".strip()

    body_sections = []
    if when_to_use_body:
        body_sections.append(("When to Use", when_to_use_body))
    if strategy_body:
        body_sections.append(("Strategy", strategy_body))
    if actions_body:
        body_sections.append(("Actions", actions_body))
    if checks_body:
        body_sections.append(("Checks", checks_body))

    body_text = "\n\n".join(
        f"## {heading}\n\n{content}".strip()
        for heading, content in body_sections
        if content.strip()
    )

    return f"""\
---
name: {node_id}
summary: {yaml_quote(summary)}
description: {yaml_quote(description)}
triggers:
{triggers_yaml}
---

# {node_id}

{body_text}
""".strip() + "\n"
