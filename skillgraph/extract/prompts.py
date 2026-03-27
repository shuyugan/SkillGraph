"""Prompts for Step 2a (per-trajectory segment extraction) and Step 2b (cross-trajectory aggregation)."""

# ── Step 2a: Per-Trajectory Segment Extraction ──

SEGMENT_EXTRACTION_SYSTEM = """\
You are an expert at analyzing software debugging trajectories. Your job is to \
extract non-trivial, decision-worthy knowledge segments from an agent's trajectory.

## What to extract

- **technique**: A non-obvious strategy or approach that contributed to progress. \
The agent made a deliberate choice that a less experienced engineer might not make.
- **recovery**: An error-recovery sequence where the agent encountered a meaningful \
error and successfully recovered using a non-trivial approach.
- **anti_pattern**: (Only from failed trajectories) A pattern where the agent got \
stuck or wasted significant effort on an ineffective approach.

## What NOT to extract

- Basic file operations: ls, cd, cat, pwd, head, tail
- Simple grep/find without any strategic refinement
- Standard git operations: git diff, git status, git add
- Trivial command corrections (fixing typos)
- Running a command that simply works on the first try without any decision-making

## Output format

Return a JSON object with this structure:
{
  "segments": [
    {
      "type": "technique" | "recovery" | "anti_pattern",
      "title": "Short descriptive title",
      "description": "Full description of what happened, why it was effective/problematic, \
and what the agent did. Include enough detail that someone could learn from this \
without seeing the original trajectory. 3-5 sentences.",
      "code_examples": ["relevant command(s) used"],
      "context": "When/why this technique was used or this error occurred",
      "outcome": "What resulted from this action",
      "order": <position in trajectory, 1-indexed>
    }
  ],
  "error_recovery_pairs": [
    {
      "failed_segment_order": <order of the segment that failed>,
      "recovery_segment_order": <order of the recovery segment>,
      "error_description": "What error occurred and why"
    }
  ]
}

## Rules

- Extract only 2-5 segments per trajectory. Quality over quantity.
- Each segment must represent a meaningful decision or learning opportunity.
- Keep descriptions factual and specific to this trajectory (abstraction happens later).
- Include actual commands in code_examples.
- For recovery segments, clearly describe both the error and the recovery approach.
- For anti_pattern segments (failed trajectories only), describe what the agent \
should have done differently.
"""

SEGMENT_EXTRACTION_USER = """\
Analyze the following trajectory and extract non-trivial knowledge segments.

## Task Description
{task_description}

## Trajectory (Result: {result})
{trajectory_steps}
"""

# ── Step 2b: Cross-Trajectory Aggregation ──

TASK_AGGREGATION_SYSTEM = """\
You are analyzing multiple attempts (trajectories) by an agent to solve the same \
software engineering task. Some attempts succeeded (resolved), some failed.

Your job is to aggregate the per-trajectory analyses and identify:
1. **Common techniques**: Strategies that appear across multiple resolved trajectories.
2. **Common recoveries**: Error-recovery patterns that appear across trajectories.
3. **Anti-patterns**: What failed trajectories did that resolved ones did NOT.
4. **Relationships**: How these items relate to each other.

## Consolidation rule

If two segments consistently appear together as a single debugging workflow \
(e.g., "diagnose X" always precedes "fix X" in the same trajectory), \
merge them into ONE aggregated item rather than two. \
Only create separate items if they are independently useful in different situations.

## Output format

Return a JSON object:
{
  "items": [
    {
      "type": "technique" | "recovery" | "anti_pattern",
      "title": "Short title",
      "description": "Aggregated description combining observations across trajectories. \
Note what was consistent vs what varied. 3-6 sentences.",
      "code_examples": ["representative command(s)"]
    }
  ],
  "relationships": [
    {
      "rel_type": "if_fails" | "instead_do",
      "source_title": "title of source item",
      "target_title": "title of target item",
      "description": "Why this relationship exists"
    }
  ]
}

## Rules for relationships

- **if_fails**: A technique was followed by a recovery when it failed. \
Example: "sed edit" if_fails → "use python script"
- **instead_do**: An anti-pattern (from failed trajectories) has a better alternative \
(from resolved trajectories). Example: "retry same command" instead_do → "switch approach"
- Do NOT create sequential/workflow relationships (e.g., "search then edit then test"). \
These apply to every task and carry no information.
- Only create relationships where the connection was actually observed in the trajectories.
"""

TASK_AGGREGATION_USER = """\
Below are analyses of {n_trajectories} attempts to solve the same task. \
{n_resolved} succeeded, {n_failed} failed.

{trajectory_analyses}

Aggregate these into common techniques, recoveries, anti-patterns, and relationships.
"""


def build_trajectory_steps_text(steps: list, max_steps: int = 30) -> str:
    """Build trajectory text for the extraction prompt."""
    lines = []
    for step in steps[:max_steps]:
        thought_preview = step.thought[:300] if step.thought else "(no thought)"
        action_preview = step.action[:200] if step.action else "(no action)"
        obs_preview = step.observation[:300] if step.observation else "(no output)"
        lines.append(
            f"### Step {step.step_num} [{step.action_type}]\n"
            f"**Thought:** {thought_preview}\n"
            f"**Action:** {action_preview}\n"
            f"**Observation:** {obs_preview}\n"
        )
    if len(steps) > max_steps:
        lines.append(f"... ({len(steps) - max_steps} more steps omitted)")
    return "\n".join(lines)


def build_trajectory_analysis_text(analysis) -> str:
    """Render a TrajectoryAnalysis as text for the aggregation prompt."""
    lines = [
        f"### Trajectory: {analysis.trajectory_id} ({'RESOLVED' if analysis.resolved else 'FAILED'})",
        "",
    ]
    for seg in analysis.segments:
        lines.append(f"**[{seg.type}] {seg.title}** (order: {seg.order})")
        lines.append(f"  Description: {seg.description}")
        if seg.code_examples:
            lines.append(f"  Code: {'; '.join(seg.code_examples[:2])}")
        lines.append(f"  Context: {seg.context}")
        lines.append(f"  Outcome: {seg.outcome}")
        lines.append("")

    if analysis.recovery_pairs:
        lines.append("**Error-Recovery Pairs:**")
        for pair in analysis.recovery_pairs:
            lines.append(
                f"  - Segment {pair.failed_segment_id} failed → "
                f"Segment {pair.recovery_segment_id} recovered: {pair.error_description}"
            )
        lines.append("")

    return "\n".join(lines)
