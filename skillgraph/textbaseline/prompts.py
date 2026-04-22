"""Prompts for the text-based markdown skill baseline."""

TRAJECTORY_ANALYZER_SYSTEM = """\
You analyze ONE software-debugging trajectory and extract durable reusable insights.

The output is NOT a graph. It is a structured summary that will later be merged into:
- a task-level memo
- a repository/domain-level markdown skill file

Generalization guidance:
- Generalize exact repo paths, one-off identifiers, and purely task-specific names
- Keep useful engineering language when it materially clarifies the pattern
- Do not scrub the result into vague generic prose

Return JSON:
{
  "trajectory_summary": "1-2 sentence summary of what this trajectory contributed",
  "problem_signals": [
    "observable symptom or trigger"
  ],
  "localization_patterns": [
    {
      "title": "short title",
      "when_useful": "when this pattern is useful",
      "pattern": "the reusable localization idea",
      "actions": ["short action", "short action"],
      "checks": ["short check"]
    }
  ],
  "fix_patterns": [
    {
      "title": "short title",
      "when_useful": "when this fix pattern applies",
      "pattern": "the reusable fix idea",
      "actions": ["short action", "short action"],
      "checks": ["short check"]
    }
  ],
  "verification_patterns": [
    {
      "title": "short title",
      "when_useful": "when to use this verification strategy",
      "pattern": "the reusable verification idea",
      "actions": ["short action", "short action"],
      "checks": ["short check"]
    }
  ],
  "pitfalls": [
    "generalized anti-pattern or failure mode"
  ],
  "notable_snippets": [
    {
      "language": "python | bash | diff | text",
      "code": "optional short generalized snippet using <placeholder> identifiers",
      "note": "why this snippet is worth keeping"
    }
  ]
}

Keep lists short and high-signal. If a category has nothing useful, return [].
"""

TRAJECTORY_ANALYZER_USER = """\
Analyze this trajectory and extract reusable insights.

## Task Description
{task_description}

## Trajectory
{trajectory_steps}
"""


TASK_MEMO_INIT_SYSTEM = """\
You create an initial task memo from the first successful trajectory insights for one task.

The task memo is a compact markdown document, not a graph. It should be useful as
an intermediate artifact for later repository-level skill consolidation.

Required structure:

# Task Memo: <task_id>

## Task Summary

## Problem Signals

## Localization Patterns

## Fix Patterns

## Verification Patterns

## Pitfalls

## Notable Snippets

Rules:
- Keep it compact and readable
- Use bullets and short subsections
- Keep broadly useful engineering language when it helps
- Avoid raw transcript style
- Avoid repo paths and one-off identifiers

Output complete markdown only.
"""

TASK_MEMO_INIT_USER = """\
Create the initial task memo.

## Task ID
{task_id}

## Domain
{domain}

## Trajectory Insights
{trajectory_insights_json}
"""


TASK_MEMO_UPDATE_SYSTEM = """\
You update an existing task memo with insights from another successful trajectory of the SAME task.

Your job:
- merge overlapping content
- add genuinely new localization/fix/verification patterns
- sharpen wording when the new insights are better
- keep the memo compact instead of endlessly appending

Rules:
- preserve the same section structure
- prefer deduplication and consolidation
- do not turn the memo into a raw case log
- keep broadly useful engineering language when it helps

Output complete updated markdown only.
"""

TASK_MEMO_UPDATE_USER = """\
Update this task memo with the new successful trajectory insights.

## Current Task Memo
{current_memo}

## New Trajectory Insights
{trajectory_insights_json}
"""


TASK_MEMO_PITFALL_SYSTEM = """\
You update an existing task memo using insights from a failed trajectory of the SAME task.

Only integrate:
- new pitfalls / anti-patterns
- brief warnings that sharpen existing patterns

Do not add new sections. Keep the memo compact.

Output complete updated markdown only.
"""

TASK_MEMO_PITFALL_USER = """\
Update this task memo with failed-trajectory insights.

## Current Task Memo
{current_memo}

## Failed Trajectory Insights
{trajectory_insights_json}
"""


TASK_MEMO_REFINE_SYSTEM = """\
You lightly refine a task memo for readability and compactness.

Rules:
- preserve the same meaning
- preserve the same section structure
- compress repetitive bullets
- keep useful engineering language
- do not remove distinctive high-signal patterns

Output complete updated markdown only.
"""

TASK_MEMO_REFINE_USER = """\
Refine this task memo for readability and compactness.

{current_memo}
"""


DOMAIN_FILE_INIT_SYSTEM = """\
You create the initial repository/domain markdown skill file from one task memo.

Required structure:

# <Domain> Text Skill Baseline

## Overview

## Problem Signals

## Localization Patterns

## Fix Patterns

## Verification Patterns

## Pitfalls

## Reusable Snippets

Rules:
- this is a SINGLE markdown file baseline
- keep section structure stable
- keep it compact and navigable with headings/bullets
- no graph structure, no node references, no edge references

Output complete markdown only.
"""

DOMAIN_FILE_INIT_USER = """\
Create the initial domain markdown skill file.

## Domain
{domain}

## Source Task Memo
{task_memo}
"""


DOMAIN_FILE_UPDATE_SYSTEM = """\
You update an existing repository/domain markdown skill file with one more task memo.

Your job:
- merge overlapping patterns into existing sections
- add genuinely new patterns where needed
- keep the document compact and organized
- prefer section-level consolidation over endless append-only growth

Rules:
- keep the same section structure
- no graph structure
- no edge language
- avoid raw task-specific notes
- retain useful engineering/domain language when it helps retrieval

Output complete updated markdown only.
"""

DOMAIN_FILE_UPDATE_USER = """\
Update this domain markdown skill file with the new task memo.

## Current Domain File
{current_domain_markdown}

## New Task Memo
{task_memo}
"""


DOMAIN_FILE_REFINE_SYSTEM = """\
You lightly consolidate a repository/domain markdown skill file.

Your job:
- deduplicate repeated bullets
- merge overlapping subsections
- shorten overlong prose
- keep the file useful for retrieval at inference time

Rules:
- preserve the same section structure
- do not remove distinctive useful patterns
- keep the file as one coherent markdown document

Output complete updated markdown only.
"""

DOMAIN_FILE_REFINE_USER = """\
Consolidate this domain markdown skill file.

{current_domain_markdown}
"""

