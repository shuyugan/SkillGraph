"""Data models for the Skill Graph pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Step 2a: Per-Trajectory Analysis ──


@dataclass
class Segment:
    """One extracted knowledge segment from a single trajectory."""

    segment_id: str
    trajectory_id: str
    task_id: str
    type: str  # technique | recovery | anti_pattern
    title: str
    description: str  # full description, task-specific at this stage
    code_examples: list[str] = field(default_factory=list)
    context: str = ""  # when/why this was used
    outcome: str = ""  # what happened
    order: int = 0  # position in trajectory


@dataclass
class ErrorRecoveryPair:
    """Links a failed segment to its recovery within one trajectory."""

    failed_segment_id: str
    recovery_segment_id: str
    error_description: str


@dataclass
class TrajectoryAnalysis:
    """Step 2a output: analysis of a single trajectory."""

    trajectory_id: str
    task_id: str
    resolved: bool
    segments: list[Segment] = field(default_factory=list)
    recovery_pairs: list[ErrorRecoveryPair] = field(default_factory=list)


# ── Step 2b: Cross-Trajectory Aggregation ──


@dataclass
class AggregatedItem:
    """A single technique/recovery/anti_pattern after cross-trajectory aggregation."""

    item_id: str
    type: str  # technique | recovery | anti_pattern
    title: str
    description: str  # still task-specific
    code_examples: list[str] = field(default_factory=list)
    source_segment_ids: list[str] = field(default_factory=list)


@dataclass
class Relationship:
    """A relationship discovered during aggregation."""

    rel_type: str  # if_fails | instead_do | needs
    source_item_id: str
    target_item_id: str
    description: str = ""


@dataclass
class TaskSummary:
    """Step 2b output: aggregated knowledge for one task."""

    task_id: str
    items: list[AggregatedItem] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)


# ── Step 2c: Draft Nodes ──


@dataclass
class DraftNode:
    """Step 2c output: a draft markdown node, still task-specific."""

    node_id: str  # slug derived from title
    task_id: str
    type: str  # technique | recovery | anti_pattern
    markdown: str  # full markdown with YAML frontmatter
    source_item_ids: list[str] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)


# ── Step 3+: Finalized Nodes ──


@dataclass
class SkillNode:
    """A finalized node in the skill graph."""

    node_id: str  # slug, e.g. "narrow-search-to-subdirectory"
    type: str  # technique | recovery | anti_pattern | fix_pattern
    markdown: str  # full content with frontmatter + wikilinks
    source_tasks: list[str] = field(default_factory=list)
