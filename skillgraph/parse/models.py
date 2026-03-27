"""Data models for parsed trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajectoryStep:
    step_num: int
    thought: str
    action: str               # raw action block (command string)
    action_type: str          # search / read / edit / test / verify / navigate / other
    observation: str


@dataclass
class TrajectoryRecord:
    trajectory_id: str        # e.g. "django__django-11119__6vgXR7E"
    task_id: str              # e.g. "django__django-11119"
    task_description: str
    resolved: bool
    steps: list[TrajectoryStep] = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def repo_name(self) -> str:
        """Extract repo name like 'django' from task_id."""
        return self.task_id.split("__")[0].split("-")[0] if "__" in self.task_id else ""
