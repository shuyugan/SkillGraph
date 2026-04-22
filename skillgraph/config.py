"""Central configuration for the Skill Graph pipeline."""

from pathlib import Path

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAJECTORIES_DIR = Path(
    "/space3/shuyu/project/skillDAG/harbor/skills-jobs/swebench-claude-hierarchical"
)
SKILLS_DIR = PROJECT_ROOT / "skills"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

# ── LLM ──
MODEL = "gpt-5.2"
