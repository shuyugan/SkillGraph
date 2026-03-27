"""Rule-based classification of action types from bash commands."""

from __future__ import annotations

import re

# Patterns ordered by priority (first match wins)
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("test",     re.compile(r"pytest|python\s+-m\s+(pytest|django\s+test|unittest)|nosetests|test_.*\.py|manage\.py\s+test")),
    ("edit",     re.compile(r"sed\s+-i|patch\s|echo\s+.*[>]|cat\s*<<|printf\s.*>|tee\s|ed\s|perl\s+-.*i")),
    ("verify",   re.compile(r"git\s+diff|python\s+-c\s|python3?\s+.*verify|python3?\s+.*test_fix|python3?\s+.*check")),
    ("search",   re.compile(r"grep|find\s|rg\s|ag\s|locate\s|ack\s")),
    ("read",     re.compile(r"cat\s|head\s|tail\s|less\s|more\s|nl\s|wc\s|file\s")),
    ("navigate", re.compile(r"^(cd|ls|pwd|tree)\s")),
]


def classify_action(command: str) -> str:
    """Classify a bash command into an action type."""
    command_stripped = command.strip()
    for action_type, pattern in _PATTERNS:
        if pattern.search(command_stripped):
            return action_type
    return "other"
