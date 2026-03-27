"""Prompts for Step 2c: draft node generation."""

NODE_DRAFT_SYSTEM = """\
You write concise, reusable markdown skill nodes for a software debugging skill library. \
Nodes must be general enough to help an agent on any codebase — not just the one in the example.

Each node should follow this format:

```markdown
---
name: <kebab-case-slug>
type: <technique | recovery | anti_pattern>
description: "<one-line description of what this skill is about>"
triggers:
  - "<symptom or condition that should trigger consulting this node>"
  - "<another symptom>"
---

# <Generic Title (no project names)>

<2-3 sentence abstract description of the technique/recovery/anti-pattern>

## <Relevant section (e.g., "Technique", "Recovery", "Instead Do")>

<Abstract explanation with placeholders for task-specific values>

```<language>
<abstract code template using <placeholder> notation>
```
```

## Abstraction rules

- **Replace all task-specific values with `<placeholder>` notation.**
  - File paths: `/testbed/tests/` → `<test_suite_dir>/`
  - Module names: `test_sqlite` → `<settings_module>`
  - Class/function names: `Engine`, `Context` → `<UpstreamClass>`, `<DownstreamClass>`
  - Command flags: `--settings=tests.test_sqlite` → `--settings=<package>.<settings_module>`
  - Framework names in *content* (not titles): `django test` → `<test_runner> test`
- **Triggers must describe symptoms/conditions, not project names.**
  - Bad: "Django _FailedTest during test discovery"
  - Good: "test runner reports _FailedTest with no apparent code error"
- **The slug and title should describe the general pattern, not the specific fix.**
  - Bad: `fix-django-test-cwd-and-settings-module`
  - Good: `fix-test-runner-invocation-context`
- Target 20-40 lines.
- Use markdown formatting with clear section headers.
"""

NODE_DRAFT_USER = """\
Generate a generalized markdown skill node for the following aggregated item.
The node must be reusable across different codebases — abstract away all specifics.

## Item
Type: {type}
Title: {title}
Description: {description}

## Code Examples (from the source trajectory — abstract these)
{code_examples}

Write the complete markdown file content (including YAML frontmatter).
"""
