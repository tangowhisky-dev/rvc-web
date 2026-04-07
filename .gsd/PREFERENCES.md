---
version: 1
cmux:
  enabled: true
  notifications: true
  sidebar: true
  splits: false
  browser: false
---

# GSD Skill Preferences

Project-specific guidance for skill selection and execution preferences.

See `~/.gsd/agent/extensions/gsd/docs/preferences-reference.md` for full field documentation and examples.

## Fields

- `always_use_skills`: Skills that must be available during all GSD operations
- `prefer_skills`: Skills to prioritize when multiple options exist
- `avoid_skills`: Skills to minimize or avoid (with lower priority than prefer)
- `skill_rules`: Context-specific rules (e.g., "use tool X for Y type of work")
- `custom_instructions`: Append-only project guidance (do not override system rules)
- `models`: Model preferences for specific task types
- `skill_discovery`: Automatic skill detection preferences
- `auto_supervisor`: Supervision and gating rules for autonomous modes

## Examples

```yaml
prefer_skills:
  - playwright
  - resolve_library
avoid_skills:
  - subagent  # prefer direct execution in this project

custom_instructions:
  - "Always verify with browser_assert before marking UI work done"
  - "Use Context7 for all library/framework decisions"
```
