# Skills & Plugins

BLIS development uses Claude Code skills for review workflows. Skills are checked into the repository and automatically available when you open the project with Claude Code.

```
# Review a PR
@claude /blis-pr-review

# Validate an issue
@claude /issue-review
```

## Project Skills

| Skill | Location | Purpose |
|-------|----------|---------|
| `blis-pr-review` | `.claude/skills/blis-pr-review/` | PR review: correctness, invariants (INV-1–INV-13), rules (R1–R23), cross-path parity, preemption safety, Q/A probing phase |
| `issue-review` | `.claude/skills/issue-review/` | Issue validation: VALID / NEEDS WORK / SUPERSEDED / DUPLICATE verdict |

Project skills require no installation — they are checked into the repository and automatically available.

## CI Integration

Both skills are triggered via GitHub Actions (`@claude /blis-pr-review` on PRs, `@claude /issue-review` on issues). Additionally, `/archon-pr-review` runs as a separate CI action for structural architecture review (boundary moves, surface changes, edge deltas). See [Archon PR Review](archon-pr-review.md).

## Which Skills for Which Workflow

| Workflow | Skills Used |
|----------|-------------|
| **Bug fix / small feature** | `@claude /blis-pr-review` + `/archon-pr-review` (FAST-TRACK) |
| **Large feature (new boundaries)** | Same review skills, plus archon `--plan` flag for dist tracking |
| **Issue validation** | `@claude /issue-review` |

## Optional Plugins (user-level)

General-purpose Claude Code plugins can be installed per-user for enhanced workflows:

- `superpowers` — worktrees, writing-plans, executing-plans, TDD, debugging
- `commit-commands` — git workflow (commit, push, PR creation)
- `pr-review-toolkit` — generic PR review

Install from marketplaces:
```
/install-plugin https://github.com/anthropics/claude-plugins-official/tree/main/commit-commands
```

Installed plugins persist in `~/.claude/plugins/` and are available across all projects.
