# Skills & Plugins

BLIS development uses Claude Code skills and plugins for experimentation, review, and development workflows. Skills are organized in layers -- from project-specific workflows checked into the repository to general-purpose community plugins installed per-user.

```
# Quick example: invoke the convergence-review skill for a PR plan
/convergence-review pr-plan

# Or start a guided hypothesis experiment
/hypothesis-experiment
```

## Skill Layers

Skills and plugins are organized into five layers, from most specific to most general:

| Layer | Location | Purpose | Key Examples |
|-------|----------|---------|--------------|
| **Project skills** | `.claude/skills/` | BLIS-specific workflows, checked into the repo | `convergence-review` (multi-perspective review), `hypothesis-experiment` (guided experimentation) |
| **BLIS SDLC plugins** | `sdlc-plugins` marketplace | BLIS development lifecycle | `research-ideas` (iterative research ideation), `hypothesis-test` (experiment scaffolding) |
| **Superpowers** | `superpowers-marketplace` | Cross-project development skills | `superpowers` (TDD, debugging, plans, worktrees, brainstorming), `episodic-memory` (conversation search), `elements-of-style` (writing quality) |
| **Official plugins** | `claude-plugins-official` | Anthropic official tools | `commit-commands` (git workflow), `pr-review-toolkit` (PR review), `code-review`, `feature-dev`, `claude-md-management` |
| **Community plugins** | `awesome-claude-plugins` | Community-contributed tools | `audit-project` (multi-agent code review), `bug-fix`, `debugger`, `test-writer-fixer` |

Project skills take precedence -- they encode BLIS-specific conventions (20 antipattern rules, 8 system invariants, convergence protocol) that generic tools do not know about.

## Marketplaces

Marketplaces are curated GitHub repositories that collect related plugins. Claude Code can install plugins directly from these repositories. Each marketplace has a different focus:

- **claude-plugins-official** -- Anthropic's official plugin collection. Maintained by the Claude Code team with stable, well-tested tools for common development tasks.
    - URL: [https://github.com/anthropics/claude-plugins-official](https://github.com/anthropics/claude-plugins-official)
- **superpowers-marketplace** -- Community superpowers for enhanced development workflows. Provides structured approaches to TDD, debugging, worktree management, and brainstorming.
    - URL: [https://github.com/obra/superpowers-marketplace](https://github.com/obra/superpowers-marketplace)
- **awesome-claude-plugins** -- Community-contributed plugins covering diverse use cases. A broad collection including code review, testing, debugging, and documentation tools.
    - URL: [https://github.com/ComposioHQ/awesome-claude-plugins](https://github.com/ComposioHQ/awesome-claude-plugins)
- **sdlc-plugins** -- BLIS project's software development lifecycle plugins. Tailored for BLIS research workflows including hypothesis ideation and experiment scaffolding.
    - URL: [https://github.com/inference-sim/sdlc-plugins](https://github.com/inference-sim/sdlc-plugins)

## Installation

Project skills (`.claude/skills/`) require no installation -- they are checked into the repository and automatically available when you open the project with Claude Code.

For marketplace plugins, install them using the `/install-plugin` command inside Claude Code:

```
/install-plugin https://github.com/anthropics/claude-plugins-official/tree/main/commit-commands
```

Or install an entire marketplace to browse available plugins:

```
/install-plugin https://github.com/obra/superpowers-marketplace
```

Installed plugins persist in `~/.claude/plugins/` and are available across all projects on your machine.

## Which Skills for Which Workflow

Each BLIS development workflow uses a different combination of skills. Required skills are essential for the workflow to function correctly; optional skills enhance but are not strictly necessary.

| Workflow | Required Skills | Optional Skills |
|----------|----------------|-----------------|
| **PR Development** | `superpowers` (worktrees, writing-plans, executing-plans, verification), `commit-commands`, `convergence-review` | `pr-review-toolkit`, `systematic-debugging` |
| **Hypothesis Experiments** | `convergence-review`, `hypothesis-experiment` | `hypothesis-test` (from sdlc-plugins), `commit-commands` |
| **Design Process** | `convergence-review` | `superpowers` (brainstorming) |
| **Research** | `research-ideas` (from sdlc-plugins) | `episodic-memory` |

### PR Development Example

A typical PR uses skills at multiple stages:

1. `/worktree` -- create an isolated working branch (from superpowers)
2. `/writing-plans` -- draft a micro plan with behavioral contracts (from superpowers)
3. `/convergence-review pr-plan` -- multi-perspective plan review (project skill)
4. `/executing-plans` -- implement the plan with TDD (from superpowers)
5. `/convergence-review pr-code` -- multi-perspective code review (project skill)
6. `/commit` -- stage, commit, and push (from commit-commands)

### Hypothesis Experiment Example

A hypothesis experiment uses the dedicated workflow:

1. `/hypothesis-experiment` -- guided Steps 0-10 (project skill)
2. `/convergence-review h-design` -- design review gate (project skill)
3. `/convergence-review h-code` -- code review gate (project skill)
4. `/convergence-review h-findings` -- findings review gate (project skill)

## Project-Level vs User-Level

Skills and plugins live at two levels, each with different scope and persistence:

**Project-level (`.claude/skills/`)** -- Checked into the repository. Automatically available to every developer who clones the project. These encode project-specific knowledge: BLIS convergence protocol parameters, review perspective definitions, experiment workflow steps. Changes go through normal PR review.

**User-level (`~/.claude/plugins/`)** -- Installed per-user on each developer's machine. Must be installed individually. These provide general-purpose capabilities (git workflow, TDD scaffolding, debugging) that are useful across many projects. Not shared through the repository.

The distinction matters for onboarding: project skills work immediately after `git clone`, while user-level plugins require each contributor to run their own installation steps.

## Further Reading

- [PR Workflow](../contributing/pr-workflow.md) -- skill usage in PR development
- [Hypothesis Experiments](../contributing/hypothesis.md) -- skill usage in experimentation
- [Convergence Protocol](../contributing/convergence.md) -- the review protocol that `convergence-review` implements
- [Extension Recipes](../contributing/extension-recipes.md) -- step-by-step guides that skills help execute
