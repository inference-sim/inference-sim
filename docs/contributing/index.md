# Contributing

Welcome to the BLIS contributor guide. This section covers engineering standards, development workflows, and extension recipes for building on BLIS.

## Quick Start

```bash
# Build
go build -o blis main.go

# Test
go test ./...

# Lint (install once: go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.9.0)
golangci-lint run ./...
```

All three must pass before submitting a PR. CI runs on every PR (see `.github/workflows/ci.yml`).

## Your First Contribution

See [CONTRIBUTING.md](https://github.com/inference-sim/inference-sim/blob/main/CONTRIBUTING.md) for a step-by-step walkthrough that adds a trivial admission policy — the lightest extension type (~3 files).

## Development Workflows

| Workflow | When to Use |
|----------|-------------|
| [PR Workflow](pr-workflow.md) | Every PR: worktree → plan → review → implement → audit → commit |
| [Design Process](design-process.md) | New features that introduce module boundaries |
| [Macro Planning](macro-planning.md) | Multi-PR features requiring decomposition |
| [Hypothesis Experiments](hypothesis.md) | Rigorous experiments to validate simulator behavior |
| [Convergence Protocol](convergence.md) | Review gate used by all workflows above |

## Extension Recipes

[Extension Recipes](extension-recipes.md) — Step-by-step guides for adding policies, scorers, KV tiers, trace records, and per-request metrics.

## Standards

| Document | Covers |
|----------|--------|
| [Antipattern Rules (R1-R20)](standards/rules.md) | 20 rules, each tracing to a real bug |
| [System Invariants (INV-1-8)](standards/invariants.md) | Properties that must always hold |
| [Engineering Principles](standards/principles.md) | Separation of concerns, interface design, BDD/TDD |
| [Experiment Standards](standards/experiments.md) | Hypothesis families, rigor requirements |

## Templates

| Template | Purpose | Agent Prompt |
|----------|---------|--------------|
| [Design Guidelines](templates/design-guidelines.md) | DES foundations, module architecture, extension framework | N/A (reference material) |
| [Macro Plan](templates/macro-plan.md) | Multi-PR feature decomposition | [`macro-plan-prompt.md`](templates/macro-plan-prompt.md) |
| [Micro Plan](templates/micro-plan.md) | Single-PR implementation with TDD tasks | [`micro-plan-prompt.md`](templates/micro-plan-prompt.md) |
| [Hypothesis](templates/hypothesis.md) | Experiment FINDINGS.md structure | N/A (template is audience-neutral) |

Templates describe the output format (what sections to include and why). Agent prompts contain LLM-specific instructions for generating those artifacts. Claude Code skills reference the prompt files automatically.
