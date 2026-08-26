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
| [RFC Template](rfc.md) | Large features: writing the tracking issue with holes/surfaces/contracts |
| [RFC to Plan](../templates/rfc-to-plan.md) | After RFC agreement: encode .archon, create sub-issues |
| [PR Workflow](pr-workflow.md) | Every PR: worktree → plan → implement → review → commit |
| [Hypothesis Experiments](hypothesis.md) | Rigorous experiments to validate simulator behavior |

## Extension Recipes

[Extension Recipes](extension-recipes.md) — Step-by-step guides for adding policies, scorers, KV tiers, trace records, and per-request metrics.

## Standards

| Document | Covers |
|----------|--------|
| [Antipattern Rules (R1-R23)](standards/rules.md) | 23 rules, each tracing to a real bug |
| [System Invariants (INV-1-INV-13)](standards/invariants.md) | Properties that must always hold |
| [Engineering Principles](standards/principles.md) | Separation of concerns, interface design, BDD/TDD |
| [Experiment Standards](standards/experiments.md) | Hypothesis families, rigor requirements |
| [Agent Trust Boundaries](standards/agent-trust.md) | Three trust tiers for agent operations |

## Templates

| Template | Purpose |
|----------|---------|
| [Design Guidelines](templates/design-guidelines.md) | DES foundations, module architecture, extension framework |
| [RFC to Plan](../templates/rfc-to-plan.md) | Claude prompt: encode .archon from agreed RFC + create sub-issues |
