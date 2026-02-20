# Hypothesis Experiment Process

This document describes the process for running a hypothesis-driven experiment. For experiment standards (rigor, classification, analysis), see [docs/standards/experiments.md](../standards/experiments.md). For the experiment template, see [docs/templates/hypothesis.md](../templates/hypothesis.md).

## When to Run Experiments

- Validating that a new feature works as designed (post-PR confirmation)
- Testing intuitive claims about system behavior (from `docs/plans/research.md`)
- Investigating unexpected behavior observed during development
- Exploring design tradeoffs between configurations

## Steps

1. **Select or pose hypothesis** — from `docs/plans/research.md` or from a new observation
2. **Classify** — deterministic or statistical? If statistical, which subtype? (See [experiments.md](../standards/experiments.md))
3. **Create worktree** — `git worktree add .worktrees/hypothesis-<name> -b hypothesis/<name>`
4. **Design experiment** — controlled comparison (ED-1), rate awareness (ED-2), precondition verification (ED-3), seed strategy (ED-4)
5. **Implement** — create `hypotheses/<name>/run.sh`, `analyze.py`
6. **Run** — execute across required seeds; verify reproducibility (ED-5)
7. **Analyze** — produce comparison tables, compute effect sizes
8. **Classify findings** — confirmation, bug, new rule, new invariant, design limitation, or surprise
9. **Audit against standards** — check findings against `docs/standards/rules.md` and `docs/standards/invariants.md`
10. **Document** — write `FINDINGS.md` with results, root cause, classification, and audit
11. **File issues** — for any bugs (`--label bug`), design limitations (`--label design`), or new rules/invariants discovered. Every issue must be labeled.
12. **Commit and PR** — rebase on upstream/main, push, create PR

## Quality Gates

- [ ] Hypothesis classified (deterministic or statistical + subtype)
- [ ] Experiment design follows ED-1 through ED-5
- [ ] Results reproducible via `./run.sh`
- [ ] Findings classified per the findings table
- [ ] Standards audit completed
- [ ] Issues filed for all actionable findings

## References

- Standards: [docs/standards/experiments.md](../standards/experiments.md)
- Template: [docs/templates/hypothesis.md](../templates/hypothesis.md)
- Hypothesis catalog: [docs/plans/research.md](../plans/research.md)
- Validated experiments:
  - `hypotheses/h3-signal-freshness/` — signal freshness (Tier 2, statistical dominance)
  - `hypotheses/h8-kv-pressure/` — KV cache pressure (Tier 3, statistical monotonicity)
  - `hypotheses/h9-prefix-caching/` — prefix caching effectiveness (Tier 2, statistical monotonicity)
  - `hypotheses/h12-conservation/` — request conservation invariant (Tier 1, deterministic)
  - `hypotheses/h14-pathological-templates/` — pathological policy templates (Tier 2, statistical dominance)
  - `hypotheses/prefix-affinity/` — prefix-affinity routing (Tier 2, statistical dominance)
