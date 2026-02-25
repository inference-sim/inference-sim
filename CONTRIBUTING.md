# Contributing to BLIS

This guide covers the engineering standards that keep BLIS (Blackbox Inference Simulator) correct and maintainable.

## Quick Start

```bash
# Build
go build -o simulation_worker main.go

# Test
go test ./...

# Install linter (one-time setup)
go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.9.0

# Lint
golangci-lint run ./...
```

All three must pass before submitting a PR. CI uses golangci-lint v2.9.0 (see `.github/workflows/ci.yml`).

## Your First Contribution

This walkthrough adds a trivial admission policy — the lightest extension type (~3 files). Follow it step-by-step to learn the patterns, then apply them to your own contribution.

**What we'll build:** A `CountingAdmit` admission policy that admits the first N requests and rejects the rest. We'll use test-driven development, starting with a test for the feature we want to implement.

### Step 1: Create a branch

```bash
git checkout -b feature/counting-admit
```

### Step 2: Write the failing test

Add a test to `sim/admission_test.go`:

```go
func TestCountingAdmit_RejectsAfterLimit(t *testing.T) {
	// GIVEN a CountingAdmit policy with limit=2
	policy := &CountingAdmit{Limit: 2}
	req := &Request{ID: "test", InputTokens: make([]int, 3)}
	state := &RouterState{Clock: 0}

	// WHEN 3 requests arrive
	r1, _ := policy.Admit(req, state)
	r2, _ := policy.Admit(req, state)
	r3, reason := policy.Admit(req, state)

	// THEN the first 2 are admitted and the 3rd is rejected
	if !r1 {
		t.Error("first request should be admitted")
	}
	if !r2 {
		t.Error("second request should be admitted")
	}
	if r3 {
		t.Errorf("third request should be rejected, got reason: %s", reason)
	}
}
```

Run: `go test ./sim/... -run TestCountingAdmit -v`
Expected: **FAIL** (type `CountingAdmit` does not exist yet)

### Step 3: Implement the policy

In `sim/admission.go`, add after the existing policies:

```go
// CountingAdmit admits the first Limit requests, then rejects all subsequent ones.
type CountingAdmit struct {
	Limit int
	count int
}

func (c *CountingAdmit) Admit(_ *Request, _ *RouterState) (bool, string) {
	c.count++
	if c.count <= c.Limit {
		return true, ""
	}
	return false, "counting-admit limit exceeded"
}
```

### Step 4: Register in the factory

Two files need changes:

In `sim/bundle.go`, add `"counting-admit"` to the `validAdmissionPolicies` map:

```go
validAdmissionPolicies = map[string]bool{"": true, "always-admit": true, "token-bucket": true, "reject-all": true, "counting-admit": true}
```

In `sim/admission.go`, add a case to the `NewAdmissionPolicy` factory switch:

```go
case "counting-admit":
    return &CountingAdmit{Limit: 100} // hardcoded for tutorial simplicity
```

> **Note:** In a real policy, you would wire the limit through the factory parameters (e.g., `Limit: int(capacity)`) or via `PolicyBundle` YAML config. Hardcoded defaults would fail code review — see how `token-bucket` uses `capacity` and `refillRate`.

### Step 5: Verify tests pass

```bash
go test ./sim/... -run TestCountingAdmit -v   # Your new test
go test ./...                                    # All tests still pass
golangci-lint run ./...                          # No lint issues
```

### Step 6: Commit and open a PR

```bash
git add sim/admission.go sim/admission_test.go sim/bundle.go
git commit -m "feat(sim): add counting-admit admission policy

- Admits first N requests, rejects the rest
- Registered in factory with default limit=100"
git push -u origin feature/counting-admit
gh pr create --title "feat: add counting-admit admission policy" --body "My first BLIS contribution!"
```

**That's it!** You've added a complete, tested, registered policy. Real contributions follow the same pattern — just with more contracts and a formal implementation plan.

> **Important:** This example is for learning only. Do **not** submit this as a real PR — `CountingAdmit` is a toy policy with no practical use. For your actual first contribution, check [open issues](https://github.com/inference-sim/inference-sim/issues) for tasks labeled `good first issue`.

## Contributing with Claude Code

> **Canonical source:** [`docs/process/pr-workflow.md`](docs/process/pr-workflow.md). If this section diverges, pr-workflow.md is authoritative.

BLIS development workflows are orchestrated through [Claude Code](https://claude.ai/code) skills — structured sequences that handle worktree creation, plan generation, multi-perspective review with convergence enforcement, and PR creation. Contributors with Claude Code get the full automated pipeline. Contributors without it follow the manual path below and still go through the same quality gates (maintainers run the automated reviews on submitted PRs).

**Prerequisites:** Claude Code installed with project skills available (`convergence-review`, `hypothesis-experiment`) and general Claude Code skills (`writing-plans`, `executing-plans`, `commit-push-pr`). See [`docs/process/pr-workflow.md`](docs/process/pr-workflow.md) for the full skill table. Before your first contribution, read [`docs/templates/design-guidelines.md`](docs/templates/design-guidelines.md) — it covers module architecture, extension types, and DES foundations.

### Choosing Your Journey

| You want to... | Journey | Starts with |
|---|---|---|
| Fix a bug or make a small change | [Bug Fix / Small Change](#bug-fix--small-change) | A GitHub issue or observed bug |
| Add a new policy, scorer, or extension | [New Policy or Extension](#new-policy-or-extension) | An existing interface to implement |
| Build a new feature or subsystem | [New Feature (Idea to PR)](#new-feature-idea-to-pr) | An idea or requirement |
| Validate simulator behavior | [Hypothesis Experiment](#running-or-contributing-hypothesis-experiments) | A behavioral prediction |

For hypothesis experiments, see [Running or Contributing Hypothesis Experiments](#running-or-contributing-hypothesis-experiments) below. With Claude Code, the `hypothesis-experiment` skill orchestrates the full Steps 0–10 workflow.

### Bug Fix / Small Change

The lightest path. For bug fixes, docs updates, and single-PR changes that don't introduce new module boundaries.

1. **Create worktree** — `/superpowers:using-git-worktrees fix-<name>`
2. **Write micro plan** — `/superpowers:writing-plans` using `@docs/templates/micro-plan.md`
3. **Review plan** — `/pr-review-toolkit:review-pr` then `/convergence-review pr-plan <plan-path>`
4. **Human approval** — review contracts and tasks, approve to proceed
5. **Implement** — `/superpowers:executing-plans @<plan-path>`
6. **Review code** — `/pr-review-toolkit:review-pr` then `/convergence-review pr-code`
7. **Self-audit + commit** — deliberate critical thinking, then `/commit-commands:commit-push-pr`

Full process: [`docs/process/pr-workflow.md`](docs/process/pr-workflow.md)

### New Policy or Extension

For adding a routing policy, admission policy, scorer, scheduler, priority policy, or tier composition — anything behind an existing interface.

1. **Identify extension type** — see [Adding New Components](#adding-new-components) below
2. **Create worktree** — `/superpowers:using-git-worktrees <extension-name>`
3. **Write micro plan** — `/superpowers:writing-plans` using `@docs/templates/micro-plan.md` and `@docs/extension-recipes.md`
4. **Follow steps 3–7 from Bug Fix** (review → approve → implement → review → commit)

No design doc needed for policy templates. For tier compositions, a design doc is recommended — see the extension type table in [Adding New Components](#adding-new-components). Full process: [`docs/process/pr-workflow.md`](docs/process/pr-workflow.md)

### New Feature (Idea to PR)

The full pipeline for features that introduce new module boundaries, new interfaces, or span multiple PRs.

**Phase 1 — Idea to Design:**
1. **Explore approaches** — discuss design options with Claude, settle on an approach
2. **Write design doc** — following [`docs/templates/design-guidelines.md`](docs/templates/design-guidelines.md)
3. **Review design** — `/convergence-review design <path>` (8 perspectives)
4. **Human approval** — review design doc before planning begins

Full process: [`docs/process/design.md`](docs/process/design.md)

**Phase 2 — Design to Macro Plan** (skip if single-PR):

5. **Write macro plan** — decompose into PRs following [`docs/templates/macro-plan.md`](docs/templates/macro-plan.md)
6. **Review macro plan** — `/convergence-review macro-plan <path>` (8 perspectives)
7. **Human approval** — review PR decomposition and module contracts

Full process: [`docs/process/macro-plan.md`](docs/process/macro-plan.md)

**Phase 3 — Plan to PR** (repeat for each PR):

8. **Follow the Bug Fix journey** (steps 1–7) using the macro plan section or design doc as the source document

Each phase produces an artifact that feeds the next. Human approval gates between phases prevent wasted work.

### Without Claude Code

If you are not using Claude Code, here is the simplified workflow:

1. **Branch** — `git checkout -b feature/my-change`
2. **Plan** — write an implementation plan following `docs/templates/micro-plan.md`. Include behavioral contracts (GIVEN/WHEN/THEN) and a task breakdown. Post the plan as a PR draft or issue comment for review.
3. **Implement** — follow TDD: write a failing test, implement the minimal code to pass it, run `go test ./...`, run `golangci-lint run ./...`, commit. Repeat for each contract.
4. **Self-review** — check the [Antipattern Checklist](#antipattern-checklist) below. Run `go build ./... && go test ./... && golangci-lint run ./...` one final time.
5. **PR** — push your branch and open a PR. Maintainers will run the automated review protocols (convergence-review with 10 perspectives).

The automated review tools (convergence-review, pr-review-toolkit) are run by maintainers — you do not need Claude Code installed. Your PR will go through the same quality gates regardless of tooling.

For design docs and macro plans: follow the same templates ([`docs/templates/design-guidelines.md`](docs/templates/design-guidelines.md), [`docs/templates/macro-plan.md`](docs/templates/macro-plan.md)) and submit for review. Maintainers will run convergence review.

Full process: [`docs/process/pr-workflow.md`](docs/process/pr-workflow.md) (the same workflow applies regardless of tooling)

## Engineering Principles

See [`docs/standards/principles.md`](docs/standards/principles.md) for the full principles guide covering: separation of concerns, interface design, configuration design, canonical constructors, output channel separation, error handling boundaries, and BDD/TDD development.

Key points for new contributors:
- `sim/` is a library — never call `os.Exit` or `logrus.Fatalf`. Return errors. Only `cmd/` may terminate.
- Write behavioral contracts (GIVEN/WHEN/THEN) before tests. Test observable behavior, not internal structure.
- If your PR touches request lifecycle, KV cache, or metrics, add or extend invariant tests (see [`docs/standards/invariants.md`](docs/standards/invariants.md)).

## Antipattern Checklist

20 rules, each tracing to a real bug. See [`docs/standards/rules.md`](docs/standards/rules.md) for full details.

Before submitting a PR, verify:

- [ ] R1: No silent `continue`/`return` dropping data
- [ ] R2: Map keys sorted before float accumulation or ordered output
- [ ] R3: Every new CLI flag validated (zero, negative, NaN, Inf)
- [ ] R4: All struct construction sites audited for new fields
- [ ] R5: Resource allocation loops handle mid-loop failure with rollback
- [ ] R6: No `logrus.Fatalf` or `os.Exit` in `sim/` packages
- [ ] R7: Invariant tests alongside any golden tests
- [ ] R8: No exported mutable maps
- [ ] R9: `*float64` for YAML fields where zero is valid
- [ ] R10: YAML strict parsing (`KnownFields(true)`)
- [ ] R11: Division by runtime-derived denominators guarded
- [ ] R12: Golden dataset regenerated if output changed
- [ ] R13: New interfaces work for 2+ implementations
- [ ] R14: No method spans multiple module responsibilities
- [ ] R15: Stale PR references resolved
- [ ] R16: Config params grouped by module
- [ ] R17: Routing scorer signals documented for freshness tier
- [ ] R18: CLI flag values not silently overwritten by defaults.yaml
- [ ] R19: Unbounded retry/requeue loops have circuit breakers
- [ ] R20: Detectors and analyzers handle degenerate inputs (empty, skewed, zero)

## Adding New Components

BLIS has four extension types. Identify which type your change is, then follow the corresponding recipe. See `docs/templates/design-guidelines.md` Section 5 for full details.

| Extension Type | What It Is | Design Doc Required? | Example |
|---|---|---|---|
| **Policy Template** | New algorithm behind an existing interface | No | New routing algorithm |
| **Subsystem Module** | New module with its own interface and events | Yes | AutoScaler, P/D disaggregation |
| **Backend Swap** | Alternative implementation of internal module | Yes (covers both phases) | SGLang latency model |
| **Tier Composition** | Wrapper layering behavior on existing module | Recommended | NVMe KV tier |

### Policy Template (lightest — ~3 files)

1. Implement the interface in the corresponding file (`sim/admission.go`, `sim/routing.go`, `sim/priority.go`, `sim/scheduler.go`)
2. Register in `sim/bundle.go` (valid names map + `IsValid*` function)
3. Add `case` to factory function
4. Add behavioral tests (`TestMyPolicy_Scenario_Behavior`)
5. Update CLAUDE.md and README

### Subsystem Module (heaviest — new interface + integration)

Requires a design doc defining the module contract (observes / controls / owns / invariants / events / extension friction). See design guidelines Section 5.3.

1. Write design doc with module contract, event integration, state ownership, failure modes, default behavior
2. Create implementation plan via `docs/templates/micro-plan.md`
3. Implement interface + default implementation + factory
4. Integrate into cluster event pipeline
5. Add CLI flags with full validation
6. Add behavioral tests + invariant tests
7. Update CLAUDE.md, README, and design guidelines module map if needed

### Backend Swap (two phases — extract interface, then add alternative)

**Phase A (refactoring):** Extract interface from hardcoded logic, verify existing tests pass unchanged.
**Phase B (extension):** Implement new backend behind extracted interface, add configuration to select between backends.

See design guidelines Section 5.4 for the full two-phase recipe.

### Tier Composition (delegation pattern — ~4 files)

1. Implement the same interface as the inner module (Liskov substitution)
2. Compose existing tiers using delegation pattern
3. Update factory with validation
4. Add CLI flags with validation (zero, negative, NaN/Inf guards)
5. Aggregate metrics from all tiers
6. Add conservation invariant tests

### New Trace Record Type

1. Define record struct in `sim/trace/record.go` (pure data, no `sim/` dependency)
2. Add slice field to `SimulationTrace`
3. Add recording method
4. Hook into cluster event pipeline (`if cs.trace != nil`)
5. Update `Summarize()` aggregation
6. Add behavioral tests

## Running or Contributing Hypothesis Experiments

> **Canonical source:** [`docs/process/hypothesis.md`](docs/process/hypothesis.md). If this section diverges, hypothesis.md is authoritative.

BLIS uses hypothesis-driven experimentation to validate system behavior, surface bugs, and document design tradeoffs. Experiments are organized into 6 families (workload/arrival, scheduler invariants, performance-regime, structural model, robustness, cross-policy comparative).

**To run existing experiments:**
```bash
cd hypotheses/h13-determinism
./run.sh
```
See `hypotheses/README.md` for the full list and coverage gaps.

**To propose a new hypothesis:**
File a GitHub issue using the "Hypothesis Proposal" template. Include: the hypothesis sentence, family, diagnostic value, and rough experiment design.

**To implement and run a new experiment:**
Follow `docs/process/hypothesis.md` for the full process (Steps 0-10). Key phases:
1. Create worktree, classify hypothesis, design experiment
2. **Design Review** (5 perspectives) → convergence → **human approval**
3. Implement `run.sh` and `analyze.py` using shared harness (`hypotheses/lib/`)
4. **Code Review** (5 perspectives) → convergence
5. Run experiments, document FINDINGS.md
6. **FINDINGS Review** (10 perspectives) → convergence
7. Self-audit (6 dimensions), verification gate, commit and PR

**Review protocol:** Three review gates at different lifecycle stages, each using the universal convergence protocol (zero CRITICAL + zero IMPORTANT from all reviewers). External contributors without AI review infrastructure should submit their artifacts via PR — maintainers will run the review protocols. Only standard-library Python packages are needed (json, math, re, sys, pathlib).

| Document | Purpose |
|---|---|
| `hypotheses/README.md` | Existing experiments, coverage gaps |
| `docs/process/hypothesis.md` | Full process (Steps 0-10, review gates, convergence protocol) |
| `docs/standards/experiments.md` | Rigor requirements (families, types, VV&UQ, RCV rules) |
| `docs/templates/hypothesis.md` | FINDINGS.md template |

## Code Style

- Composition over inheritance
- Timestamp-based event ordering via min-heap
- Partitioned RNG per subsystem for deterministic isolation
- BDD-style test naming: `TestType_Scenario_Behavior`
- Conventional commits: `feat(scope)`, `fix(scope)`, `refactor(scope)`, `test(scope)`, `docs(scope)`

## Key References

| Document | What It Covers | When to Read |
|---|---|---|
| `CLAUDE.md` | Code architecture, file organization, CLI flags, compact rule/invariant tables | Always — authoritative for current codebase state |
| `docs/standards/rules.md` | 20 antipattern rules with evidence, checks, enforcement | When reviewing or writing code |
| `docs/standards/invariants.md` | 8 system invariants (INV-1 through INV-8) with verification strategies | When touching request lifecycle, KV cache, or metrics |
| `docs/standards/experiments.md` | Experiment taxonomy, rigor requirements, findings classification | When running hypothesis experiments |
| `docs/process/pr-workflow.md` | End-to-end PR lifecycle (worktree → plan → review → implement → audit → PR) | Before starting any PR |
| `docs/design/` | System architecture, core engine, concepts glossary, configuration reference | When learning how BLIS works before contributing |
| `docs/templates/design-guidelines.md` | DES foundations, module architecture, extension framework | Before designing a new feature or extending BLIS |
| `docs/templates/micro-plan.md` | Template for single-PR implementation plans | When creating any PR implementation plan |
| `docs/templates/macro-plan.md` | Template for multi-PR feature expansions | When planning a large feature with multiple PRs |
