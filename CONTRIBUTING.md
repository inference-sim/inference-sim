# Contributing to BLIS

This guide covers the engineering standards that keep BLIS (Blackbox Inference Simulator) correct and maintainable.

## Quick Start

```bash
# Build
go build -o blis main.go

# Test
go test ./...

# Install linter (one-time setup)
go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.9.0

# Lint
golangci-lint run ./...
```

All three must pass before submitting a PR. CI uses golangci-lint v2.9.0 (see `.github/workflows/ci.yml`).

```bash
# Local docs preview (requires Python + mkdocs-material)
pip install mkdocs-material==9.7.3
# Contributing docs served directly at docs/contributing/index.md
mkdocs serve
```

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

> **Canonical source:** [`docs/contributing/pr-workflow.md`](docs/contributing/pr-workflow.md). If this section diverges, pr-workflow.md is authoritative.

BLIS development workflows are orchestrated through [Claude Code](https://claude.ai/code) skills — structured sequences that handle worktree creation, plan generation, multi-perspective review with convergence enforcement, and PR creation. Contributors with Claude Code get the full automated pipeline. Contributors without it follow the manual path below and still go through the same quality gates (maintainers run the automated reviews on submitted PRs).

**Prerequisites:** Claude Code installed with project skills available (`blis-pr-review`, `issue-review`) and general Claude Code skills (`writing-plans`, `executing-plans`, `commit-push-pr`). See [`docs/contributing/pr-workflow.md`](docs/contributing/pr-workflow.md) for the full workflow. Before your first contribution, read [`docs/contributing/templates/design-guidelines.md`](docs/contributing/templates/design-guidelines.md) — it covers module architecture, extension types, and DES foundations.

### Choosing Your Journey

| You want to... | Journey | Starts with |
|---|---|---|
| Fix a bug or make a small change | [Bug Fix / Small Change](#bug-fix--small-change) | A GitHub issue or observed bug |
| Add a new policy, scorer, or extension | [New Policy or Extension](#new-policy-or-extension) | An existing interface to implement |
| Build a new feature or subsystem | [New Feature (Idea to PR)](#new-feature-idea-to-pr) | An idea or requirement |

### Bug Fix / Small Change

The lightest path. For bug fixes, docs updates, and single-PR changes that don't introduce new module boundaries.

1. **Create worktree** — `git worktree add .worktrees/fix-<name> -b fix-<name>`
2. **Read the issue *and its comment thread*** — `scripts/deliver-issue-refinements.sh <issue-number>`. A body is written once and the design is then refined in comments; a plan made from the body alone builds an out-of-date spec. See [Comments can refine the body](docs/contributing/pr-workflow.md#comments-can-refine-the-body) for which comments carry authority and how they combine with the body
3. **Write plan** — behavioral contracts (GIVEN/WHEN/THEN) and TDD task breakdown
4. **Review plan** — review for correctness before proceeding
5. **Human approval** — review contracts and tasks, approve to proceed
6. **Implement** — execute TDD tasks from the plan
7. **Review code** — review implementation for correctness
8. **Self-audit + commit** — deliberate critical thinking, then commit and push

Full process: [`docs/contributing/pr-workflow.md`](docs/contributing/pr-workflow.md)

### New Policy or Extension

For adding a routing policy, admission policy, scorer, scheduler, priority policy, or tier composition — anything behind an existing interface.

1. **Identify extension type** — see [Adding New Components](#adding-new-components) below
2. **Create worktree** — `git worktree add .worktrees/<extension-name> -b <extension-name>`
3. **Write plan** — follow [`docs/contributing/extension-recipes.md`](docs/contributing/extension-recipes.md) for the recipe
4. **Follow steps 4–8 from Bug Fix** (review → approve → implement → review → commit)

No design doc needed for policy templates. Full process: [`docs/contributing/pr-workflow.md`](docs/contributing/pr-workflow.md)

### New Feature (Idea to PR)

The full pipeline for features that introduce new module boundaries, new interfaces, or span multiple PRs.

**Phase 1 — RFC:**
1. **Write tracking issue** — following [`docs/contributing/rfc.md`](docs/contributing/rfc.md) (plain-English description + holes/surfaces/contracts)
2. **Team discusses** — agree on holes, surfaces, contracts in the issue thread
3. **Human approval** — agreement to proceed

**Phase 2 — Plan** (after agreement):
4. **Encode .archon plan** — following [`docs/contributing/templates/rfc-to-plan.md`](docs/contributing/templates/rfc-to-plan.md) (Claude encodes, creates sub-issues)
5. **PR0** — create `feature/<name>` branch, persist plan to `specs/NNN-feature/`, push

**Phase 3 — Deliver** (repeat for each sub-issue, PRs target the feature branch):
6. **Follow the Bug Fix journey** (steps 1–8) for each hole
7. **Final PR** — merge feature branch → main when dist=0 + tests pass

Each phase produces an artifact that feeds the next. Human approval gates between phases prevent wasted work.

### Without Claude Code

If you are not using Claude Code, here is the simplified workflow:

1. **Branch** — `git checkout -b feature/my-change`
2. **Read the issue and its comment thread** — `scripts/deliver-issue-refinements.sh <issue-number>`. The design may have been corrected in a comment after the body was written; a later comment by someone with write access overrides the body on any point it addresses ([the rule](docs/contributing/pr-workflow.md#comments-can-refine-the-body))
3. **Plan** — write behavioral contracts (GIVEN/WHEN/THEN) and a task breakdown. Post as a PR draft or issue comment for review.
4. **Implement** — follow TDD: write a failing test, implement the minimal code to pass it, run `go test ./...`, run `golangci-lint run ./...`, commit. Repeat for each contract.
5. **Self-review** — check the [Antipattern Checklist](#antipattern-checklist) below. Run `go build ./... && go test ./... && golangci-lint run ./...` one final time.
6. **PR** — push your branch and open a PR. Maintainers will run `@claude /blis-pr-review` and `/archon-pr-review`.

For large features: write the RFC following [`docs/contributing/rfc.md`](docs/contributing/rfc.md) and submit for team discussion. The `.archon` encoding can be done with Claude Code after agreement.

Full process: [`docs/contributing/pr-workflow.md`](docs/contributing/pr-workflow.md) (the same workflow applies regardless of tooling)

## Engineering Principles

See [`docs/contributing/standards/principles.md`](docs/contributing/standards/principles.md) for the full principles guide covering: separation of concerns, interface design, configuration design, canonical constructors, output channel separation, error handling boundaries, and BDD/TDD development.

Key points for new contributors:
- `sim/` is a library — never call `os.Exit` or `logrus.Fatalf`. Return errors. Only `cmd/` may terminate.
- Write behavioral contracts (GIVEN/WHEN/THEN) before tests. Test observable behavior, not internal structure.
- If your PR touches request lifecycle, KV cache, or metrics, add or extend invariant tests (see [`docs/contributing/standards/invariants.md`](docs/contributing/standards/invariants.md)).

## Antipattern Checklist

23 rules, each tracing to a real bug. See [`docs/contributing/standards/rules.md`](docs/contributing/standards/rules.md) for full details.

Before submitting a PR, verify:

- [ ] R1: No silent `continue`/`return` dropping data
- [ ] R2: Map keys sorted before float accumulation or ordered output
- [ ] R3: Every new numeric parameter validated (CLI flags AND library constructors)
- [ ] R4: All struct construction sites audited for new fields
- [ ] R5: Resource allocation loops handle mid-loop failure with rollback
- [ ] R6: No `logrus.Fatalf` or `os.Exit` in `sim/` packages
- [ ] R7: Invariant tests alongside any golden tests
- [ ] R8: No exported mutable maps
- [ ] R9: `*float64` for YAML fields where zero is valid
- [ ] R10: strict config parsing, YAML and JSON (`KnownFields(true)` / no unknown keys)
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
- [ ] R21: No `range` over slices that can shrink during iteration
- [ ] R22: Pre-check estimates consistent with actual operation accounting
- [ ] R23: Parallel code paths apply equivalent transformations

## Adding New Components

BLIS has four extension types. Identify which type your change is, then follow the corresponding recipe. See `docs/contributing/templates/design-guidelines.md` Section 5 for full details.

| Extension Type | What It Is | Design Doc Required? | Example |
|---|---|---|---|
| **Policy Template** | New algorithm behind an existing interface | No | New routing algorithm |
| **Subsystem Module** | New module with its own interface and events | Yes | AutoScaler, P/D disaggregation |
| **Backend Swap** | Alternative implementation of internal module | Yes (covers both phases) | SGLang latency model |
| **Tier Composition** | Wrapper layering behavior on existing module | Recommended | NVMe KV tier |

### Adding a New Model to the Catalog

A model runs **if and only if it is in the catalog** (NS-6, #1733), and the catalog itself must be **located explicitly** — `--catalog <path>` or the `BLIS_CATALOG` environment variable, with no default and no search path (S4, #1731). BLIS reads the model's `config.json` from `<catalog>/models/<short-name>/config.json` and never downloads one at run time, so an uncatalogued model is refused rather than fetched-and-cached. The catalog is the [`blis-catalog`](https://github.com/inference-sim/blis-catalog) repository (#1771 removed the in-repo `model_configs/` tree), so the catalog entry is the load-bearing step:

1. **Commit the model's HuggingFace `config.json` at `models/<short-name>/config.json` in a `blis-catalog` checkout** (`<short-name>` is the part of the model id after the `/`, lowercased). Without it, `blis run` refuses the model. Point BLIS at your checkout with `--catalog <clone-root>` (or `export BLIS_CATALOG=$PWD/blis-catalog`).

**That is the whole procedure — `defaults.yaml` needs no edit at all.** As of #1768 it holds no
per-model entry of any kind: the `defaults:` block (`GPU` / `tensor_parallelism` / `hf_repo`) is
deleted, having been unreachable on every run path since the deployment became a required
`--hardware`/`--tp` input. Two consequences for a contributor:

1. **Do not add a `defaults:` block back.** `defaults.yaml` is decoded into a fixed struct
   (`cmd.Config`) with strict field checking (R10), so an unrecognized top-level key is a parse
   error rather than an inert extra — a file carrying `defaults:` fails to load with
   `field defaults not found in type cmd.Config`.
2. **Provenance goes in the commit, not the YAML.** The retired `hf_repo` field recorded where a
   committed `config.json` came from; say that in the commit message or PR instead (the
   `blis-catalog` repository records it per model in `models/<name>/model.yaml`).

Nothing further is needed for latency either: the trained-physics coefficients
(`trained_physics_coefficients`) are a single global block shared by every model, not a per-model
list.

### Policy Template (lightest — ~3 files)

1. Implement the interface in the corresponding file (`sim/admission.go`, `sim/routing.go`, `sim/priority.go`, `sim/scheduler.go`)
2. Register in `sim/bundle.go` (valid names map + `IsValid*` function)
3. Add `case` to factory function
4. Add behavioral tests (`TestMyPolicy_Scenario_Behavior`)
5. Update README and the relevant `docs/` guide (not CLAUDE.md — it holds pointers, per its charter)

### Subsystem Module (heaviest — new interface + integration)

Requires a design doc defining the module contract (observes / controls / owns / invariants / events / extension friction). See design guidelines Section 5.3.

1. Write design doc with module contract, event integration, state ownership, failure modes, default behavior
2. Create implementation plan with behavioral contracts and TDD tasks
3. Implement interface + default implementation + factory
4. Integrate into cluster event pipeline
5. Add CLI flags with full validation
6. Add behavioral tests + invariant tests
7. Update README, the relevant `docs/` guide, and the design guidelines module map if needed

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


## Code Style

- Composition over inheritance
- Timestamp-based event ordering via min-heap
- Partitioned RNG per subsystem for deterministic isolation
- BDD-style test naming: `TestType_Scenario_Behavior`
- Conventional commits: `feat(scope)`, `fix(scope)`, `refactor(scope)`, `test(scope)`, `docs(scope)`

## Key References

| Document | What It Covers | When to Read |
|---|---|---|
| `CLAUDE.md` | Agent operating context: catalog/deployment rules, canonical commands, and pointers to the standards docs and topic guides (not a per-PR changelog — see its charter) | Always — the starting point for working in this repo |
| `docs/contributing/standards/rules.md` | 23 antipattern rules with evidence, checks, enforcement | When reviewing or writing code |
| `docs/contributing/standards/invariants.md` | The invariant registry, grouped by scope (run-level, subsystem) plus a cross-cutting code-boundary group: 19 core invariants (INV-1 through INV-19, the last six enforcement-anchored) plus INV-A, INV-A2, INV-W3, INV-BC-DP1, the LoRA family (INV-L1–INV-L7, by pointer), PD disaggregation (INV-PD-*) and pool/transfer (INV-P2-*), with verification strategies. Every `INV-*` ID cited in code is resolvable from it. | When touching request lifecycle, KV cache, metrics, or placement |
| `docs/contributing/pr-workflow.md` | End-to-end PR lifecycle (worktree → plan → review → implement → audit → PR) | Before starting any PR |
| `docs/concepts/` | System architecture, core engine, concepts glossary, roofline estimation | When learning how BLIS works before contributing |
| `docs/contributing/templates/design-guidelines.md` | DES foundations, module architecture, extension framework | Before designing a new feature or extending BLIS |
| `docs/contributing/rfc.md` | RFC template for large features (holes/surfaces/contracts) | When planning a multi-PR feature |
| `docs/contributing/templates/rfc-to-plan.md` | Claude prompt for encoding RFC into .archon plan | After RFC agreement, before implementation |
