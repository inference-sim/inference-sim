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

This walkthrough adds a test, the lightest kind of contribution, and shows you where a request's value reaches the Result document. Follow it step by step to learn the patterns, then apply them to your own contribution.

**What we'll build:** a test that every row of the Result's `request_rows` carries the `slo_class` its workload client declared. We use test-driven development: we write the test, break the code it guards to watch it fail, and restore the code.

### Step 1: Create a branch

```bash
git checkout -b feature/row-slo-class-test
```

### Step 2: Find where the value is recorded

Each row member is filled from one of three records, read together in `requestRows` (`cmd/result_rows.go`): the fate ledger (`sim/fate.go`: fate and drop reason), the facts the cluster notes when a request is created (`RequestFact`, set by `noteFacts` in `sim/cluster/fates.go`: arrival, prompt length, tags), and the engine's per-request metrics (`RequestMetrics` in `sim/metrics_utils.go`: latencies, token counts, cache hits). A tag the workload sets on a request, such as its SLO class, reaches the row through `RequestFact`.

### Step 3: Write the test

Add to `cmd/result_output_test.go`. It runs a workload spec with two clients, one per class, and checks each row against the class of its client:

```go
func TestResult_RowSLOClassIsItsClientsClass(t *testing.T) {
	// GIVEN a workload spec whose clients declare slo_class "critical" and "batch"
	spec := filepath.Join(t.TempDir(), "spec.yaml")
	body := `version: "2"
seed: 5
category: language
aggregate_rate: 10
num_requests: 20
clients:
  - {id: a, slo_class: critical, rate_fraction: 0.5, arrival: {process: poisson},
     input_distribution: {type: constant, params: {value: 64}},
     output_distribution: {type: constant, params: {value: 8}}}
  - {id: b, slo_class: batch, rate_fraction: 0.5, arrival: {process: poisson},
     input_distribution: {type: constant, params: {value: 64}},
     output_distribution: {type: constant, params: {value: 8}}}
`
	if err := os.WriteFile(spec, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	// WHEN it runs
	out, stderr, err := runKernelCLI(t, "run", "--scenario", kernelTestScenario, "--workload-spec", spec)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	// THEN both classes appear on rows, and every row has one
	seen := map[string]int{}
	for _, row := range parseResult(t, out).RequestRows {
		seen[row.SLOClass]++
	}
	if seen["critical"] == 0 || seen["batch"] == 0 || len(seen) != 2 {
		t.Errorf("rows by slo_class: %v, want both classes and no other", seen)
	}
}
```

Run: `go test ./cmd -run TestResult_RowSLOClass -v`. It passes, so prove it can fail: comment out the `SLOClass: req.SLOClass` line in `noteFacts` (`sim/cluster/fates.go`), run it again and watch it fail, then restore the line. A test you have not seen fail may not test anything.

### Step 4: Add a member of your own

To state a new value on every row, follow the same path: set the field on the request, copy it into `RequestFact` in `noteFacts`, state it in `requestRows`, and add the member to `RequestRow` in blis-substrate `contracts/spec/result`. The last is a substrate change, released and repinned in `go.mod` ([cross-repo changes](docs/contributing/cross-repo-changes.md)).

### Step 5: Verify

```bash
go test ./cmd -run TestResult_RowSLOClass -v   # Your new test
go test ./...                                    # All tests still pass
golangci-lint run ./...                          # No lint issues
```

### Step 6: Commit and open a PR

```bash
git add cmd/result_output_test.go
git commit -m "test(cmd): a Result row's slo_class is its client's class"
git push -u origin feature/row-slo-class-test
gh pr create --title "test: a Result row's slo_class is its client's class" --body "My first BLIS contribution!"
```

**That's it!** Real contributions follow the same pattern, with more contracts and a formal implementation plan.

> **Important:** This example is for learning only; the test it adds may already exist by the time you read this. For your actual first contribution, check [open issues](https://github.com/inference-sim/inference-sim/issues) for tasks labeled `good first issue`.

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
5. **PR0** — create `feature/<name>` branch, persist plan to a new `specs/<NNN>-<feature>/` directory, push

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
| **Backend Swap** | Alternative implementation of internal module | Yes (covers both phases) | Alternative KV store |
| **Tier Composition** | Wrapper layering behavior on existing module | Recommended | NVMe KV tier |

### Adding a New Model to the Catalog

A model runs **if and only if it is in the catalog** (NS-6, #1733), and the catalog itself must be **located explicitly** — `--catalog <path>` or the `BLIS_CATALOG` environment variable, with no default and no search path (S4, #1731). BLIS reads the model's `config.json` from `<catalog>/models/<short-name>/config.json` and never downloads one at run time, so an uncatalogued model is refused rather than fetched-and-cached. The catalog is the [`blis-catalog`](https://github.com/inference-sim/blis-catalog) repository (#1771 removed the in-repo `model_configs/` tree), so the catalog entry is the load-bearing step:

1. **Commit the model's HuggingFace `config.json` at `models/<short-name>/config.json` in a `blis-catalog` checkout** (`<short-name>` is the part of the model id after the `/`, lowercased). Without it, `blis run` refuses the model. Point BLIS at your checkout with `--catalog <clone-root>` (or `export BLIS_CATALOG=$PWD/blis-catalog`).

**That is the whole procedure — `defaults.yaml` needs no edit at all.** As of #1768 it holds no
per-model entry of any kind: the `defaults:` block (`GPU` / `tensor_parallelism` / `hf_repo`) is
deleted; the deployment is stated by the run's scenario. Two consequences for a contributor:

1. **Do not add a `defaults:` block back.** `defaults.yaml` is decoded into a fixed struct
   (`cmd.Config`) with strict field checking (R10), so an unrecognized top-level key is a parse
   error rather than an inert extra — a file carrying `defaults:` fails to load with
   `field defaults not found in type cmd.Config`.
2. **Provenance goes in the commit, not the YAML.** The retired `hf_repo` field recorded where a
   committed `config.json` came from; say that in the commit message or PR instead (the
   `blis-catalog` repository records it per model in `models/<name>/model.yaml`).

`defaults.yaml` carries only the `lora:` block and `version`. Latency needs nothing here either:
the model is priced by `blis-latency-kernel` from its catalog graph and the `blis-registry`
coefficient sets a scenario names, so a new model also needs its `graph.yaml` in the catalog and
a scenario that deploys it.

### Policy Template (lightest — ~3 files)

1. Implement the interface in the repository that owns the policy (see [extension recipes](docs/contributing/extension-recipes.md)): admission and routing in llm-d-router, the instance scheduler and its priority in blis-vllm (`core/sched`)
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
| `docs/contributing/standards/invariants.md` | The invariant registry, grouped by scope (run-level, subsystem) plus a cross-cutting code-boundary group: 19 core invariants (INV-1 through INV-19, the last six enforcement-anchored) plus INV-A, INV-A2, INV-W3, the LoRA family (by pointer), PD disaggregation (INV-PD-*) and pool/transfer (INV-P2-*), with verification strategies. Every `INV-*` ID cited in code is resolvable from it. | When touching request lifecycle, KV cache, metrics, or placement |
| `docs/contributing/pr-workflow.md` | End-to-end PR lifecycle (worktree → plan → review → implement → audit → PR) | Before starting any PR |
| `docs/concepts/` | System architecture, core engine, concepts glossary | When learning how BLIS works before contributing |
| `docs/contributing/templates/design-guidelines.md` | DES foundations, module architecture, extension framework | Before designing a new feature or extending BLIS |
| `docs/contributing/rfc.md` | RFC template for large features (holes/surfaces/contracts) | When planning a multi-PR feature |
| `docs/contributing/templates/rfc-to-plan.md` | Claude prompt for encoding RFC into .archon plan | After RFC agreement, before implementation |
