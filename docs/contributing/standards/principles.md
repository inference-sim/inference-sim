# BLIS Engineering Principles

Principles guide design decisions. The [antipattern rules](rules.md) are specific, checkable manifestations of these principles. The [invariants](invariants.md) are properties that must always hold.

## Separation of Concerns

- `sim/` is a library — never call `os.Exit`, `logrus.Fatalf`, or terminate the process. Return errors. Only `cmd/` may terminate. *(Enforced by R6)*
- Cluster-level policies (admission, routing) receive `*RouterState` with global view. Instance-level policies (priority, scheduler) receive only local data. Never leak cluster state to instance-level code.
- Bridge types (`RouterState`, `RoutingSnapshot`) live in `sim/` to avoid import cycles.
- Unidirectional dependency: `cmd/ -> sim/cluster/ -> sim/` and `sim/cluster/ -> sim/trace/`. `sim/` never imports subpackages.

## Interface Design

- Single-method interfaces where possible (`AdmissionPolicy`, `RoutingPolicy`, `PriorityPolicy`, `InstanceScheduler`).
- Query methods must be pure — no side effects, no state mutation, no destructive reads. Separate `Get()` and `Consume()` for query-and-clear.
- Factory functions must validate inputs: `IsValid*()` check + switch/case + panic on unknown.
- Interfaces defined by behavioral contract, not one implementation's data model. *(Enforced by R13)*
- Methods operate within a single module's responsibility. *(Enforced by R14)*

## Configuration Design

- Group configuration by module. *(Enforced by R16)*
- Each module's config independently specifiable and validatable.

## Canonical Constructors

- Every struct constructed in multiple places needs a canonical constructor. Struct literals appear in exactly one place. *(Enforced by R4)*
- Before adding a field, grep for ALL construction sites.

## Output Channel Separation

- **stdout** (deterministic): simulation results — metrics JSON, fitness scores, anomaly counters, KV cache metrics, per-SLO metrics, trace summaries. Use `fmt.Println`/`fmt.Printf`.
- **stderr** (diagnostic): configuration echoes, progress markers, warnings, errors. Use `logrus.*`, controlled by `--log`.
- Rule of thumb: if a user piping to a file would want to capture it, use `fmt`. If it's debugging context, use `logrus`.

## Error Handling Boundaries

| Layer | Strategy | Example |
|-------|----------|---------|
| CLI (`cmd/`) | `logrus.Fatalf` for user errors | Invalid `--rate` value |
| Library (`sim/`) | `panic()` for invariant violations | Unknown policy name in factory |
| Library (`sim/`) | `error` return for recoverable failures | File I/O, parse errors |
| Runtime (`sim/`) | `bool` return for expected conditions | KV allocation failure -> preemption |

Never use `continue` in an error path without propagating, counting, or documenting why it's safe. *(Enforced by R1)*

## BDD/TDD Development

1. Write behavioral contracts first (GIVEN/WHEN/THEN)
2. Implement tests before code
3. Use table-driven tests
4. Test laws, not just values — invariant tests alongside golden tests *(Enforced by R7)*
5. Refactor survival test: "Would this test still pass if the implementation were completely rewritten but the behavior preserved?"
6. THEN clauses drive test quality — structural THEN produces structural test

**Prohibited assertion patterns** (structural — break on refactor):
- Type assertions: `policy.(*ConcreteType)`
- Internal field access: `obj.internalField`
- Exact formula reproduction: `assert.Equal(score, 0.6*cache + 0.4*load)`

**Required assertion patterns** (behavioral — survive refactor):
- Observable output: `assert.Equal(policy.Compute(req, clock), 0.0)`
- Invariant verification: `assert.Equal(completed+queued+running+dropped, injected)`
- Ordering/ranking: `assert.True(scoreA > scoreB)`

## Test Suite Performance

As the test suite grows (invariant tests, golden tests, hypothesis-promoted regression tests), keep total `go test ./...` time manageable:

- **Individual test budget:** No single test should exceed 5 seconds without using `testing.Short()` to provide a fast-path skip. Tests that run full cluster simulations (e.g., 10K requests across 8 instances) should check `testing.Short()` and reduce to a minimal configuration.
- **CI target:** Total `go test ./...` should complete in under 60 seconds. If it exceeds this, audit for tests that can use smaller configurations without losing behavioral coverage.
- **Benchmark isolation:** Performance benchmarks (`Benchmark*` functions) run only with `go test -bench=.`, never in the default `go test ./...` path. This is Go's default behavior — just don't put benchmark assertions in regular tests.

## Documentation Single Source of Truth

Every piece of documentation lives in exactly one canonical location. Other files may contain **working copies** (summaries for quick reference) with explicit canonical-source headers.

**The canonical-source pattern:**

> **Canonical source:** [`docs/contributing/standards/rules.md`](rules.md). If this section diverges, rules.md is authoritative.

**When updating any standard, invariant, rule, or recipe:**
1. Update the canonical source FIRST
2. Then update any working copies that reference it
3. If you can't update working copies immediately, the canonical-source header ensures readers know which version to trust

**Single-source-of-truth map:**

| Content | Canonical Source | Working Copies |
|---------|-----------------|----------------|
| Antipattern rules (R1-R20) | `docs/contributing/standards/rules.md` | CLAUDE.md (table), CONTRIBUTING.md (checklist) |
| System invariants (INV-1–INV-8) | `docs/contributing/standards/invariants.md` | CLAUDE.md (summary), `docs/concepts/core-engine.md` (formulas), `docs/concepts/architecture.md` (signal freshness) |
| Engineering principles | `docs/contributing/standards/principles.md` | CLAUDE.md (summary) |
| Extension recipes (policies, scorers, KV tiers) | `docs/contributing/extension-recipes.md` | — |
| Design process | `docs/contributing/design-process.md` | CONTRIBUTING.md (summary) |
| Macro-plan process | `docs/contributing/macro-planning.md` | CONTRIBUTING.md (summary) |
| File organization and architecture | CLAUDE.md (File Organization tree) | README.md (Project Structure tree) |
| Hypothesis catalog and specifications | `docs/plans/research.md` | — |
| Experiment status and coverage | `hypotheses/README.md` | — |
| Experiment standards | `docs/contributing/standards/experiments.md` | — (note: review protocol subsection references `docs/contributing/convergence.md` as canonical) |
| Convergence protocol | `docs/contributing/convergence.md` | `docs/contributing/hypothesis.md` (summary), `docs/contributing/pr-workflow.md` (summary), `docs/contributing/standards/experiments.md` (review protocol summary), `.claude/skills/convergence-review/SKILL.md` (protocol copy) |
| Hypothesis experiment workflow | `docs/contributing/hypothesis.md` | CONTRIBUTING.md (summary), hypotheses/README.md (step list), `.claude/skills/hypothesis-experiment/SKILL.md` (workflow steps), `.claude/skills/hypothesis-experiment/review-prompts.md` (perspective prompts) |
| PR workflow | `docs/contributing/pr-workflow.md` | CONTRIBUTING.md (summary), `.claude/skills/convergence-review/pr-prompts.md` (perspective prompts) |
