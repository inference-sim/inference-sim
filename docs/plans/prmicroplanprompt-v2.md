You are operating inside a real repository with full code access.

You are tasked with producing a PR-SPECIFIC IMPLEMENTATION PLAN that combines:
1. Design rigor (behavioral contracts, architecture validation)
2. Executable task breakdown (TDD, bite-sized steps, verifications)

The source of work may be:
- A section in an approved Macro Plan (e.g., "Phase 2, PR 4")
- One or more GitHub issues (e.g., "#183, #189, #195")
- A design document (e.g., "docs/plans/2026-02-18-hardening-design.md")
- A feature request or bug report description

This plan has TWO AUDIENCES:
1) A human reviewer who validates behavioral correctness
2) Automated agents (via executing-plans skill) who execute the tasks

The plan must be comprehensive enough that agents can implement WITHOUT
additional codebase exploration.

======================================================================
DOCUMENT HEADER (REQUIRED)
======================================================================

Every plan MUST start with this exact header format:

```markdown
# [PR Title] Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** [One sentence a non-contributor could understand — what capability does this PR add? Avoid type names, package paths, or implementation jargon.]

**The problem today:** [2-3 sentences explaining what's missing or broken without this PR. What can't users or the system do? Why does it matter?]

**What this PR adds:** [Numbered list of 2-4 concrete capabilities, each explained in plain language with a brief example. E.g., "Decision traces — a log of every routing decision: 'request_42 was sent to instance_2 because it had the highest score of 0.87'"]

**Why this matters:** [1-2 sentences connecting this PR to the broader project vision. How does this enable downstream work?]

**Architecture:** [2-3 sentences about the technical approach — packages, key types, integration points. Implementation jargon is OK here since the motivation is already established above.]

**Source:** [Link to the source of work. Examples:
  - Macro plan: "Phase 2, PR 4 in docs/plans/macro-plan.md"
  - Issues: "GitHub issues #183, #189, #195, #196, #197, #198, #199, #200"
  - Design doc: "docs/plans/2026-02-18-hardening-design.md"
  - Feature request: "GitHub issue #42"]

**Behavioral Contracts:** See Part 1, Section B below

---
```

The header has TWO audiences reading in order:
1. A human reviewer who needs to understand WHY before HOW (Goal → Problem → What → Why)
2. An implementing agent who needs the technical approach (Architecture)

======================================================================
PHASE 0 — COMPONENT CONTEXT
======================================================================

Identify this PR's place in the system architecture:

1) Which building block is being added or modified?
2) What are the adjacent blocks it interacts with?
3) What invariants does this PR touch?
4) What state ownership changes (if any)?
5) Construction Site Audit: For every struct this PR adds fields to,
   grep for ALL places that struct is constructed (struct literals,
   factory functions). List each site with file:line. If there are
   multiple construction sites, the plan MUST either:
   a) Add a canonical constructor and refactor all sites, OR
   b) Update every site explicitly (list each in a task)

Then inspect ONLY the relevant parts of the repository.

List confirmed facts (with file:line citations).
Flag anything from the source document (macro plan, design doc, or
issue description) that doesn't match current code as a DEVIATION —
these must be resolved before implementation begins.

======================================================================
OUTPUT FORMAT (STRICT)
======================================================================

--- PART 1: Design Validation (Human Review, target <120 lines) ---

A) Executive Summary (5-10 lines)
   - What this PR builds (plain language, not type/package names)
   - Where it fits in the system (what comes before it, what depends on it)
   - Adjacent blocks it interacts with
   - Any DEVIATION flags from Phase 0

B) Behavioral Contracts (Phase 1)
   - 3-15 named contracts (BC-1, BC-2, ...)
   - Format: GIVEN / WHEN / THEN / MECHANISM
   - Grouped: positive contracts, negative contracts, error handling

C) Component Interaction (Phase 2)
   - Component diagram (text)
   - API contracts
   - State changes and ownership

D) Deviation Log (Phase 3)
   - Compare micro plan vs source document
   - Table: | Source Says | Micro Does | Reason |

E) Review Guide (Phase 7-B)
   - The tricky part
   - What to scrutinize
   - What's safe to skim
   - Known debt

--- PART 2: Executable Implementation (Agent Execution) ---

F) Implementation Overview (Phase 4 summary)
   - Files to create/modify (one-line each)
   - Key decisions
   - Confirmation: no dead code, all paths exercisable

G) Task Breakdown (Phase 4 detailed)
   - 6-12 tasks in TDD format (see Phase 4 template below)
   - Continuous execution (no pause points between tasks)
   - Each task: test → fail → implement → pass → lint → commit

H) Test Strategy (Phase 6)
   - Map contracts to tasks/tests
   - Golden dataset update strategy
   - Shared test infrastructure usage

I) Risk Analysis (Phase 7-A)
   - Risks with likelihood/impact/mitigation

--- PART 3: Quality Assurance ---

J) Sanity Checklist (Phase 8)
   - Pre-implementation verification
   - All items from Phase 8 template

--- APPENDIX: File-Level Implementation Details ---

K) Detailed specifications
   - Complete function signatures with doc comments
   - Struct definitions
   - Event execution logic
   - Metric aggregation rules
   - RNG subsystem usage
   - Any behavioral subtleties (file:line citations)

======================================================================
PHASE 1 — BEHAVIORAL CONTRACTS (Human-Reviewable)
======================================================================

This defines what this PR guarantees. Use named contracts (BC-1, BC-2, ...)
that can be referenced in tests, reviews, and future PRs.

For each contract:

  BC-N: <Name>
  - GIVEN <precondition>
  - WHEN <action>
  - THEN <observable outcome>
  - MECHANISM: <one sentence explaining how> (optional but recommended)

Group contracts into:

1) Behavioral Contracts (what MUST happen)
   - Normal operation
   - Edge cases
   - Backward compatibility

2) Negative Contracts (what MUST NOT happen)
   - Invariant violations this PR could cause
   - Cross-boundary state leaks
   - Performance regressions

3) Error Handling Contracts
   - What happens on invalid input
   - What happens on resource exhaustion
   - Panic vs error return vs log-and-continue (be explicit)

TARGET: 3-15 contracts per PR. Pure refactoring PRs with no new behavior
may have as few as 3. More than 15 means the PR may be too large.

No vague wording. "Should" is banned — use "MUST" or "MUST NOT."

THEN CLAUSE QUALITY GATE:
Every THEN clause must describe OBSERVABLE BEHAVIOR, not internal
structure. The THEN clause directly becomes the test assertion — a
structural THEN produces a structural test.

Check each THEN clause against this filter:
- Does it contain a concrete type name? → Rewrite to describe behavior
  BAD:  "THEN it returns a *ConstantPriority"
  GOOD: "THEN it returns a policy that computes 0.0 for any request"
- Does it reference an internal field? → Rewrite to describe output
  BAD:  "THEN the router's scoreCache has 3 entries"
  GOOD: "THEN the next routing decision uses cached affinity"
- Does it reproduce a formula? → Rewrite to describe ordering/outcome
  BAD:  "THEN score equals 0.6*cacheHit + 0.4*(1-load)"
  GOOD: "THEN instances with higher cache hit rates rank higher"
- Does it survive a refactor? → If renaming a struct or changing an
  internal algorithm would invalidate this THEN, it is structural

======================================================================
PHASE 2 — COMPONENT INTERACTION (Human-Reviewable)
======================================================================

Describe this PR's building block and how it connects to the system.
This is the "box-and-arrow" view, NOT the file-level view.

1) Component Diagram (text-based)
   - This PR's component and its responsibility
   - Adjacent components (existing or new)
   - Data flow direction between them
   - What crosses each boundary (types, not implementations)

2) API Contracts
   - New interfaces or types (signature + one-line semantics)
   - Method preconditions and postconditions
   - Failure modes and how callers handle them

3) State Changes
   - New mutable state and its owner
   - State lifecycle (created when, destroyed when, accessed by whom)

4) Extension Friction Assessment
   - For the main new type/field this PR adds, count: how many files
     must change to add ONE more field of the same kind?
   - If >3 files, document whether this is acceptable or whether a
     structural improvement should happen first/concurrently
   - This is not a blocker — it's awareness for the reviewer

TARGET: under 40 lines. Infrastructure PRs that introduce multiple
interacting types may go up to 60 lines with justification.
Beyond 60 lines, the PR scope is likely too broad.

======================================================================
PHASE 3 — DEVIATION LOG
======================================================================

Compare this micro plan against the source document (macro plan section,
design doc, or issue description).

For each difference:

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|

Categories of deviation:
- SIMPLIFICATION: Source specified more than needed at this stage
- CORRECTION: Source was wrong about existing code or behavior
- DEFERRAL: Feature moved to a later PR (explain why)
- ADDITION: Something the source missed
- SCOPE_CHANGE: Issue description expanded or narrowed during investigation

If there are zero deviations, state "No deviations from source document."

======================================================================
PHASE 4 — EXECUTABLE TASK BREAKDOWN
======================================================================

Break implementation into 6-12 tasks following TDD principles.
Each task is completable in one focused session (~30-45 minutes).

**Execution is continuous** — all tasks run sequentially without pausing
for human input. Execution only stops on test failure, lint failure, or
build error. Group tasks into logical sections (e.g., core types,
integration, edge cases) for readability, but these are NOT pause points.

TASK TEMPLATE:

### Task N: [Component/Feature Name]

**Contracts Implemented:** BC-X, BC-Y (reference Phase 1)

**Files:**
- Create: `exact/path/to/file.go`
- Modify: `exact/path/to/existing.go:123-145` (line range if known)
- Test: `exact/path/to/test_file.go`

**Step 1: Write failing test for [specific contract]**

Context: [1-2 sentences explaining what we're testing and why]

```go
// Complete test code here
// Include setup, execution, assertions
func TestComponent_Scenario_Behavior(t *testing.T) {
    // GIVEN [precondition from contract]

    // WHEN [action from contract]

    // THEN [expected outcome from contract]
    assert.Equal(t, expected, actual)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./path/to/package/... -run TestComponent_Scenario -v`
Expected: FAIL with "[expected error message]"

**Step 3: Implement minimal code to satisfy contract**

Context: [1-2 sentences about the implementation approach]

In `path/to/file.go`:
```go
// Complete implementation code
// Include type definitions, method signatures, logic
type Component struct {
    field1 Type1
    field2 Type2
}

func (c *Component) Method(param Type) (ReturnType, error) {
    // implementation
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./path/to/package/... -run TestComponent_Scenario -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./path/to/package/...`
Expected: No new issues (pre-existing issues OK, don't fix them)

**Step 6: Commit with contract reference**

```bash
git add path/to/file.go path/to/test_file.go
git commit -m "feat(package): implement Component.Method (BC-X, BC-Y)

- Add Component type with Method
- Implement contract BC-X: [brief description]
- Implement contract BC-Y: [brief description]

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

REPEAT TASK TEMPLATE for each task (6-12 total).

IMPORTANT TASK DESIGN RULES:

1. **Each task implements 1-3 related contracts** - don't split single
   contracts across tasks, don't pack unrelated contracts together

2. **Complete code in every step** - no "add validation" or "implement logic"
   without showing the exact code

3. **Exact commands with expected output** - agent should know if verification
   succeeded or failed

4. **Reference shared test infrastructure** - use existing helpers from
   shared packages (e.g., sim/internal/testutil), don't duplicate

5. **Golden dataset updates** - if task changes output format or metrics,
   include step to update testdata/goldendataset.json with regeneration command

6. **Dependency ordering** - tasks must be ordered so each can build on
   previous completed work

7. **No dead code** - every struct field, every method, every parameter must
   be used by the end of the task or a subsequent task in this PR

8. **Commit messages** - use conventional commits format with contract references
   (feat/fix/refactor/test/docs)

9. **Behavioral assertions only** - every assertion in a test must
   verify OBSERVABLE BEHAVIOR, not internal structure. Apply the
   refactor survival test: "Would this test still pass if the
   implementation were completely rewritten but the behavior preserved?"

   PROHIBITED assertion patterns (structural — these break on refactor):
   - Type assertions: `policy.(*ConcreteType)` — test behavior instead
   - Internal field access: `obj.internalField` — test through public API
   - Exact formula reproduction: `assert.Equal(score, 0.6*cache + 0.4*load)`
     — test the ranking/ordering outcome instead
   - Implementation count: `assert.Equal(len(obj.items), 3)` — test
     what the items produce, not how many there are

   REQUIRED assertion patterns (behavioral — these survive refactor):
   - Observable output: `assert.Equal(policy.Compute(req, clock), 0.0)`
   - Behavioral outcome: `assert.Equal(decision.TargetInstance, 1)`
   - Invariant verification: `assert.Equal(completed+queued+running, injected)`
   - Ordering/ranking: `assert.True(scoreA > scoreB)` when contract says
     A should rank higher than B

10. **THEN clauses must be behavioral** - if a behavioral contract's
    THEN clause contains a concrete type name, internal field name, or
    implementation detail, rewrite the THEN clause BEFORE writing the
    test. The THEN clause drives the assertion; a structural THEN
    produces a structural test.

    BAD:  "THEN it returns a *ConstantPriority"
    GOOD: "THEN it returns a policy that computes 0.0 for any request"

    BAD:  "THEN the router's scoreCache has 3 entries"
    GOOD: "THEN the next routing decision uses cached scores (latency < uncached)"

    BAD:  "THEN the score equals 0.6*cacheHit + 0.4*(1-load)"
    GOOD: "THEN instances with higher cache hit rates score higher than
           instances with lower cache hit rates, all else being equal"

======================================================================
PHASE 5 — REMOVED (Merged into Phase 4 Task Verification)
======================================================================

Exercisability is proven by the task-level verification steps.
No separate section needed.

======================================================================
PHASE 6 — TEST STRATEGY
======================================================================

Map contracts to tasks and tests:

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|--------------------------|
| BC-1     | Task 1 | Unit    | TestFoo_GivenX_ThenY     |
| BC-2     | Task 1 | Unit    | TestFoo_GivenZ_ThenW     |
| BC-3     | Task 2 | Golden  | TestCluster_SingleInstance_MatchesGolden |
| ...      | ...    | ...     | ...                      |

Test types:
- Unit: specific function/method behavior
- Integration: cross-component or CLI-level
- Golden: regression against known-good output (testdata/goldendataset.json)
- Invariant: system law that must hold regardless of output values (see req 6)
- Failure: error paths, panics, edge cases
- Benchmark: performance-sensitive paths (optional)

Additional requirements:

1. **Shared test infrastructure**: Use existing helpers from shared test
   packages (e.g., sim/internal/testutil). If new helpers are needed, add
   them to the shared package in an early task — not duplicated locally.

2. **Golden dataset updates**: If this PR changes output format or adds new
   metrics, document:
   - Which task updates the golden dataset
   - Exact regeneration command
   - How to verify the update is correct (compare key metrics)

3. **Lint requirements**: `golangci-lint run ./...` must pass with zero new
   issues. Pre-existing issues are acceptable; do not fix unrelated lint
   warnings (scope creep).

4. **Test naming convention**: Use BDD-style names that describe the scenario:
   `TestType_Scenario_Behavior` (e.g., `TestTokenBucket_CapacityExceeded_RejectsRequest`)

5. **Test isolation**: Each test must be independently runnable (no order
   dependencies). Use table-driven tests for multiple scenarios of the same behavior.

6. **Invariant tests alongside golden tests (MANDATORY)**: Golden dataset
   tests are characterization tests — they capture *what the code does*, not
   *what the code should do*. If the code has a bug when the golden dataset
   is generated, the test encodes the bug as the expected value. This has
   happened: issue #183 found that the codellama golden dataset expected 499
   completions because one request was silently dropped — a bug that the
   golden test perpetuated instead of catching.

   **Rule:** Every golden dataset test MUST be paired with at least one
   invariant test that verifies a system law derived from the *specification*
   (not from running the code). Invariant tests answer "is the code correct?"
   while golden tests answer "did the code change?"

   Key invariants for this simulator (derived from CLAUDE.md):
   - **Request conservation:** completed + still_queued + still_running = injected
   - **KV block conservation:** allocated_blocks + free_blocks = total_blocks
   - **Clock monotonicity:** simulation clock never decreases
   - **Causality:** arrival_time ≤ enqueue_time ≤ schedule_time ≤ completion_time
   - **Determinism:** same seed produces byte-identical output across runs

   When adding a golden test, ask: "If this golden value were wrong, would
   any other test catch it?" If the answer is no, add an invariant test.
   If this PR touches request lifecycle, KV cache, or metrics, at least one
   invariant test MUST be added or extended.

======================================================================
PHASE 7 — RISK ANALYSIS & REVIEW GUIDE
======================================================================

PART A: Risks

For each risk:
- Risk description
- Likelihood (low/medium/high)
- Impact (low/medium/high)
- Mitigation (specific test or design choice)
- Which task mitigates the risk

PART B: Review Guide (for the human reviewer)

In 5-10 lines, tell the reviewer:

1) THE TRICKY PART: What's the most subtle or error-prone aspect?
2) WHAT TO SCRUTINIZE: Which contract(s) are hardest to verify?
3) WHAT'S SAFE TO SKIM: Which parts are mechanical/boilerplate?
4) KNOWN DEBT: Any pre-existing issues encountered but not fixed?

This section exists because human attention is scarce.
Direct it to where it matters most.

======================================================================
PHASE 8 — DESIGN SANITY CHECKLIST
======================================================================

Before implementation, verify:

- [ ] No unnecessary abstractions.
- [ ] No feature creep beyond PR scope.
- [ ] No unexercised flags or interfaces.
- [ ] No partial implementations.
- [ ] No breaking changes without explicit contract updates.
- [ ] No hidden global state impact.
- [ ] All new code will pass golangci-lint.
- [ ] Shared test helpers used from existing shared test package (not duplicated locally).
- [ ] CLAUDE.md updated if: new files/packages added, file organization
      changed, plan milestone completed, new CLI flags added.
- [ ] No stale references left in CLAUDE.md.
- [ ] Deviation log reviewed — no unresolved deviations.
- [ ] Each task produces working, testable code (no scaffolding).
- [ ] Task dependencies are correctly ordered.
- [ ] All contracts are mapped to specific tasks.
- [ ] Golden dataset regeneration documented (if needed).
- [ ] Construction site audit completed (Phase 0, item 5) — all struct
      construction sites listed and covered by tasks.
- [ ] Every new CLI flag validated for: zero, negative, NaN, Inf, empty string.
- [ ] Every error path either returns error, panics with context, or increments
      a counter — no silent `continue` that drops data.
- [ ] No map iteration feeds float accumulation or determines output ordering
      without sorted keys.
- [ ] Library code (sim/, sim/cluster/, sim/workload/) never calls logrus.Fatalf
      or os.Exit — errors must be returned to callers.
- [ ] Any loop that allocates resources (blocks, slots, counters) handles mid-loop
      failure by rolling back all mutations from previous iterations.
- [ ] No exported mutable maps — validation lookup maps are unexported with
      IsValid*() accessors.
- [ ] YAML config structs use *float64 (pointer) for fields where zero is a valid
      user-provided value.
- [ ] YAML loading uses strict parsing (yaml.KnownFields(true) or equivalent) —
      typos in field names cause errors, not silent acceptance.
- [ ] Every division operation where the denominator derives from runtime state
      has a zero-denominator guard or a documented invariant proving it non-zero.
- [ ] New interfaces accommodate at least two implementations (even if only one
      exists today) — no methods that only make sense for one backend.
- [ ] No method spans multiple module responsibilities (scheduling + latency +
      metrics in one function). Extract each concern into its module's interface.
- [ ] Configuration parameters grouped by module — not added to a monolithic
      config struct mixing unrelated concerns.
- [ ] Grepped for references to this PR number (planned for PR, TODO.*PR) in the
      codebase — resolved all stale references.
- [ ] If this PR is part of a macro plan, the macro plan status is updated.

======================================================================
APPENDIX — FILE-LEVEL IMPLEMENTATION DETAILS
======================================================================

This section has NO LENGTH LIMIT. It should contain everything needed
to implement the PR without further codebase exploration.

For each file to be created or modified, provide:

**File: `exact/path/to/file.go`**

**Purpose:** [1-2 sentences]

**Complete Implementation:**

```go
// Package documentation

package name

import (
    // all imports
)

// Complete type definitions with doc comments
// Complete function implementations
// Complete test code
// Include all struct fields, all methods, all parameters

// Behavioral notes:
// - [Any subtlety, e.g., "horizon boundary: requests at exactly
//    horizon time are NOT completed"]
// - [Citation: existing behavior to preserve, with file:line]
```

**Key Implementation Notes:**
- RNG usage: [Which subsystem from PartitionedRNG? e.g., "SubsystemRouter"]
- Metrics: [What metrics are collected? Where aggregated?]
- Event ordering: [Priority? Timestamp? Secondary tie-breaking?]
- State mutation: [What gets modified? Who owns it?]
- Error handling: [Panic, return error, log-and-continue?]

---

Include this level of detail for EVERY file touched by this PR.

======================================================================
EXECUTION HANDOFF
======================================================================

After creating the plan, the workflow continues with:

**Option 1: Subagent-Driven Development (in current session)**
- Invoke superpowers:subagent-driven-development
- Fresh subagent per task
- Code review between tasks
- Fast iteration

**Option 2: Worktree with executing-plans (recommended for complex PRs)**
- Create isolated worktree (superpowers:using-git-worktrees)
- Continue in same session (.worktrees/) or open new session (sibling directory)
- Invoke superpowers:executing-plans with this plan
- Continuous execution (stops only on failure)
- Invoke commit-commands:commit-push-pr when complete

======================================================================
QUALITY BAR
======================================================================

This plan must:
- Survive expert review (behavioral contracts are sound)
- Survive systems-level scrutiny (architecture is correct)
- Eliminate dead code (all code exercisable immediately)
- Reduce implementation bugs (TDD, explicit verifications)
- Stay strictly within source document scope (deviations justified)
- Pass golangci-lint with zero new issues
- Enable automated execution (complete code, exact commands)
- Map every contract to a task (traceability)

======================================================================
LINTING REQUIREMENTS
======================================================================

This project uses golangci-lint for static analysis.
Version is pinned in CI (see .github/workflows/ci.yml).

Local verification (run before submitting PR):
```bash
golangci-lint run ./...
```

Rules:
1. All NEW code must pass lint with zero issues.
2. Do not fix pre-existing lint issues in unrelated code (scope creep).
3. If a lint rule seems wrong, document why and discuss before disabling.

======================================================================

Think carefully.
Inspect deeply.
Design defensively.
Break into executable tasks.
Verify every step.
Direct the reviewer's attention wisely.
