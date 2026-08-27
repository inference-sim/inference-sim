# Review Perspectives

Targeted review lenses used across the SDLC. Each perspective catches issues the others miss. Referenced by `rfc.md` (design), `pr-workflow.md` (plan + code), and `blis-pr-review` (CI).

---

## Design Review Perspectives (8)

Used when reviewing an RFC before team agreement. See [rfc.md](rfc.md).

1. **Motivation & Scoping** — Are goals clear? Is the modeling decisions table complete? Does every simplification state what real-system behavior is lost?
2. **Module Contract Completeness** — Does every hole have all contract fields? Are invariants named (INV-N) and cross-referenced?
3. **Extension Framework Fit** — Is the extension type correct? Is the no-op default specified? Is parallel development possible?
4. **Trade-off Quality** — Does every non-obvious decision have alternatives with rationale? What breaks if it's wrong?
5. **Validation Strategy** — How will correctness be verified (which invariants)? How will fidelity be validated (against what)?
6. **Staleness Resistance** — Is content described behaviorally (what crosses a boundary and why) not structurally (how)?
7. **Domain Expertise** — DES: are new events classified (exogenous/endogenous)? vLLM: does this match real serving behavior? Platform: scaling assumptions valid?
8. **Prohibited Content** — No Go struct definitions, no method implementations, no file:line references. Describe behavior, not implementation.

---

## Plan Review Perspectives (10)

Used when reviewing an implementation plan before coding. See [pr-workflow.md](pr-workflow.md) Step 2.5.

1. **Substance & Design** — design bugs, mathematical errors, logical inconsistencies, scale mismatches, missing edge cases. Are the behavioral contracts logically sound? Could the design actually achieve what the contracts promise?
2. **Cross-Document Consistency** — scope mismatch between plan and source issue, stale file paths, deviation log completeness.
3. **Architecture Boundary Verification** — import cycle risks, boundary violations, types in wrong packages, high touch-point multipliers, library code calling `logrus.Fatalf`.
4. **Codebase Readiness** — stale comments, pre-existing bugs in files the plan will modify, missing dependencies, unclear insertion points.
5. **Plan Structural Validation** — task dependencies (can each task start given what comes before?), behavioral contracts present, executive summary clarity, under-specified tasks.
6. **DES Expert** — event ordering bugs, clock monotonicity violations, stale signal propagation, heap priority errors, work-conserving property violations.
7. **vLLM/SGLang Expert** — batching semantics mismatch, KV cache eviction differences, chunked prefill errors, preemption policy differences.
8. **Distributed Inference Platform Expert** — multi-instance coordination bugs, routing load imbalance, stale snapshot propagation, admission control edge cases.
9. **Performance & Scalability** — O(n²) where O(n) suffices, hot-path allocations, map iteration in loops, memory growth at scale.
10. **Security & Robustness** — input validation completeness, panic paths from user input, resource exhaustion vectors, degenerate input handling.

---

## Code Review Perspectives (10)

Used when reviewing implementation before commit. See [pr-workflow.md](pr-workflow.md) Step 4.5 and `.claude/skills/blis-pr-review/`.

1. **Substance & Design** — logic bugs, design mismatches between contracts and implementation, mathematical errors, silent regressions. Does the implementation actually achieve what the behavioral contracts promise?
2. **Code Quality + Error Handling** — error path cleanup, map iteration sorted (R2/INV-6), construction site drift (R4), library code calling logrus.Fatalf (R6), exported mutable maps (R8), YAML pointer types (R9), division zero guards (R19), CLAUDE.md/docs drift.
3. **Test Behavioral Quality** — are tests behavioral (test WHAT not HOW)? Would they survive a refactor? Golden tests without companion invariant tests? Tests that pass even if the feature is broken? Contracts claiming "for any input" / "never" / "always" need fuzz targets or property-based tests — a table of N cases proves N cases, not universality.
4. **Getting-Started Experience** — would a new user or contributor get stuck? Missing examples, undocumented output, incomplete guides, unclear extension points?
5. **Automated Reviewer Simulation** — what Copilot/Claude/Codex would flag: exported mutable globals, user-controlled panic paths, YAML typo acceptance, NaN/Inf gaps, redundant code.
6. **DES Expert** — event ordering bugs, clock monotonicity (INV-3), stale signal propagation, heap priority errors, work-conserving violations (INV-8).
7. **vLLM/SGLang Expert** — batching semantics mismatch, KV cache eviction differences, chunked prefill errors, preemption policy differences, scheduling assumption violations.
8. **Distributed Inference Platform Expert** — multi-instance coordination bugs, routing load imbalance, stale snapshot propagation, admission control edge cases, horizontal scaling assumptions.
9. **Performance & Scalability** — O(n²) where O(n) suffices, hot-path allocations, map iteration in loops, memory growth, degradation at 1000+ requests or 10+ instances.
10. **Security & Robustness** — input validation completeness, panic paths from user input, resource exhaustion vectors, degenerate input handling (empty, zero, NaN, Inf).
