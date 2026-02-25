# PR Review Perspective Prompts

Reference file for the convergence-review skill. Contains exact prompts for the 20 PR review perspectives across plan review and code review gates.

**Canonical source:** `docs/process/pr-workflow.md` (v3.0). If prompts here diverge from pr-workflow.md, the process doc is authoritative.

**Dispatch pattern:** Launch each perspective as a parallel Task agent:
```
Task(subagent_type="general-purpose", model=REVIEW_MODEL, run_in_background=True,
     prompt="<prompt from below>\n\n<artifact content>")
```
Model selection is controlled by the `--model` flag in the convergence-review skill (default: `haiku`).

---

## Section A: PR Plan Review (10 perspectives) — Step 2.5

### PP-1: Substance & Design

```
Review this implementation plan for substance: Are the behavioral contracts logically sound? Are there mathematical errors, scale mismatches, or unit confusions? Could the design actually achieve what the contracts promise? Check formulas, thresholds, and edge cases from first principles — not just structural completeness.

PLAN CONTENTS:
<paste plan file>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PP-2: Cross-Document Consistency

```
Does this micro plan's scope match the source document? Are file paths consistent with the actual codebase? Does the deviation log account for all differences between what the source says and what the micro plan does? Check for stale references to completed PRs or removed files.

PLAN CONTENTS:
<paste plan file>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PP-3: Architecture Boundary Verification

```
Does this plan maintain architectural boundaries? Check:
(1) Individual instances don't access cluster-level state
(2) Types are in the right packages (sim/ vs sim/cluster/ vs cmd/)
(3) No import cycles introduced
(4) Does the plan introduce multiple construction sites for the same type?
(5) Does adding one field to a new type require >3 files?
(6) Does library code (sim/) call logrus.Fatalf anywhere in new code?
(7) Dependency direction: cmd/ → sim/cluster/ → sim/ (never reversed)

PLAN CONTENTS:
<paste plan file>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PP-4: Codebase Readiness

```
We're about to implement this PR. Review the codebase for readiness. Check each file the plan will modify for:
- Stale comments ("planned for PR N" where N is completed)
- Pre-existing bugs that would complicate implementation
- Missing dependencies
- Unclear insertion points
- TODO/FIXME items in the modification zone

PLAN CONTENTS:
<paste plan file>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PP-5: Structural Validation (perform directly, no agent)

> **For Claude:** Perform these 4 checks directly. Do NOT dispatch an agent.

**Check 1 — Task Dependencies:**
For each task, verify it can actually start given what comes before it. Trace the dependency chain: what files does each task create/modify? Does any task require a file or type that hasn't been created yet?

**Check 2 — Template Completeness:**
Verify all sections from `docs/templates/micro-plan.md` are present and non-empty: Header, Part 1 (A-E), Part 2 (F-I), Part 3 (J), Appendix.

**Check 3 — Executive Summary Clarity:**
Read the executive summary as if you're a new team member. Is the scope clear without reading the rest?

**Check 4 — Under-specified Tasks:**
For each task, verify it has complete code. Flag any step an executing agent would need to figure out on its own.

### PP-6: DES Expert

```
Review this plan as a discrete-event simulation expert. Check for:
- Event ordering bugs in the proposed design
- Clock monotonicity violations (INV-3)
- Stale signal propagation between event types (INV-7)
- Heap priority errors (cluster uses (timestamp, priority, seqID))
- Event-driven race conditions
- Work-conserving property violations (INV-8)
- Incorrect assumptions about DES event processing semantics

PLAN CONTENTS:
<paste plan file>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PP-7: vLLM/SGLang Expert

```
Review this plan as a vLLM/SGLang inference serving expert. Check for:
- Batching semantics that don't match real continuous-batching servers
- KV cache eviction policies that differ from vLLM's implementation
- Chunked prefill behavior mismatches
- Preemption policy differences from vLLM
- Missing scheduling features that real servers have
- Flag any assumption about LLM serving that this plan gets wrong

PLAN CONTENTS:
<paste plan file>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PP-8: Distributed Inference Platform Expert

```
Review this plan as a distributed inference platform expert (llm-d, KServe, vLLM multi-node). Check for:
- Multi-instance coordination bugs
- Routing load imbalance under high request rates
- Stale snapshot propagation between instances
- Admission control edge cases at scale
- Horizontal scaling assumption violations
- Prefix-affinity routing correctness across instances

PLAN CONTENTS:
<paste plan file>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PP-9: Performance & Scalability

```
Review this plan as a performance and scalability analyst. Check for:
- Algorithmic complexity issues (O(n^2) where O(n) suffices)
- Unnecessary allocations in hot paths (event loop, batch formation)
- Map iteration in O(n) loops that could grow
- Benchmark-sensitive changes
- Memory growth patterns
- Changes that would degrade performance at 1000+ requests or 10+ instances

PLAN CONTENTS:
<paste plan file>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PP-10: Security & Robustness

```
Review this plan as a security and robustness reviewer. Check for:
- Input validation completeness (all CLI flags, YAML fields, config values)
- Panic paths reachable from user input (R3, R6)
- Resource exhaustion vectors (unbounded loops, unlimited memory growth) (R19)
- Degenerate input handling (empty, zero, negative, NaN, Inf) (R3, R20)
- Configuration injection risks
- Silent data loss paths (R1)

PLAN CONTENTS:
<paste plan file>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

---

## Section B: PR Code Review (10 perspectives) — Step 4.5

### PC-1: Substance & Design

```
Review this diff for substance: Are there logic bugs, design mismatches between contracts and implementation, mathematical errors, or silent regressions? Check from first principles — not just structural patterns. Does the implementation actually achieve what the behavioral contracts promise?

DIFF:
<paste git diff output>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PC-2: Code Quality + Antipattern Check

```
Review this diff for code quality. Check all of these:
(1) Any new error paths that use `continue` or early `return` — do they clean up partial state? (R1, R5)
(2) Any map iteration that accumulates floats — are keys sorted? (R2)
(3) Any struct field added — are all construction sites updated? (R4)
(4) Does library code (sim/) call logrus.Fatalf anywhere in new code? (R6)
(5) Any exported mutable maps — should they be unexported with IsValid*() accessors? (R8)
(6) Any YAML config fields using float64 instead of *float64 where zero is valid? (R9)
(7) Any division where the denominator derives from runtime state without a zero guard? (R11)
(8) Any new interface with methods only meaningful for one implementation? (R13)
(9) Any method >50 lines spanning multiple concerns (scheduling + latency + metrics)? (R14)
(10) Any changes to docs/standards/ files — are CLAUDE.md working copies updated? (DRY)

DIFF:
<paste git diff output>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PC-3: Test Behavioral Quality

```
Review the tests in this diff. For each test, rate as Behavioral, Mixed, or Structural:
- Behavioral: tests observable behavior (GIVEN/WHEN/THEN), survives refactoring
- Mixed: some behavioral assertions, some structural coupling
- Structural: asserts internal structure (field access, type assertions), breaks on refactor

Also check:
- Are there golden dataset tests that lack companion invariant tests? (R7)
- Do tests verify laws (conservation, monotonicity, causality) not just values?
- Would each test still pass if the implementation were completely rewritten?

DIFF:
<paste git diff output>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PC-4: Getting-Started Experience

```
Review this diff for user and contributor experience. Simulate both journeys:
(1) A user doing capacity planning with the CLI — would they find everything they need?
(2) A contributor adding a new algorithm — would they know how to extend this?

Check:
- Missing example files or CLI documentation
- Undocumented output metrics
- Incomplete contributor guide updates
- Unclear extension points
- README not updated for new features

DIFF:
<paste git diff output>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PC-5: Automated Reviewer Simulation

```
The upstream community uses GitHub Copilot, Claude, and Codex to review PRs. Do a rigorous check so this will pass their review. Look for:
- Exported mutable globals
- User-controlled panic paths
- YAML typo acceptance (should use KnownFields(true))
- NaN/Inf validation gaps
- Redundant or dead code
- Style inconsistencies
- Missing error returns

DIFF:
<paste git diff output>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PC-6: DES Expert

```
Review this diff as a discrete-event simulation expert. Check for:
- Event ordering bugs in the implementation
- Clock monotonicity violations (INV-3)
- Stale signal propagation between event types (INV-7)
- Heap priority errors
- Work-conserving property violations (INV-8)
- Event-driven race conditions

DIFF:
<paste git diff output>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PC-7: vLLM/SGLang Expert

```
Review this diff as a vLLM/SGLang inference serving expert. Check for:
- Batching semantics that don't match real continuous-batching servers
- KV cache eviction mismatches with vLLM
- Chunked prefill behavior errors
- Preemption policy differences
- Missing scheduling features
- Flag any assumption about LLM serving that this code gets wrong

DIFF:
<paste git diff output>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PC-8: Distributed Inference Platform Expert

```
Review this diff as a distributed inference platform expert (llm-d, KServe, vLLM multi-node). Check for:
- Multi-instance coordination bugs
- Routing load imbalance
- Stale snapshot propagation
- Admission control edge cases
- Horizontal scaling assumption violations
- Prefix-affinity routing correctness

DIFF:
<paste git diff output>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PC-9: Performance & Scalability

```
Review this diff as a performance and scalability analyst. Check for:
- Algorithmic complexity regressions (O(n^2) where O(n) suffices)
- Unnecessary allocations in hot paths
- Map iteration in O(n) loops
- Benchmark-sensitive changes
- Memory growth patterns
- Changes degrading performance at 1000+ requests or 10+ instances

DIFF:
<paste git diff output>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PC-10: Security & Robustness

```
Review this diff as a security and robustness reviewer. Check for:
- Input validation completeness (CLI flags, YAML fields, config values)
- Panic paths reachable from user input
- Resource exhaustion vectors (unbounded loops, unlimited memory growth)
- Degenerate input handling (empty, zero, NaN, Inf)
- Configuration injection risks
- Silent data loss in error paths

DIFF:
<paste git diff output>

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```
