# PR Review Perspective Prompts

Reference file for the convergence-review skill. Contains exact prompts for the 27 PR review perspectives across plan review, code review, and docs review gates.

**Canonical source:** `docs/contributing/pr-workflow.md` (v4.0). After the human-first rewrite, pr-workflow.md contains the same checklist content in human-readable format; this file preserves the agent dispatch format. Content is aligned; format differs intentionally. If checklist *content* diverges, pr-workflow.md is authoritative.

**Dispatch pattern:** All perspectives are passed to a single foreground agent in one call. The artifact is sent once; each perspective appears as a `## [<ID>] <Name>` section. See SKILL.md Phase A Step 3 for the full assembly specification. Model selection is controlled by the `--model` flag in the convergence-review skill (default: `haiku`).

---

## Section A: PR Plan Review (10 perspectives) — Step 2.5

### PP-1: Substance & Design

```
Review this implementation plan for substance: Are the behavioral contracts logically sound? Are there mathematical errors, scale mismatches, or unit confusions? Could the design actually achieve what the contracts promise? Check formulas, thresholds, and edge cases from first principles — not just structural completeness.

PLAN CONTENTS:
<paste plan file>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PP-2: Cross-Document Consistency

```
Does this micro plan's scope match the source document? Are file paths consistent with the actual codebase? Does the deviation log account for all differences between what the source says and what the micro plan does? Check for stale references to completed PRs or removed files.

PLAN CONTENTS:
<paste plan file>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PP-5: Structural Validation (perform directly, no agent)

> **For Claude:** Perform these 4 checks directly. Do NOT dispatch an agent.

**Check 1 — Task Dependencies:**
For each task, verify it can actually start given what comes before it. Trace the dependency chain: what files does each task create/modify? Does any task require a file or type that hasn't been created yet?

**Check 2 — Template Completeness:**
Verify all sections from `docs/contributing/templates/micro-plan-prompt.md` are present and non-empty: Header, Part 1 (A-E), Part 2 (F-I), Part 3 (J), Appendix.

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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

---

## Section B: PR Code Review (10 perspectives) — Step 4.5

### PC-1: Substance & Design

```
Review this diff for substance: Are there logic bugs, design mismatches between contracts and implementation, mathematical errors, or silent regressions? Check from first principles — not just structural patterns. Does the implementation actually achieve what the behavioral contracts promise?

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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
(10) Any changes to docs/contributing/standards/ files — are CLAUDE.md working copies updated? (DRY)

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
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

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

---

## Section C: PR Docs Review (7 perspectives) — Step 4.5 (docs-only PRs)

### PD-1: Substance & Accuracy

```
Review this diff for substance and factual accuracy. Check for:
- Factual claims that are wrong (wrong issue numbers, wrong PR citations, wrong counts)
- File paths that don't exist or have been renamed
- Rule or invariant references that are stale (e.g., "R1-R20" when the current range is R1-R23)
- Count claims that don't match the actual count (e.g., "7 gate types" when there are 8)
- Procedure steps that describe behavior not matching the current implementation

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PD-2: Cross-Document Consistency

```
Review this diff for cross-document consistency. Check for:
- Working copies not updated when their canonical source was changed (check the source-of-truth map in docs/contributing/standards/principles.md)
- CLAUDE.md sections that reference file paths, rules, or counts that this diff changes
- Stale references in one document to content in another document that was modified
- Scope mismatch: did this change touch a canonical source but miss one or more of its working copies?

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PD-3: Canonical Source Integrity

```
Review this diff for canonical source integrity. Check for:
- A canonical source losing its "canonical" designation (e.g., a "canonical source" header removed)
- Contradictions introduced between two canonical sources (e.g., two files that should agree but now say different things)
- A working copy claiming authority it shouldn't have (e.g., "If this section diverges, THIS FILE is authoritative" when it should be the canonical source)
- A document that previously deferred to a canonical source now stating its own version of the truth

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PD-4: Completeness

```
Review this diff for completeness. Check for:
- Structured documents (tables, checklists, numbered lists) where a new entry was added but a parallel list elsewhere wasn't updated
- Count references that are now off by one (e.g., "8 perspectives" in the table of contents but only 7 in the list)
- Acceptance criteria or checklist items that are referenced but not fulfilled by the diff
- Link targets that are mentioned in the new/changed text but don't exist in the repo
- Section references (e.g., "see Section C") that now point to the wrong section number after a renumbering

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PD-5: Structural Validation (perform directly, no agent)

> **For Claude:** Perform these 4 checks directly. Do NOT dispatch an agent.

**Check 1 — Template Compliance:**
For each modified document, verify it follows its stated template or structure. If it's a table, all columns present. If it's a Gherkin-style contract, GIVEN/WHEN/THEN all present. If it's a checklist, all items have checkboxes.

**Check 2 — Internal Link Validity:**
For each internal link or file reference added or changed in the diff, verify the target file or section exists. Pay particular attention to links using the `[text](path)` format — check that the path resolves from the document's location.

**Check 3 — Source-of-Truth Map Consistency:**
Check `docs/contributing/standards/principles.md` for the source-of-truth map. For each canonical source modified in this diff, verify all listed working copies are also updated in this diff (or are provably unaffected).

**Check 4 — Parallel Structure Preservation:**
If the modified document uses parallel structure (e.g., all gate types have the same columns, all perspectives have the same footer), verify the new content preserves that parallel structure exactly.

### PD-6: Getting-Started Experience

```
Review this diff from the perspective of a new contributor. Simulate two journeys:
(1) A new contributor reading the docs to understand BLIS workflow for the first time
(2) An experienced contributor using updated docs to execute a specific workflow step

Check for:
- Instructions that are now inconsistent with each other (step A says one thing, step B contradicts it)
- A new contributor following instructions exactly and getting stuck because a prerequisite step was removed or changed
- Examples that illustrate the old behavior but haven't been updated to show the new behavior
- Terminology introduced without definition (jargon added without a "means X" explanation)

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PD-7: DRY Compliance

```
Review this diff for DRY (Don't Repeat Yourself) violations. Focus on STRUCTURAL duplication — do not re-report cross-document sync issues already covered by PD-2 (Cross-Document Consistency). PD-2 checks "did this diff update all working copies?" — PD-7 checks "does this diff introduce new duplication or orphan a copy?"

Check for:
- Content duplicated in two or more places without a "canonical source" header on one of them (new duplication introduced by this diff)
- A fact stated in multiple documents that could diverge — is there a clear canonical source that others defer to? (structural question about the repo's doc architecture)
- New content added that is already stated elsewhere — should it be a link/reference instead of a copy?
- Content removed from one place but not removed from its copies, creating a shadow/orphaned version

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```
