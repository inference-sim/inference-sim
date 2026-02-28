# Macro Plan Template (Multi-PR Feature Plan)

This template defines the output format for a macro-level implementation plan. Use this when a feature spans 2+ PRs and requires a dependency DAG between them.

!!! note "For Claude Code users"
    The macro-planning skill generates plans from this template automatically.
    The agent prompt version is at [`macro-plan-prompt.md`](macro-plan-prompt.md).

**Prerequisite:** Before writing a macro plan, read [design-guidelines.md](design-guidelines.md) — it covers DES foundations, module architecture, and the extension framework. The macro plan must be consistent with these guidelines.

---

## Abstraction Level Rule

A macro plan describes **what to build and in what order**, not how to implement it.

- ✅ Module contracts (observes / controls / owns / invariants / events / extension friction)
- ✅ Frozen interface signatures (facts about already-merged code)
- ❌ Go struct definitions with field lists
- ❌ Method implementations
- ❌ File paths with line numbers
- ❌ Interface signatures for code that hasn't been written yet

**The staleness test:** Would this content mislead if the implementation changes during micro-planning? If yes, it's too concrete for a macro plan.

---

## Sections

### A) High-Level Objectives and Model Scoping

- 3–7 crisp objectives
- Explicit non-goals
- Model scoping table: what is modeled, simplified, or omitted, with justification for each simplification

| Component | Modeled | Simplified | Omitted | Justification |
|-----------|---------|------------|---------|---------------|
| *(example)* Scaling latency | — | Fixed delay | — | Same steady-state throughput |

### B) Concept Model

Building blocks and their interactions. Each block uses the module contract template from [design-guidelines.md](design-guidelines.md) Section 4.3:

- **Observes:** What signals does this module read?
- **Controls:** What decisions does it make?
- **Owns:** What mutable state does it manage exclusively?
- **Invariants:** What properties must always hold?
- **Events:** What events does it produce and consume?
- **Extension friction:** How many files to add one more variant?

Include a **real-system correspondence** table mapping building blocks to llm-d / vLLM / SGLang equivalents:

| Building Block | llm-d | vLLM | SGLang | Other |
|----------------|-------|------|--------|-------|
| *(example)* Router | EPP + routing filter | N/A | N/A | — |

### C) Architectural Evolution

Current architecture → target architecture. What new packages, interfaces, or event types are introduced?

### D) PR Series (Ordered)

Each PR entry includes:
- Scope and deliverables
- Extension type: policy template / subsystem module / backend swap / tier composition
- Module contract (what this PR guarantees to the next)
- Dependencies (which PRs must merge before this one starts)
- Whether interfaces are frozen or flexible at this stage

### E) Frozen Interfaces

Interfaces that are stable and can be developed against in parallel. Only include signatures for code that has already been merged. Aspirations about unwritten code use behavioral descriptions, not Go syntax.

### F) Dependency DAG

Visual or tabular dependency graph showing which PRs can be parallelized and which are sequential.

### G) Risk Register

For each non-obvious architectural decision:
- Risk description
- Cost of being wrong (in PRs of rework)
- Validation gate (if cost ≥ 3 PRs, validation is mandatory with specific success criteria)
- Abort plan (what changes if validation fails)

### H) Cross-Cutting Infrastructure

Test infrastructure, documentation, and CI changes — each assigned to a specific PR. CLAUDE.md update ownership: the PR that causes the change updates it.

### I) Extension Friction Assessment

For each new module boundary: how many files must change to add one more variant? Compare against reference targets in [design-guidelines.md](design-guidelines.md) Section 4.5.

### J) Design Bug Prevention

Checklist to prevent common macro-plan anti-patterns:

- [ ] No scaffolding creep (every struct/method/flag exercised by end of introducing PR)
- [ ] No documentation drift (CLAUDE.md updated in same PR that causes the change)
- [ ] No test infrastructure duplication (shared packages created early)
- [ ] No golden dataset staleness (regeneration steps included)
- [ ] No DES-specific anti-patterns: Type Catalog trap, fidelity for its own sake, golden without invariant
- [ ] State vs statistics separation maintained (event-driven state vs aggregated statistics)

---

## Appendix: Repository Recon

Before writing the plan, inspect the codebase and document:
- Top-level packages and responsibilities
- Core data structures and interfaces
- Key invariants and assumptions
- Current module boundaries vs target module map
- Architectural constraints that must not be violated

Separate: confirmed facts (with file:line citations), inferred behavior (labeled), and open uncertainties.
