# Macro Plan Template (Multi-PR Feature Plan)

This template defines the output format for a macro-level implementation plan. Use this when a feature spans 2+ PRs and requires a dependency DAG between them.

!!! note "For Claude Code users"
    The `writing-plans` skill generates plans from this template automatically.
    The agent prompt version is at [`macro-plan-prompt.md`](macro-plan-prompt.md).

**Prerequisite:** Before writing a macro plan, read [design-guidelines.md](design-guidelines.md) — it covers DES foundations, module architecture, and the extension framework. The macro plan must be consistent with these guidelines.

**For comprehensive guidance:** The [detailed prompt version](macro-plan-prompt.md) contains analytical frameworks (Banks et al. model scoping criteria), constraints (concept model 80-line limit, mandatory validation when cost-of-being-wrong ≥ 3 PRs), and worked examples for each section.

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

### A) Executive Summary

Under 15 lines. What this feature adds, why it matters, and how it fits in the system. This is the human-review core — a reviewer should understand the plan's scope from this section alone.

### B) High-Level Objectives and Model Scoping

- 3–7 crisp objectives
- Explicit non-goals
- Model scoping table: what is modeled, simplified, or omitted, with justification for each simplification

| Component | Modeled | Simplified | Omitted | Justification |
|-----------|---------|------------|---------|---------------|
| *(example)* Scaling latency | — | Fixed delay | — | Same steady-state throughput |

### C) Concept Model

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

The concept model must fit in under 80 lines.

### D) Architectural Evolution

Current architecture → target architecture. What new packages, interfaces, or event types are introduced?

### E) PR Series (Ordered)

Each PR entry includes:
- Scope and deliverables
- Extension type: policy template / subsystem module / backend swap / tier composition
- Module contract (what this PR guarantees to the next)
- Dependencies (which PRs must merge before this one starts)
- Whether interfaces are frozen or flexible at this stage

### F) Frozen Interfaces

Interfaces that are stable and can be developed against in parallel. Only include signatures for code that has already been merged. Aspirations about unwritten code use behavioral descriptions, not Go syntax.

### G) Dependency DAG

Visual or tabular dependency graph showing which PRs can be parallelized and which are sequential.

### H) Risk Register

For each non-obvious architectural decision:
- Risk description
- Cost of being wrong (in PRs of rework)
- **If cost-of-being-wrong ≥ 3 PRs, validation is MANDATORY** with specific success criteria and an abort plan
- Abort plan (what changes if validation fails)

### I) Cross-Cutting Infrastructure

Test infrastructure, documentation, and CI changes — each assigned to a specific PR. CLAUDE.md update ownership: the PR that causes the change updates it.

### J) Extension Friction Assessment

For each new module boundary: how many files must change to add one more variant? Compare against reference targets in [design-guidelines.md](design-guidelines.md) Section 4.5.

### K) Design Bug Prevention

Checklist to prevent common macro-plan anti-patterns:

- [ ] No scaffolding creep (every struct/method/flag exercised by end of introducing PR)
- [ ] No documentation drift (CLAUDE.md updated in same PR that causes the change)
- [ ] No test infrastructure duplication (shared packages created early)
- [ ] No golden dataset staleness (regeneration steps included)
- [ ] No DES-specific anti-patterns: Type Catalog trap, fidelity for its own sake, golden without invariant, mixing exogenous and endogenous events
- [ ] New events classified as exogenous (external arrivals) or endogenous (internal scheduling)
- [ ] State vs statistics separation maintained (event-driven state vs aggregated statistics)
- [ ] Model scoping applies Banks et al. criteria (what questions does this answer? what must be modeled vs simplified vs omitted?)

---

## Appendix: Repository Recon

Before writing the plan, inspect the codebase and document:
- Top-level packages and responsibilities
- Core data structures and interfaces
- Key invariants and assumptions
- Current module boundaries vs target module map
- Architectural constraints that must not be violated

Separate: confirmed facts (with file:line citations), inferred behavior (labeled), and open uncertainties.
