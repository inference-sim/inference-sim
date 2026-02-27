# Universal Convergence Protocol

**Status:** Active (v1.0 — extracted 2026-02-26 from hypothesis.md)

This document defines the convergence protocol used by all review gates across BLIS workflows:
- **PR workflow** (docs/contributing/pr-workflow.md): Plan Review (10 perspectives), Code Review (10 perspectives)
- **Hypothesis workflow** (docs/contributing/hypothesis.md): Design Review (5 perspectives), Code Review (5 perspectives), FINDINGS Review (10 perspectives)
- **Design process** (docs/contributing/design-process.md): Design Doc Review (8 perspectives)
- **Macro-plan process** (docs/contributing/macro-planning.md): Macro Plan Review (8 perspectives)

> **Executable implementation:** The `convergence-review` skill automates this protocol — dispatching perspectives, tallying findings independently, and enforcing the re-run gate. Invoke with `/convergence-review <gate-type> [artifact-path] [--model opus|sonnet|haiku]` (default: `haiku`).

---

## The Protocol

1. Run all N perspectives in parallel (one round)
2. Collect all findings; each classified as CRITICAL / IMPORTANT / SUGGESTION
3. **If zero CRITICAL and zero IMPORTANT across all N reviewers:** Converged — proceed to next step
4. **If any CRITICAL or IMPORTANT from any reviewer:** Fix all issues, return to step 1 (re-run **entire** round)
5. Repeat until convergence

## Rules

- **Max 10 rounds per gate.** Each gate has its own independent round counter. If a gate fails to converge within 10 rounds, suspend the work: document the remaining issues as future work.
- **No minimum round count.** Convergence in Round 1 is valid if no reviewer flags any CRITICAL or IMPORTANT item.
- **Hard gate — NO EXCEPTIONS.** You MUST re-run after fixes. You may NOT skip the re-run, propose alternative steps, or rationalize that fixes were "trivial enough." The re-run is the only evidence of convergence. This is non-negotiable.
- **SUGGESTION-level items** (documentation nits, cosmetic fixes, off-by-one line citations) do not block convergence.

## Severity Levels

Each reviewer must classify every finding:

- **CRITICAL**: Must fix before proceeding. Examples: missing control experiment (RCV-4), status classification contradicted by data, silent data loss in analyzer, cross-document contradiction.
- **IMPORTANT**: Should fix before proceeding. The key test: **would proceeding with this unfixed item mislead a reader or produce incorrect conclusions?** Examples: sub-threshold effect size in one seed, stale text contradicting current results, undocumented confound.
- **SUGGESTION**: Does not affect correctness or reader understanding. Examples: off-by-one line citation (±2 lines), cosmetic terminology, style consistency.

**When in doubt between IMPORTANT and SUGGESTION:** If fixing the item would change any conclusion, metric, or user guidance, it is IMPORTANT. If it would only improve readability without changing any conclusion, it is SUGGESTION. If multiple reviewers classify the same item at different severities, the highest severity applies.

## Agent Failure Handling

- **Timeout:** 5 minutes per reviewer agent. If an agent exceeds this, check its output file and restart if stalled.
- **Failure:** If a reviewer agent fails or hangs, fall back to performing that review directly (read the artifact yourself with that reviewer's checklist). Do not skip a reviewer perspective.
- **External contributors:** Submit your artifacts via PR. Maintainers will run the review protocol on your behalf.

## Expected Convergence Rates

Gates with more perspectives (FINDINGS Review: 10) will naturally converge more slowly than gates with fewer (Design Review: 5). This is correct behavior — more eyes = higher quality bar. Typical expectations:
- Hypothesis Design Review (5 perspectives): 1-2 rounds *(empirical, from PR #310-#433)*
- Hypothesis Code Review (5 perspectives): 1-3 rounds *(empirical)*
- Design Doc Review (8 perspectives): 1-2 rounds *(estimated — no empirical data yet)*
- Macro Plan Review (8 perspectives): 1-2 rounds *(estimated)*
- PR Plan/Code Review (10 perspectives): 1-3 rounds *(empirical, from PR #381-#433)*
- FINDINGS Review (10 perspectives): 1-5 rounds *(empirical)*

## References

- PR workflow: [docs/contributing/pr-workflow.md](pr-workflow.md)
- Hypothesis workflow: [docs/contributing/hypothesis.md](hypothesis.md)
- Design process: [docs/contributing/design-process.md](design-process.md)
- Macro-plan process: [docs/contributing/macro-planning.md](macro-planning.md)
