# RFC to Plan: Claude Prompt Template

Use this prompt after the team agrees on an RFC (tracking issue with holes/surfaces/contracts). Give it to Claude to encode the design into a machine-checkable plan and create sub-issues.

For archon-specific encoding steps: see [archon/encode-plan.md](../contributing/archon/encode-plan.md).

---

## Prompt

```
Here is the agreed tracking issue: #<NUMBER> [paste link or content]

Read the tracking issue discussions and final decisions. Then:

1. ENCODE: Follow docs/contributing/archon/encode-plan.md to encode the agreed
   holes, surfaces, contracts, and allowed imports into a .archon file.

2. COMPILE + DISTANCE: Compile the plan and measure baseline distance.

3. SUB-ISSUES: Create sub-issues under the tracking issue:
   - Sub-issue 0: "Encode .archon plan + persist to specs/NNN-feature/"
     This is always the first PR — commits the plan files to the feature branch.
     Expected dist: baseline (no reduction — just persists the plan).
   - Sub-issues 1–N: one per hole, ordered by dependency arrows.
     Each sub-issue body contains:
     - Which hole it fills
     - Surface to implement (plain English)
     - Contracts to satisfy
     - Allowed imports (whitelist)
     - Expected dist reduction (e.g., "dist 13 → 10, fills H2 + 3 arrows")
     - Which sub-issues must land first (dependencies)
     - The line: `archon-plan: specs/NNN-feature/feature.plan.json`
       (CI uses this to auto-detect the plan for dist tracking)

   IMPORTANT: Every sub-issue (including sub-issue 0) must contain the
   `archon-plan:` line with the same path. This is how CI knows to run
   both the standard delta review AND the plan-based dist tracking.

4. Report: summary of baseline dist, number of holes, delivery order.

Use the sub-issue format from docs/contributing/templates/archon-issue-examples.md.
Each sub-issue must include the full hole content (surface, contracts,
evidence types, invariants, no-op default) so blis-pr-review can verify
delivery without reading .archon files.
```

---

## Output

After running this prompt, you will have:
- Sub-issues created under the tracking issue with delivery order
- Each sub-issue is a self-contained, plain-English work order
- Baseline distance reported
- Sub-issue 0 ready for PR0 (persists plan files to `specs/NNN-feature/`)

From PR1 onward, CI tracks dist against the plan via `/archon-pr-review --plan`.
