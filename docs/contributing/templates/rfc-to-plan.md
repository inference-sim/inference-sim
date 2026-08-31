# RFC to Plan: Claude Prompt Template

Use this prompt after the team agrees on an RFC (tracking issue with holes/surfaces/contracts). Give it to Claude to encode the design into a machine-checkable plan and create sub-issues.

For archon-specific encoding steps: see [archon/encode-plan.md](../archon/encode-plan.md).

---

## Prompt

```
Here is the agreed tracking issue: #<NUMBER> [paste link or content]

Read the tracking issue discussions and final decisions. Then:

1. ENCODE: Follow docs/contributing/archon/encode-plan.md to encode the agreed
   holes, surfaces, contracts, and allowed imports into a .archon file.

2. COMPILE + DISTANCE: Compile the plan and measure baseline distance.

3. SUB-ISSUES: Create sub-issues under the tracking issue:
   - Sub-issue 0: "Create feature branch + encode .archon plan + persist to specs/NNN-feature/"
     This is always the first PR. It creates the feature branch (feature/<name>),
     commits the plan files, and pushes. No implementation code.
     Expected dist: baseline (no reduction — just persists the plan).
   - Sub-issues 1–N: one per hole, ordered by dependency arrows.
     PRs for these target the feature branch (not main).
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
   `archon-plan:` line with the same path. This is how CI knows to add
   plan-based dist tracking to the review.

   The path must end in `.json` and the plan must be committed to the
   branch — CI reads it out of git, not the working tree. Substitute the
   placeholders: a literal `specs/[NNN-feature]/...` is rejected and the
   review posts a warning instead of running the dist ratchet.

   The line must also be copied into each PR body, including the final
   PR to main. CI reads the PR body first and falls back to the bodies of
   the issues the PR closes, but GitHub only links closing issues for a PR
   targeting the default branch — a hole PR targets the feature branch, so
   nothing is linked. The final PR does close the tracking issue, which
   carries no such line. In both cases the PR body is the only place CI
   can find it.

4. Report: summary of baseline dist, number of holes, delivery order.

Use the sub-issue format from docs/contributing/templates/archon-issue-examples.md.
Each sub-issue must include the full hole content (surface, contracts,
evidence types, invariants, no-op default) so blis-pr-review can verify
delivery without reading .archon files.

CROSS-CHECK before filing: For each hole, verify its surface actually
requires every arrow the plan declares from it. If a declared import
isn't reachable from the surface as written, fix the surface or the
arrow before filing. The sub-issue body and the .archon file must agree.
```

---

## Branch Strategy

- **PR0** creates `feature/<name>` branch off main. Commits plan files. No PR to main — just the branch.
- **PR1+** are worktrees off the feature branch, with PRs targeting the feature branch.
- **Final PR** merges the feature branch → main (includes plan files + all implementation).

## Output

After running this prompt, you will have:
- Sub-issues created under the tracking issue with delivery order
- Each sub-issue is a self-contained, plain-English work order
- Baseline distance reported
- Sub-issue 0 ready for PR0 (creates feature branch + persists plan files)

From PR1 onward, `/archon-pr-review --plan` tracks dist against the plan on the feature branch.
