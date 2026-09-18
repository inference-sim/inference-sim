---
name: issue-review
description: Thoroughly review a GitHub issue for correctness, evidence quality, scope, parity, and actionability. Produces a VALID / NEEDS WORK / SUPERSEDED / DUPLICATE verdict.
---

# Issue Review Skill

When invoked, perform the following review in order. Work through every section; do not skip sections because earlier ones look clean.

---

## Step 0 — Identify the issue

Resolve the issue to review using the first matching source:

1. **Skill argument**: if a URL was passed (e.g. `https://github.com/owner/repo/issues/123`), parse the owner, repo, and number from it.
2. **GitHub Actions context**: if no argument was supplied but the skill is running inside a claude-code-action trigger (the current issue is already in the conversation context), use that issue — no fetch needed, the body and metadata are already present.
3. **Fallback**: if neither of the above applies, ask the user for the issue URL before proceeding.

When an explicit fetch is needed (cases 1 and 3), run:

```bash
gh issue view <NUMBER> --repo <OWNER>/<REPO>
```

to get the full body, labels, and linked PRs/commits.

---

## Step 1 — Issue Type Identification

Identify which type this issue is and verify the label matches:

| Type | Definition |
|------|-----------|
| **Bug** | Incorrect behavior that violates a contract or produces wrong results. Requires code proof of the incorrect behavior. |
| **Feature request** | New capability or improved UX. Requires clear acceptance criteria. |
| **Design issue** | Architectural decision or disambiguation needed. Requires description of conflicting behaviors or semantics. |
| **Hardening / refactoring** | Improve correctness, modularity, or performance of existing code. Requires motivation (what goes wrong without this, or what becomes possible with it). |
| **Tracking issue** | Umbrella for multiple sub-issues. Should link to each sub-issue explicitly, not contain implementation details. |

> A missing CLI flag is an **enhancement**, not a bug — unless the absence causes incorrect results (then explain the causal chain).

State: **"Issue type: X"** and whether the label is correct.

---

## Step 2 — Validity (all issue types)

### 2a. Grounded in current HEAD of main?

- For **bugs**: read the exact code referenced. Are file paths and line numbers current? Has subsequent work already fixed this?
- For **features**: does the proposed location still exist? Has the interface changed since the issue was filed?
- For **all**: if the issue references a specific commit, has the code changed since then? Is the issue still relevant?

Use `grep`, `Read`, or `git log` to verify. Do not assume paths are current.

### 2b. Sufficient evidence?

- **Bug**: EXACT code path that produces wrong behavior (file:line, snippet, step-by-step trace showing incorrect output). "I observed X" without code proof is a symptom report, not an actionable bug.
- **Feature**: clear acceptance criteria — what does "done" look like? What should the user be able to do that they cannot today?
- **Design**: concrete example of the ambiguity or conflict, with code showing where the two interpretations diverge.
- **Hardening**: before/after contrast — what specific failure or limitation does this address?

---

## Step 3 — Root Cause / Motivation

### 3a. Core problem correctly identified?

- **Bugs**: does the issue distinguish symptom from cause? ("throughput is wrong" is a symptom; "decodeTokens = PI - InputLen misses the prefill-generated token" is a cause.)
- **Features**: is the motivation a real user need or speculative? Is there evidence (user report, experiment finding, parity gap with production system)?
- **Design**: are the conflicting semantics clearly stated? (e.g., "HTTP timeout is wall-clock seconds; DES deadline is simulation ticks — these can diverge when replay speed != 1x")

### 3b. One issue or multiple?

- Each root cause / feature / design question should be its own issue.
- If the investigation uncovered multiple problems, they should be filed separately and cross-referenced.
- Umbrella/tracking issues are fine but should link sub-issues, not contain implementation details.

### 3c. Related / duplicate issues checked?

- Are there open issues with overlapping descriptions? If so, link them.
- If this is a subset of an existing tracking issue, say so.

---

## Step 4 — Proposed Solution (if present)

### 4a. Correct for ALL cases?

- **Bugs**: check every completion path — normal, length-capped (MaxModelLen), zero-output, preempted requests, OutputLen=1, MaxModelLen=InputLen+1.
- **Features**: does the design handle error cases, invalid input, and interaction with existing features?
- **All**: does the proposal use **oracle fields** (`req.OutputTokens`, `req.InputTokens` — set at generation, represent intent) vs **execution fields** (`req.ProgressIndex`, `req.LengthCapped`, `req.State` — represent what happened)? Oracle-based decisions about actual behavior are suspect.

### 4b. Affected files and tests identified?

- Which files need modification? Are the line numbers current?
- Which existing tests will break and need updating?
- What new tests are needed?

---

## Step 5 — Cross-Path Parity (run / replay / observe)

Check whether the issue applies to other command paths:

- If the issue is in `sim/` (the DES kernel): it affects both `blis run` and `blis replay`. State this explicitly.
- If the issue is in `cmd/` or `sim/workload/`: check whether the same pattern exists in `run`, `replay`, AND `observe`.
- If the issue is in the `observe` HTTP path: does the equivalent DES path have a corresponding gap?
- For features: a new flag or spec field for one command should prompt the question — do the other commands need it too?

State which paths are affected and which are not, with justification for any that are excluded.

---

## Step 6 — Scope and Actionability

Can an implementer (human or AI agent) act on this issue without ambiguity?

- Is the expected behavior stated precisely enough to write a test?
- Are there unstated assumptions about the environment, workload, or configuration?
- For features: are the acceptance criteria verifiable? (not "improve UX" but "error message includes minimum TP value")

---

## Step 7 — Verdict

Provide a clear, labeled verdict:

| Verdict | Meaning |
|---------|---------|
| **VALID** | Issue is real, well-scoped, evidence is sufficient, parity is addressed, and an implementer can act on it without ambiguity. |
| **NEEDS WORK** | Specify exactly what is missing: code proof, edge case analysis, parity check, deduplication, scope split, or acceptance criteria. |
| **SUPERSEDED** | Already fixed or made irrelevant by subsequent work — recommend closing with explanation. |
| **DUPLICATE** | Link to the existing issue — recommend closing. |

After the verdict, list any concrete suggestions the issue author should add or change to bring the issue to VALID status (if not already there).
