# H-Cross-Model Erratum Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct factual errors in the H-Cross-Model FINDINGS.md where the prefix cache was incorrectly reported as non-functional.

**The problem today:** The H-Cross-Model FINDINGS.md contains three incorrect claims: (1) "the prefix cache is NOT producing KV cache hits," (2) "H9 prefix cache not producing cache hits with synthetic tokens," and (3) "H9 control confirmed cache non-hits." These claims were debunked by systematic debugging (issue #399): the prefix cache works correctly with a measured hit rate of 0.4000 matching theoretical prediction exactly. The analysis error was that `prefix_length` is additive to `input_distribution.value` (generator.go:171-172 prepends prefix tokens), so total input = 512 prefix + 512 sampled = 1024, not 512. With 1024 total and 512 cached, the 512 cache-miss tokens produce the same step time as prefix=0 with 512 input — masking the cache benefit in TTFT mean.

**What this PR adds:**
1. Erratum notes at 4 locations in FINDINGS.md correcting the analysis error with the verified explanation
2. Updated findings classification table (H9 row: "Open question" → "Analysis error — corrected")
3. Updated evidence quality table (control experiment row: corrected characterization)

**Why this matters:** Incorrect experiment findings mislead future researchers. The erratum prevents someone from investigating a non-existent bug and documents the correct prefix cache behavior for future experiments.

**Architecture:** Docs-only change — single file modified (`hypotheses/h-cross-model/FINDINGS.md`). No code, no tests.

**Source:** GitHub issue #399 (closed as not-a-bug)

**Closes:** N/A — #399 already closed

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds erratum notes to 4 locations in `hypotheses/h-cross-model/FINDINGS.md` where the prefix cache was incorrectly reported as non-functional. The error was in the expected-value calculation, not the simulator code. No code changes.

### B) Behavioral Contracts

**BC-1: Erratum accuracy**
- GIVEN the H-Cross-Model FINDINGS.md with incorrect H9 cache analysis
- WHEN erratum notes are added
- THEN each erratum note MUST contain: (1) the incorrect claim, (2) the correct explanation referencing issue #399, (3) the verified cache hit rate (0.4000 for N=10)

**BC-2: Factual preservation**
- GIVEN the existing FINDINGS.md content
- WHEN erratum notes are added
- THEN all original text MUST be preserved (not deleted) — errata annotate, they don't rewrite history

**BC-3: Status update**
- GIVEN the H9 finding was classified as "Open question"
- WHEN the erratum is applied
- THEN the findings classification table MUST update H9 to "Analysis error — corrected" with action "See erratum; #399 closed as not-a-bug"

### C) Component Interaction

No component interaction — single-file documentation change.

### D) Deviation Log

No deviations from source document.

### E) Review Guide

- **The tricky part:** Ensuring the erratum explanation is technically precise (prefix is additive, not replacing).
- **What to scrutinize:** The mathematical claim that 22.74ms matches the "cache working" prediction.
- **What's safe to skim:** Boilerplate erratum formatting.
- **Known debt:** None.

---

## Part 2: Executable Implementation

### F) Implementation Overview

- Modify: `hypotheses/h-cross-model/FINDINGS.md` (4 locations)
- No files created; no tests; no code changes

### G) Task Breakdown

### Task 1: Add erratum notes to H-Cross-Model FINDINGS.md

**Contracts Implemented:** BC-1, BC-2, BC-3

**Files:**
- Modify: `hypotheses/h-cross-model/FINDINGS.md` (lines 92-98, 120, 139, 170)

**Step 1: Add erratum to H9 detailed results (lines 92-98)**

After the existing H9 paragraph (ending with "filed as an open question, not a confirmed mechanism."), add an erratum block:

```markdown
> **Erratum (2026-02-23, #399 closed as not-a-bug):** The above analysis is incorrect. The prefix cache IS working correctly. The error was in the expected TTFT calculation: `prefix_length=512` + `input_distribution.value=512` produces 1024 total input tokens (prefix is prepended at `generator.go:171-172`), not 512. With 1024 total and 512 prefix cached, requests 2-N have 512 cache-miss tokens — identical step time as prefix=0 with 512 input. The ~1ms TTFT increase is the cold-start penalty on request 1 averaged over N=10: `beta1 * 512 / 10 = 998us`. Verified experimentally: Cache Hit Rate = 0.4000 (matches theoretical 288/720 exactly), mean TTFT = 22.735ms (matches prediction of 22.714ms within 0.02ms). See #399 for full derivation.
```

**Step 2: Add erratum to root cause analysis (line 120)**

After the existing paragraph about "H9 prefix cache non-hits," add:

```markdown
> **Erratum (2026-02-23, #399):** The claim above that "the prefix cache is NOT producing KV cache hits" is incorrect. The cache produces hits at exactly the theoretical rate (0.4000 for N=10 requests). The analysis error was assuming total input = 512 when it is actually 1024 (prefix is additive). See erratum in Results section above.
```

**Step 3: Update findings classification table (line 139)**

Change:
```
| H9 prefix cache not producing cache hits with synthetic tokens | Open question | Requires code investigation of KV cache block-hash matching with workload generator tokens |
```
To:
```
| ~~H9 prefix cache not producing cache hits with synthetic tokens~~ H9 prefix cache works correctly; analysis used wrong expected TTFT | ~~Open question~~ Analysis error — corrected | See erratum; #399 closed as not-a-bug |
```

**Step 4: Update evidence quality table (line 170)**

Change:
```
| Control experiments (Round 2) | 2 executed (Prefix-Affinity high-rate, H9 isolation) | Prefix-Affinity control refuted rate-dependent hypothesis; H9 control confirmed cache non-hits |
```
To:
```
| Control experiments (Round 2) | 2 executed (Prefix-Affinity high-rate, H9 isolation) | Prefix-Affinity control refuted rate-dependent hypothesis; ~~H9 control confirmed cache non-hits~~ H9 control actually confirms cache IS working (see erratum, #399) |
```

**Step 5: Verify no broken markdown**

Run: `python3 -c "open('hypotheses/h-cross-model/FINDINGS.md').read()"` to verify the file is valid.

**Step 6: Commit**

```bash
git add hypotheses/h-cross-model/FINDINGS.md
git commit -m "docs(hypotheses): add erratum to H-Cross-Model H9 prefix cache analysis

The H9 analysis incorrectly claimed the prefix cache was not producing
hits. Investigation (#399) proved the cache works correctly — the error
was in the expected TTFT calculation (prefix tokens are additive to
sampled input, producing 1024 total tokens, not 512). Cache Hit Rate
0.4000 matches theoretical prediction exactly.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|------------------------|
| BC-1 | Task 1 | Manual | Verify erratum notes contain all 3 required elements |
| BC-2 | Task 1 | Manual | Verify original text preserved (no deletions) |
| BC-3 | Task 1 | Manual | Verify findings table updated |

No automated tests — docs-only change.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Erratum wording technically imprecise | Low | Medium | Mathematical derivation verified experimentally |
| Markdown formatting broken | Low | Low | Step 5 verification |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No code changes
- [x] Deviation log reviewed — no deviations
- [x] Each task produces verifiable output
- [x] All contracts mapped to Task 1
- [x] No CLAUDE.md update needed (no code/structure changes)
- [x] No golden dataset update needed
- [x] No construction site audit needed (no struct changes)

---

## Appendix: File-Level Implementation Details

**File: `hypotheses/h-cross-model/FINDINGS.md`**

**Purpose:** Add erratum notes correcting the H9 prefix cache analysis at 4 locations.

**Locations:**
1. After line 98 (H9 detailed results) — new erratum blockquote
2. After line 120 (root cause analysis) — new erratum blockquote
3. Line 139 (findings classification table) — inline correction with strikethrough
4. Line 170 (evidence quality table) — inline correction with strikethrough

**Key principle:** Use `> **Erratum**` blockquotes to annotate without rewriting. Use `~~strikethrough~~` for inline table corrections to preserve the original text while showing the correction.
