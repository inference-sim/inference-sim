# Audit and Fix analyze.py Parsers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix broken regex parsers and add defensive warnings across all hypothesis experiment analyzers so parser-output mismatches are caught immediately rather than silently producing wrong data.

**The problem today:** Two hypothesis analyzers (H5, H12) use a regex pattern `r"Preemptions?: (\d+)"` that never matches the actual simulator output format `"Preemption Rate: %.4f"`. These parsers silently default to 0, producing incorrect analysis results. More broadly, all 8 analyzers that parse BLIS output use a "silent zero default" pattern where failed regex matches produce 0 with no warning — the exact bug class that masked H10's preemptions for 2 rounds of experiments.

**What this PR adds:**
1. **Corrected preemption regexes** — H5 and H12 analyzers now parse `"Preemption Rate: <float>"` correctly, matching the actual `cmd/root.go:544` format
2. **Defensive stderr warnings** — all 8 analyzers emit a warning when a section header is present in output but the metric regex within that section fails to match, catching future parser-output mismatches immediately
3. **Audit documentation** — the PR description documents the full cross-reference of all regex patterns vs format strings, serving as a living reference for future hypothesis experiments

**Why this matters:** Hypothesis experiments are the project's primary validation mechanism. Parser bugs that silently produce wrong data undermine the entire experiment framework. This fix closes the gap identified in PR #310 and prevents the same class of bug from recurring.

**Architecture:** Python-only changes in `hypotheses/*/analyze.py`. No Go code changes. Each analyzer gets a small helper function that wraps the regex-with-warning pattern, reducing boilerplate and ensuring consistent defensive behavior.

**Source:** GitHub issue #318

**Closes:** Fixes #318

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes 2 confirmed parser bugs and hardens 8 hypothesis analyzers against future parser-output mismatches.

**What it builds:** Corrected regex patterns for H5/H12 preemption parsing, plus a defensive warning system across all analyzers that detects when output section headers are present but metric regexes fail to match.

**Where it fits:** These are experiment infrastructure files (`hypotheses/*/analyze.py`). They parse BLIS CLI output (`cmd/root.go` format strings and `sim/metrics_utils.go` JSON schema). No simulator code is modified.

**Adjacent components:** `cmd/root.go` (output producer), `sim/metrics_utils.go` (JSON schema). Changes are read-only consumers of these formats.

**Deviation flags:** None — issue #318 scope matches what the audit found.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: H5 Preemption Parsing
- GIVEN simulator output containing `"Preemption Rate: 0.1750"`
- WHEN H5 analyze.py parses the output
- THEN the preemption rate is extracted as float `0.175`, not defaulted to 0
- MECHANISM: Regex changed from `r"Preemptions?: (\d+)"` to `r"Preemption Rate: ([0-9.]+)"`

BC-2: H12 Preemption Parsing
- GIVEN simulator output containing `"Preemption Rate: 0.1750"`
- WHEN H12 analyze.py parses the output
- THEN the preemption rate is extracted as float `0.175`, not defaulted to 0
- MECHANISM: Same regex fix as BC-1

BC-3: Warning on Mismatched Regex
- GIVEN simulator output containing `"=== KV Cache Metrics ==="` section header
- WHEN a metric regex within that section fails to match (e.g., wrong format)
- THEN a warning is emitted to stderr identifying the missing metric and file
- MECHANISM: Helper function checks for section header presence before deciding whether to warn

**Negative Contracts:**

BC-4: No False Warnings
- GIVEN simulator output where the KV Cache Metrics section is absent (all values zero)
- WHEN the analyzer parses the output
- THEN no warning is emitted for missing preemption/cache metrics
- MECHANISM: Warning only fires when section header IS present but metric regex doesn't match

BC-5: No Behavioral Change for Correct Parsers
- GIVEN an analyzer where all regexes already match correctly (H8, H9, H10, H14, prefix-affinity, H3)
- WHEN the analyzer runs against valid simulator output
- THEN output is byte-identical to the pre-change version (warnings only go to stderr)
- MECHANISM: Warnings go to stderr only; stdout parsing logic unchanged for correct regexes

**Error Handling:**

BC-6: Graceful Degradation
- GIVEN an analyzer encounters output that doesn't contain expected sections
- WHEN parsing completes
- THEN the analyzer MUST NOT crash — it defaults to 0 with a warning, allowing partial results
- MECHANISM: The existing default-to-0 pattern is preserved; warnings are additive

### C) Component Interaction

```
cmd/root.go (stdout producer)
    │
    ├── "=== Simulation Metrics ===" + JSON block
    ├── "=== Anomaly Counters ===" (conditional)
    ├── "=== KV Cache Metrics ===" (conditional)
    └── "=== Trace Summary ===" (conditional)
    │
    ▼
hypotheses/*/analyze.py (consumers — THIS PR)
    │
    ├── JSON block parsing (re.finditer + json.loads)
    ├── Text metric parsing (re.search for specific format strings)
    └── stderr warnings (NEW — when section present but regex misses)
```

No new types, interfaces, or state. Extension friction: N/A (no new abstractions).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Document any mismatches" | Fixes mismatches + adds warnings | ADDITION: Warnings prevent future mismatches from going undetected |
| Lists 6 files to audit | Audits all 9 files | ADDITION: Issue updated to include H5, H10, H13 from the branch |

### E) Review Guide

**The tricky part:** The warning logic must distinguish "section absent because metrics are zero" (legitimate, no warning) from "section present but regex wrong" (bug, warn). The conditional printing in `cmd/root.go:497-502` and `cmd/root.go:540-541` determines which sections appear.

**What to scrutinize:** BC-3 and BC-4 — verify the section-header check correctly prevents false warnings.

**What's safe to skim:** BC-1 and BC-2 are straightforward regex replacements. BC-5 and BC-6 are preserved existing behavior.

**Known debt:** The `cluster["field"] if cluster else 0` pattern (direct dict access vs `.get()`) is inconsistent across files. Not fixing in this PR — it's a style issue, not a correctness issue, and changing it would muddy the diff.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `hypotheses/h5-token-bucket-burst/analyze.py` — fix broken preemption regex, add warnings
- `hypotheses/h12-conservation/analyze.py` — fix broken preemption regex, add warnings
- `hypotheses/h3-signal-freshness/analyze.py` — add warnings
- `hypotheses/h8-kv-pressure/analyze.py` — add warnings
- `hypotheses/h9-prefix-caching/analyze.py` — add warnings
- `hypotheses/h10-tiered-kv/analyze.py` — add warnings (verify existing fix)
- `hypotheses/h13-determinism/analyze.py` — no changes needed (no regex parsing)
- `hypotheses/h14-pathological-templates/analyze.py` — add warnings
- `hypotheses/prefix-affinity/analyze.py` — add warnings

**Key decisions:**
- Warnings use `sys.stderr` to avoid polluting analyzer stdout
- Warning pattern: check for section header presence before warning about missing metrics
- No changes to H13 (it does byte-identical comparison, no regex parsing)

### G) Task Breakdown

---

### Task 1: Fix H5 broken preemption regex

**Contracts Implemented:** BC-1

**Files:**
- Modify: `hypotheses/h5-token-bucket-burst/analyze.py:62`

**Step 1: Fix the regex and convert rate to count**

Context: H5 line 62 uses `r"Preemptions?: (\d+)"` which expects an integer after "Preemption:" or "Preemptions:". The actual format is `"Preemption Rate: %.4f"` (root.go:544). Change to match the float format and convert rate→count (matching H10's corrected pattern from `0cb9768`). Downstream code in H5 uses `preemptions` as an integer for display, so we preserve the count semantic.

In `hypotheses/h5-token-bucket-burst/analyze.py`, replace lines 60-64:
```python
    # Preemption count
    preemptions = 0
    preempt_match = re.search(r"Preemptions?: (\d+)", content)
    if preempt_match:
        preemptions = int(preempt_match.group(1))
```
with:
```python
    # Extract preemption rate from KV cache summary section.
    # Format: "Preemption Rate: 0.1750" (float, printed by cmd/root.go:544).
    # Convert rate to approximate count for downstream display/comparison.
    preemptions = 0
    preempt_match = re.search(r"Preemption Rate: ([0-9.]+)", content)
    if preempt_match:
        preemption_rate = float(preempt_match.group(1))
        completed = cluster["completed_requests"] if cluster else 0
        preemptions = int(round(preemption_rate * completed))
```

**Step 2: Verify the fix is consistent with H10's corrected pattern**

Run: `grep -n "Preemption Rate" hypotheses/h10-tiered-kv/analyze.py`
Expected: Line 48 shows `r"Preemption Rate: ([0-9.]+)"` — same pattern we just used.

**Step 3: Commit**

```bash
git add hypotheses/h5-token-bucket-burst/analyze.py
git commit -m "fix(hypotheses): correct H5 preemption regex to match actual output format (BC-1)

The regex r\"Preemptions?: (\d+)\" never matched the actual format
\"Preemption Rate: %.4f\" from cmd/root.go:544, silently defaulting
to 0. Changed to r\"Preemption Rate: ([0-9.]+)\" (float extraction).

Closes part of #318.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Fix H12 broken preemption regex

**Contracts Implemented:** BC-2

**Files:**
- Modify: `hypotheses/h12-conservation/analyze.py:45`

**Step 1: Fix the regex and convert rate to count**

Context: H12 line 45 has the same broken pattern as H5. Fix identically, using rate→count conversion (matching H10's corrected pattern). H12 displays preemptions as an integer in its conservation table (line 197), so preserving the count semantic is important.

In `hypotheses/h12-conservation/analyze.py`, replace lines 43-47:
```python
    # Preemption count
    preemptions = 0
    preempt_match = re.search(r"Preemptions?: (\d+)", content)
    if preempt_match:
        preemptions = int(preempt_match.group(1))
```
with:
```python
    # Extract preemption rate from KV cache summary section.
    # Format: "Preemption Rate: 0.1750" (float, printed by cmd/root.go:544).
    # Convert rate to approximate count for downstream display/comparison.
    preemptions = 0
    preempt_match = re.search(r"Preemption Rate: ([0-9.]+)", content)
    if preempt_match:
        preemption_rate = float(preempt_match.group(1))
        completed = cluster["completed_requests"] if cluster else 0
        preemptions = int(round(preemption_rate * completed))
```

**Step 2: Verify no other files have the broken pattern**

Run: `grep -rn "Preemptions\?" hypotheses/`
Expected: Zero matches (both H5 and H12 are now fixed).

**Step 3: Commit**

```bash
git add hypotheses/h12-conservation/analyze.py
git commit -m "fix(hypotheses): correct H12 preemption regex to match actual output format (BC-2)

Same fix as H5: r\"Preemptions?: (\d+)\" → r\"Preemption Rate: ([0-9.]+)\".
Closes part of #318.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add defensive stderr warnings to all 8 analyzers

**Contracts Implemented:** BC-3, BC-4, BC-5, BC-6

**Files:**
- Modify: `hypotheses/h3-signal-freshness/analyze.py`
- Modify: `hypotheses/h5-token-bucket-burst/analyze.py`
- Modify: `hypotheses/h8-kv-pressure/analyze.py`
- Modify: `hypotheses/h9-prefix-caching/analyze.py`
- Modify: `hypotheses/h10-tiered-kv/analyze.py`
- Modify: `hypotheses/h12-conservation/analyze.py`
- Modify: `hypotheses/h14-pathological-templates/analyze.py`
- Modify: `hypotheses/prefix-affinity/analyze.py`

**Step 1: Add warning helper and warnings to each file**

Context: Each analyzer has a `parse_output(content)` function that uses the silent-zero default pattern. We add a small helper function at the top of each file and insert warning calls after each regex that could silently fail. The warning only fires when the relevant section header IS present in the output (BC-4: no false warnings when the section is legitimately absent).

Add `import sys` at the top of each file (if not already present), then add the following helper function in each file's `parse_output` function scope, before the first regex:

```python
def _warn_if_section_present(content, section_header, metric_name, filename):
    """Warn on stderr if a section header exists but a metric regex didn't match."""
    if section_header in content:
        print(f"WARNING: '{metric_name}' not found in '{filename}' "
              f"despite '{section_header}' section being present. "
              f"Check regex against cmd/root.go format strings.",
              file=sys.stderr)
```

Then after each silent-zero default block, add a warning call. The pattern transforms from:

```python
preemptions = 0
preempt_match = re.search(r"Preemption Rate: ([0-9.]+)", content)
if preempt_match:
    preemptions = float(preempt_match.group(1))
```

to:

```python
preemptions = 0.0
preempt_match = re.search(r"Preemption Rate: ([0-9.]+)", content)
if preempt_match:
    preemptions = float(preempt_match.group(1))
else:
    _warn_if_section_present(content, "=== KV Cache Metrics ===",
                             "Preemption Rate", output_file)
```

Apply the same pattern for each metric in each file:

**Section-to-metric mapping for warnings:**

| Section Header | Metrics | Files That Parse It |
|---------------|---------|-------------------|
| `=== KV Cache Metrics ===` | Preemption Rate | H5, H8, H9, H10, H12 |
| `=== KV Cache Metrics ===` | Cache Hit Rate | H8, H9, H10, prefix-affinity |
| `=== Anomaly Counters ===` | Rejected Requests | H5, H10, H12, H14 |
| `=== Anomaly Counters ===` | HOL Blocking Events | H3, H14 |
| `=== Anomaly Counters ===` | Priority Inversions | H14 |

For JSON block parsing (`=== Simulation Metrics ===`): warn if the header is present but no JSON block is extracted. This catches catastrophic parse failures.

**Step 2: Verify no warnings fire on existing output**

If any hypothesis experiment has cached output files in its directory, run the analyzer and check stderr:

Run: `ls hypotheses/*/results/ 2>/dev/null | head -5`

If results exist, run a quick check:
```bash
cd hypotheses/h10-tiered-kv && python3 analyze.py 2>warnings.txt && cat warnings.txt && rm warnings.txt; cd ../..
```
Expected: No warnings (existing output should parse cleanly with the corrected regexes).

If no cached results exist, this step verifies via code inspection that BC-4 holds: warnings only fire when the section header IS present.

**Step 3: Verify `sys` import is present in all modified files**

Run: `grep -L "import sys" hypotheses/*/analyze.py | grep -v h13`
Expected: No output (all 8 files import sys). If any file is listed, add `import sys` at the top.

**Step 4: Commit**

```bash
git add hypotheses/h3-signal-freshness/analyze.py \
        hypotheses/h5-token-bucket-burst/analyze.py \
        hypotheses/h8-kv-pressure/analyze.py \
        hypotheses/h9-prefix-caching/analyze.py \
        hypotheses/h10-tiered-kv/analyze.py \
        hypotheses/h12-conservation/analyze.py \
        hypotheses/h14-pathological-templates/analyze.py \
        hypotheses/prefix-affinity/analyze.py
git commit -m "fix(hypotheses): add stderr warnings for silent-zero parser defaults (BC-3, BC-4)

All 8 analyzers now emit a warning to stderr when an output section
header is present but the metric regex within that section fails to
match. This catches parser-output mismatches immediately rather than
silently defaulting to 0 (the bug class that masked H10 preemptions
for 2 rounds).

Warnings are context-aware: no false positives when a section is
legitimately absent (e.g., KV Cache Metrics absent when all values
are zero).

Closes #318.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Cross-reference audit verification

**Contracts Implemented:** Verification of BC-1 through BC-6

**Step 1: Verify all preemption regexes now match the format string**

Run: `grep -n "Preemption" hypotheses/*/analyze.py | grep -v "h13" | grep -v "#"`
Expected: All lines show `r"Preemption Rate: ([0-9.]+)"` or `r"Preemption Rate: ([\d.]+)"` — no `r"Preemptions?"` patterns remain.

**Step 2: Verify all Cache Hit Rate regexes match**

Run: `grep -n "Cache Hit" hypotheses/*/analyze.py | grep -v "#"`
Expected: All lines show patterns matching `"Cache Hit Rate: %.4f"` — variants like `([0-9.]+)`, `([\d.]+)`, `\s+`, `\s*` are all acceptable.

**Step 3: Verify all Rejected Requests regexes match**

Run: `grep -n "Rejected Requests" hypotheses/*/analyze.py | grep -v "#"`
Expected: All lines show `r"Rejected Requests: (\d+)"` matching `"Rejected Requests: %d"`.

**Step 4: Verify all HOL Blocking and Priority Inversion regexes match**

Run: `grep -n "HOL Blocking\|Priority Inversions" hypotheses/*/analyze.py | grep -v "#"`
Expected: Patterns match `"HOL Blocking Events: %d"` and `"Priority Inversions: %d"`.

**Step 5: Verify warning helper is present in all 8 files**

Run: `grep -l "_warn_if_section_present" hypotheses/*/analyze.py | wc -l`
Expected: 8

**Step 6: Final commit (if any verification fixes needed)**

If any issues found in steps 1-5, fix and amend previous commit. Otherwise, no action needed.

---

### H) Test Strategy

| Contract | Task | Test Type | Verification |
|----------|------|-----------|--------------|
| BC-1 | Task 1 | Code inspection | Regex matches `cmd/root.go:544` format |
| BC-2 | Task 2 | Code inspection | Same regex fix as BC-1 |
| BC-3 | Task 3 | Code inspection + runtime | Warning fires when section present but regex misses |
| BC-4 | Task 3 | Code inspection | Warning suppressed when section absent |
| BC-5 | Task 3 | Code inspection | No stdout changes for correct parsers |
| BC-6 | Task 3 | Code inspection | Default-to-0 preserved; warnings are additive |

**No Go tests needed** — this PR modifies only Python scripts. Verification is via code inspection (cross-referencing regexes against format strings) and optional runtime checks against cached experiment output.

**No golden dataset updates** — no simulator output changes.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Warning helper adds enough code to obscure the core fix | Medium | Low | Helper is 4 lines; keeps diff focused | Task 3 |
| False warnings when section headers partially match | Low | Medium | Use exact string match for section headers | Task 3 |
| Changing preemption parsing breaks downstream analysis | Low | Low | Rate→count conversion (matching H10 pattern) preserves integer semantic | Task 1, 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions (helper is minimal, no classes/inheritance)
- [x] No feature creep beyond PR scope (fixes bugs + adds warnings, nothing else)
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes (warnings go to stderr only)
- [x] No hidden global state impact
- [x] Shared test helpers: N/A (Python scripts, no Go test infrastructure)
- [x] CLAUDE.md: No updates needed (no new files/packages, no CLI changes)
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces working code
- [x] Task dependencies correctly ordered (Task 1 → Task 2 → Task 3 → Task 4)
- [x] All contracts mapped to tasks
- [x] Golden dataset: N/A (no output changes)
- [x] Construction site audit: N/A (no struct changes)

**Antipattern rules:**
- [x] R1: No silent data loss — this PR *fixes* silent data loss (the core purpose)
- R2-R17: N/A (Python changes only, no Go code modified)

---

## Appendix: Audit Cross-Reference Table

### Complete regex-to-format-string cross-reference

| Analyzer | Regex Pattern | Format String (`cmd/root.go`) | Match? |
|----------|--------------|------------------------------|--------|
| **H5:62** | `r"Preemptions?: (\d+)"` | `"Preemption Rate: %.4f"` (line 544) | **NO — integer regex vs float output** |
| **H12:45** | `r"Preemptions?: (\d+)"` | `"Preemption Rate: %.4f"` (line 544) | **NO — integer regex vs float output** |
| H8:38 | `r"Preemption Rate: ([\d.]+)"` | `"Preemption Rate: %.4f"` (line 544) | Yes |
| H9:40 | `r"Preemption Rate:\s+([\d.]+)"` | `"Preemption Rate: %.4f"` (line 544) | Yes |
| H10:48 | `r"Preemption Rate: ([0-9.]+)"` | `"Preemption Rate: %.4f"` (line 544) | Yes (patched in 0cb9768) |
| H8:44 | `r"Cache Hit Rate: ([\d.]+)"` | `"Cache Hit Rate: %.4f"` (line 545) | Yes |
| H9:34 | `r"Cache Hit Rate:\s+([\d.]+)"` | `"Cache Hit Rate: %.4f"` (line 545) | Yes |
| H10:64 | `r"Cache Hit Rate: ([0-9.]+)"` | `"Cache Hit Rate: %.4f"` (line 545) | Yes |
| prefix:61 | `r"Cache Hit Rate:\s*([\d.]+)"` | `"Cache Hit Rate: %.4f"` (line 545) | Yes |
| H5:37 | `r"Rejected Requests: (\d+)"` | `"Rejected Requests: %d"` (line 501) | Yes |
| H10:57 | `r"Rejected Requests: (\d+)"` | `"Rejected Requests: %d"` (line 501) | Yes |
| H12:39 | `r"Rejected Requests: (\d+)"` | `"Rejected Requests: %d"` (line 501) | Yes |
| H14:64 | `r"Rejected Requests: (\d+)"` | `"Rejected Requests: %d"` (line 501) | Yes |
| H3:51 | `r"HOL Blocking Events: (\d+)"` | `"HOL Blocking Events: %d"` (line 500) | Yes |
| H14:55 | `r"HOL Blocking Events: (\d+)"` | `"HOL Blocking Events: %d"` (line 500) | Yes |
| H14:59 | `r"Priority Inversions: (\d+)"` | `"Priority Inversions: %d"` (line 499) | Yes |

### JSON field cross-reference

All analyzers extract JSON from the `"=== Simulation Metrics ==="` block. Fields accessed vs `MetricsOutput` struct in `sim/metrics_utils.go`:

| JSON Key | MetricsOutput Field | Accessed By |
|----------|-------------------|------------|
| `instance_id` | `InstanceID` | All 8 analyzers |
| `ttft_mean_ms` | `TTFTMeanMs` | H3, H5, H8, H9, H10, H14, prefix |
| `ttft_p99_ms` | `TTFTP99Ms` | H3, H5, H8, H9, H10 |
| `e2e_mean_ms` | `E2EMeanMs` | H5, H8, H9, H10, H14 |
| `e2e_p99_ms` | `E2EP99Ms` | H3, H5, H8, H10, H14 |
| `responses_per_sec` | `ResponsesPerSec` | H3, H5, H8, H9, H10, H14, prefix |
| `completed_requests` | `CompletedRequests` | H5, H8, H9, H10, H12 |
| `injected_requests` | `InjectedRequests` | H5, H8, H10, H12 |
| `still_queued` | `StillQueued` | H8, H10, H12 |
| `still_running` | `StillRunning` | H8, H10, H12 |
| `total_input_tokens` | `TotalInputTokens` | H9 |
| `preemption_count` | `PreemptionCount` | H8 (per-instance) |

All JSON keys match the `MetricsOutput` struct. No JSON field mismatches found.
