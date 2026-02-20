# Docs: Stale README Output & Scorer Recipe Fix

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix stale documentation in README.md (#276) and extension-recipes.md (#274) to match the current codebase after recent PRs.

**Closes:** #276, #274

**Architecture:** Docs-only changes. Two files modified: `README.md` and `docs/extension-recipes.md`. No code changes, no tests.

**Tech Stack:** Markdown

---

## Task 1: Fix README example JSON output (#276 item 1)

**Files:**
- Modify: `README.md:405-429`

**Step 1: Replace stale JSON example with current output**

The current example output (lines 405-429) contains three removed fields (`sim_start_timestamp`, `sim_end_timestamp`, `simulation_duration_s`) and lacks the three new conservation fields (`still_queued`, `still_running`, `injected_requests`).

Replace the JSON block at lines 405-429 with:

```json
{
  "instance_id": "cluster",
  "completed_requests": 100,
  "still_queued": 0,
  "still_running": 0,
  "injected_requests": 100,
  "total_input_tokens": 53074,
  "total_output_tokens": 51331,
  "vllm_estimated_duration_s": 104.33,
  "responses_per_sec": 0.96,
  "tokens_per_sec": 492.02,
  "e2e_mean_ms": 4541.01,
  "e2e_p90_ms": 7280.50,
  "e2e_p95_ms": 8102.28,
  "e2e_p99_ms": 9760.46,
  "ttft_mean_ms": 25.08,
  "ttft_p90_ms": 31.74,
  "ttft_p95_ms": 34.70,
  "ttft_p99_ms": 37.65,
  "itl_mean_ms": 8.78,
  "itl_p90_ms": 8.73,
  "itl_p95_ms": 8.73,
  "itl_p99_ms": 8.73,
  "scheduling_delay_p99_ms": 11.27
}
```

Key changes: removed `sim_start_timestamp`, `sim_end_timestamp`, `simulation_duration_s`; added `instance_id`, `still_queued`, `still_running`, `injected_requests`; rounded float values for readability.

**Step 2: Update "Key metrics" list below the JSON**

Add a bullet for the new conservation fields after the existing bullets:

```markdown
- **Conservation fields**: `still_queued`, `still_running`, and `injected_requests` verify request conservation (`injected == completed + still_queued + still_running`)
```

**Step 3: Visually verify the diff looks correct**

Run: `git diff README.md`
Expected: Only the JSON block and key metrics list changed, no unrelated modifications.

---

## Task 2: Fix README per-SLO description (#276 item 2)

**Files:**
- Modify: `README.md:30`

**Step 1: Update the per-SLO feature bullet**

Line 30 currently says:
```
- **Per-SLO-class metrics**: breakdown by SLO class with Jain fairness index (computed internally; JSON output planned)
```

Replace with:
```
- **Per-SLO-class metrics**: breakdown by SLO class with Jain fairness index (printed to stdout when multiple SLO classes present)
```

---

## Task 3: Document KV Cache and Per-SLO stdout sections (#276 item 3)

**Files:**
- Modify: `README.md` (Debugging and Observability section, after line ~544)

**Step 1: Add a new subsection before the "Evolutionary Policy Optimization" section**

Insert a new `### Stdout Metrics Sections` subsection after the Anomaly Detection subsection (after line 544, before the `---` separator at line 546). Content:

```markdown

### Stdout Metrics Sections

Beyond the primary JSON metrics, BLIS prints additional sections to stdout when relevant:

**KV Cache Metrics** — printed when any KV cache metric is non-zero:
```
=== KV Cache Metrics ===
Preemption Rate: 0.0000
Cache Hit Rate: 0.0594
KV Thrashing Rate: 0.0000
```

**Per-SLO Metrics** — printed when the workload contains 2+ SLO classes (via `--workload-spec`):
```
=== Per-SLO Metrics ===
  batch:
    TTFT: mean=45.20 p99=231.33 (n=350)
    E2E:  mean=3200.50 p99=12351.62 (n=350)
  realtime:
    TTFT: mean=42.10 p99=138.01 (n=150)
    E2E:  mean=3083.47 p99=12813.41 (n=150)
```

These sections are deterministic (same seed = same output) and appear after the JSON metrics block.
```

---

## Task 4: Fix extension-recipes.md scorer recipe (#274)

**Files:**
- Modify: `docs/extension-recipes.md:34-45`

**Step 1: Update the scorer recipe section**

Replace lines 34-45 (the current "Adding New Scorers" section body) with the corrected version. Three fixes:

1. `scorerFunc` signature: `[]RoutingSnapshot` → `(*Request, []RoutingSnapshot)`
2. Factory name: `newScorer` → `newScorerWithObserver`
3. Touch points: add note about stateful scorers needing a separate file

Replace:
```markdown
1. **Implement the scorer function** in `sim/routing_scorers.go` — a `scorerFunc` that takes `[]RoutingSnapshot` and returns `map[string]float64` with scores in [0,1] per instance
2. **Register the scorer** in the same file: add to `validScorerNames` map + `newScorer` factory switch
3. **Add behavioral tests** in `sim/routing_scorers_test.go` — monotonicity, boundary values, INV-1/INV-2 conformance
4. Extension friction: **2 touch points in 1 file**

Examples:
- See `scoreLoadBalance` in `sim/routing_scorers.go` for a simple stateless scorer
- See `scoreQueueDepth` for a scorer with edge case handling (uniform load)
```

With:
```markdown
1. **Implement the scorer function** in `sim/routing_scorers.go` (stateless) or a new file (stateful) — a `scorerFunc` that takes `(*Request, []RoutingSnapshot)` and returns `map[string]float64` with scores in [0,1] per instance. Stateful scorers also return an `observerFunc` called after each routing decision.
2. **Register the scorer** in `sim/routing_scorers.go`: add to `validScorerNames` map + `newScorerWithObserver` factory switch
3. **Add behavioral tests** — monotonicity, boundary values, INV-1/INV-2 conformance
4. Extension friction: **2 touch points** (implementation + registration)

Examples:
- See `scoreLoadBalance` in `sim/routing_scorers.go` for a simple stateless scorer
- See `scoreQueueDepth` for a scorer with edge case handling (uniform load)
- See `newPrefixAffinityScorer` in `sim/routing_prefix_scorer.go` for a stateful scorer with observer and router-side cache
```

**Step 2: Verify the diff looks correct**

Run: `git diff docs/extension-recipes.md`
Expected: Only the scorer recipe section changed, no unrelated modifications.

---

## Task 5: Commit and push

**Step 1: Run tests to confirm no breakage**

Run: `go test ./...`
Expected: All tests pass (docs-only changes should not affect tests).

**Step 2: Commit**

```bash
git add README.md docs/extension-recipes.md docs/plans/2026-02-20-docs-stale-output-and-scorer-recipe.md
git commit -m "docs: fix stale README output and scorer recipe (#276, #274)"
```
