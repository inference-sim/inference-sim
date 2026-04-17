# vLLM Data-Parallel Router Parity Implementation

**Date:** 2026-04-17
**Issue:** [#1075](https://github.com/inference-sim/inference-sim/issues/1075)
**Status:** Design Approved

## Problem Statement

BLIS cannot currently simulate vLLM's default multi-instance routing in data-parallel mode. vLLM uses a greedy weighted least-loaded formula with hardcoded weights: `score = waiting × 4 + running`, where `waiting` maps to queue depth and `running` maps to active batch size.

BLIS's existing `queue-depth` and `running-requests` scorers cannot replicate this behavior because they normalize each dimension independently before composition, whereas vLLM computes a single raw weighted sum then selects the minimum.

## Goals

1. **Exact vLLM parity**: Implement routing that produces identical instance selection to vLLM's `DPLBAsyncMPClient.get_core_engine_for_request()` (vllm/v1/engine/core_client.py:1206-1229)
2. **Composable integration**: Work within BLIS's existing scorer framework
3. **Minimal complexity**: Single atomic scorer, no new routing policy types

## Non-Goals

- Configurable weights (vLLM hardcodes 4:1 ratio in production)
- Power-of-Two-Choices algorithm (vLLM TODO for future work)
- Hybrid composition with other scorers (not recommended; use pure `vllm-dp:1`)

## Design

### Core Algorithm

**vLLM source (core_client.py:1219):**
```python
score = waiting * 4 + running
if score < min_score:
    min_score = score
    eng_index = idx
```

**BLIS implementation:**
```go
func scoreVLLMDP(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
    // Step 1: Compute raw scores using vLLM formula
    rawScores := make(map[string]int, len(snapshots))
    minRaw, maxRaw := math.MaxInt, 0

    for _, snap := range snapshots {
        raw := snap.QueueDepth*4 + snap.BatchSize
        rawScores[snap.ID] = raw
        if raw < minRaw { minRaw = raw }
        if raw > maxRaw { maxRaw = raw }
    }

    // Step 2: Inverted min-max normalization
    // Lowest load (minRaw) → 1.0 (highest score, preferred)
    // Highest load (maxRaw) → 0.0 (lowest score, avoided)
    scores := make(map[string]float64, len(snapshots))
    if maxRaw == minRaw {
        // All equal → all score 1.0
        for _, snap := range snapshots {
            scores[snap.ID] = 1.0
        }
    } else {
        for _, snap := range snapshots {
            raw := rawScores[snap.ID]
            scores[snap.ID] = float64(maxRaw-raw) / float64(maxRaw-minRaw)
        }
    }

    return scores
}
```

**Key insight:** vLLM selects **argmin(raw score)**, BLIS selects **argmax(normalized score)**. Inverted normalization preserves vLLM's preference for lowest-loaded instances.

### Signal Mapping

| vLLM Term | vLLM Source | BLIS Field | BLIS Source |
|-----------|-------------|------------|-------------|
| `waiting` | `scheduler_stats.num_waiting_reqs` | `QueueDepth` | `RoutingSnapshot.QueueDepth` |
| `running` | `scheduler_stats.num_running_reqs` | `BatchSize` | `RoutingSnapshot.BatchSize` |

Both fields are **Periodic** signals (Immediate when `--snapshot-refresh-interval=0`). Signal freshness follows INV-7.

### Integration Points

**File:** `sim/routing_scorers.go`

1. Add `"vllm-dp": true` to `validScorerNames` map (line 39)
2. Add case `"vllm-dp": return scoreVLLMDP, nil` in `newScorerWithObserver()` (line 136)
3. Implement `scoreVLLMDP()` function

**No changes required:**
- `sim/routing.go` — uses existing `WeightedScoring` policy
- `cmd/` — uses existing `--routing-scorers` flag
- Other scorers — remain unchanged

### Edge Cases

| Scenario | Behavior | Rationale |
|----------|----------|-----------|
| All instances equal load | All score 1.0 | No differentiation needed |
| Single instance | Scores 1.0 | Trivial case, always routable |
| All-zero loads | All score 1.0 | Empty cluster, all equal |
| QueueDepth=0, BatchSize varies | Prefers lower BatchSize | Formula: 0×4 + BatchSize |
| QueueDepth varies, BatchSize=0 | Prefers lower QueueDepth (4× weight) | Formula: QueueDepth×4 + 0 |

### CLI Usage

**Oracle mode (default, immediate signals):**
```bash
./blis run --model qwen/qwen3-14b --routing-scorers "vllm-dp:1"
```

**Realistic vLLM parity (100ms coordinator staleness):**
```bash
./blis run --model qwen/qwen3-14b --routing-scorers "vllm-dp:1" \
  --snapshot-refresh-interval 100000
```

vLLM's coordinator publishes instance stats every 100ms by default (`min_stats_update_interval_ms=100`). For realistic parity studies, use `--snapshot-refresh-interval 100000` (100ms in microseconds). The default `0` represents oracle routing with no signal staleness.

**Hybrid composition (not recommended for vLLM parity):**
```bash
# Experimental: blend vLLM load-balancing with prefix-cache awareness
./blis run --routing-scorers "vllm-dp:0.5,precise-prefix-cache:1"
```

Hybrid usage is discouraged because:
- `vllm-dp` is pre-normalized; composing with other scorers double-normalizes dimensions
- vLLM does not use prefix-cache signals in its default router
- Results won't match vLLM behavior

## Testing

### Unit Tests

**File:** `sim/routing_scorers_test.go`

**Test 1: Basic formula verification**
```go
func TestScoreVLLMDP_BasicFormula(t *testing.T) {
    snapshots := []RoutingSnapshot{
        {ID: "a", QueueDepth: 10, BatchSize: 5},  // 10×4 + 5 = 45
        {ID: "b", QueueDepth: 5, BatchSize: 10},  // 5×4 + 10 = 30
        {ID: "c", QueueDepth: 2, BatchSize: 2},   // 2×4 + 2 = 10 (min)
    }
    scores := scoreVLLMDP(nil, snapshots)

    // c has lowest raw score (10) → should score 1.0
    assert.Equal(t, 1.0, scores["c"])
    // a has highest raw score (45) → should score 0.0
    assert.Equal(t, 0.0, scores["a"])
    // b is midpoint: (45-30)/(45-10) = 15/35 ≈ 0.43
    assert.InDelta(t, 0.428, scores["b"], 0.01)
}
```

**Test 2: All equal → all score 1.0**
```go
func TestScoreVLLMDP_AllEqual(t *testing.T) {
    snapshots := []RoutingSnapshot{
        {ID: "a", QueueDepth: 5, BatchSize: 3},  // 5×4 + 3 = 23
        {ID: "b", QueueDepth: 5, BatchSize: 3},  // 5×4 + 3 = 23
    }
    scores := scoreVLLMDP(nil, snapshots)
    assert.Equal(t, 1.0, scores["a"])
    assert.Equal(t, 1.0, scores["b"])
}
```

**Test 3: Monotonicity and boundaries**
```go
func TestScoreVLLMDP_MonotonicityAndBoundaries(t *testing.T) {
    snapshots := []RoutingSnapshot{
        {ID: "a", QueueDepth: 0, BatchSize: 0},   // 0 (min)
        {ID: "b", QueueDepth: 3, BatchSize: 2},   // 14
        {ID: "c", QueueDepth: 5, BatchSize: 10},  // 30 (max)
    }
    scores := scoreVLLMDP(nil, snapshots)

    // Boundaries
    assert.Equal(t, 1.0, scores["a"], "min load should score 1.0")
    assert.Equal(t, 0.0, scores["c"], "max load should score 0.0")

    // Monotonicity: lower load → higher score
    assert.Greater(t, scores["a"], scores["b"])
    assert.Greater(t, scores["b"], scores["c"])

    // No NaN/Inf
    for _, score := range scores {
        assert.False(t, math.IsNaN(score))
        assert.False(t, math.IsInf(score, 0))
    }
}
```

**Test 4: Single instance**
```go
func TestScoreVLLMDP_SingleInstance(t *testing.T) {
    snapshots := []RoutingSnapshot{
        {ID: "a", QueueDepth: 42, BatchSize: 17},
    }
    scores := scoreVLLMDP(nil, snapshots)
    assert.Equal(t, 1.0, scores["a"])
}
```

### Behavioral Contracts

The `vllm-dp` scorer must satisfy:

- **BC-VLLM-1 (4:1 weighting):** For any two instances A, B where `QD_A = QD_B + 1` and `BS_A = BS_B - 4`, their raw scores must be equal (the 4:1 ratio means +1 queue depth ≡ -4 batch size in routing preference)
- **BC-VLLM-2 (inversion):** Instance with lowest raw score must score 1.0; instance with highest raw score must score 0.0
- **BC-VLLM-3 (monotonicity):** For raw scores R_A < R_B, normalized scores S_A > S_B
- **BC-17-9 (no NaN/Inf):** All scores must be finite (inherited from scorer contract)

### Integration Testing

Run existing BLIS integration tests to ensure no regressions:
```bash
go test ./sim/... -v
go test ./sim/cluster/... -v
```

Expected: All existing tests pass with no changes.

### Parity Validation

**Manual validation:** Create a 3-instance workload with varied load distributions, run both vLLM and BLIS, compare routing decisions:

```bash
# BLIS
./blis run --model qwen/qwen3-14b --num-instances 3 \
  --routing-scorers "vllm-dp:1" --trace-output validation

# vLLM (external validation script)
python validate_vllm_parity.py --trace validation.csv
```

**Expected outcome:** 100% routing decision agreement for all requests.

## Alternatives Considered

### Alternative 1: Configurable Weights

**Design:** Allow `--routing-scorers "vllm-dp:4:1"` to specify waiting/running weights.

**Rejected because:**
- vLLM hardcodes 4:1 in production code (not configurable)
- vLLM's TODO mentions switching to Power-of-Two-Choices, not tuning weights
- Adds parser complexity for speculative benefit

### Alternative 2: Raw-Sum Composition Mode

**Design:** New `--routing-policy weighted-raw` that sums scorer outputs before normalization, allowing `"queue-depth:4,running-requests:1"` to compose into vLLM behavior.

**Rejected because:**
- Over-engineered for single use case (100+ lines vs. 30 lines)
- Introduces new routing policy concept (cognitive overhead)
- Doesn't generalize well (most scorers expect independent normalization)

### Alternative 3: Reuse Existing Scorers

**Design:** Compose `queue-depth` and `running-requests` scorers with 4:1 weights.

**Rejected because:**
- Produces **different rankings** due to independent normalization
- Example: Instances with loads [10q+5b, 5q+10b, 2q+2b] have raw scores [45, 30, 10] but independent min-max on each dimension produces incorrect composite rankings
- Issue #1075 explicitly notes this limitation

## Known Limitations

### Speculative Increment Not Modeled

**What vLLM does:** After routing to an engine, vLLM's `DPLBAsyncMPClient` speculatively increments the local cached waiting count:

```python
# vllm/v1/engine/core_client.py:1223-1225
# Increment local waiting count for better balancing between stats
# updates from the coordinator (which happen every 100ms).
current_counts[eng_index][0] += self.client_count
```

**Why:** vLLM's coordinator publishes fresh stats every 100ms. Between updates, multiple requests arrive and read the same stale `lb_engines` cache. Without the increment, they all pile onto the same "least loaded" instance. The increment prevents this — after routing to engine X, that engine's cached waiting count increases, making it look more loaded to the next request.

**BLIS behavior:** BLIS does not replicate this because BLIS's DES is sequential — requests are routed one at a time, so the concurrent-routing scenario does not arise. In periodic-snapshot mode (`--snapshot-refresh-interval > 0`), BLIS will concentrate more traffic per snapshot window than real vLLM would.

**Impact:** Oracle mode (`--snapshot-refresh-interval 0`) is unaffected and provides the cleanest comparison baseline for algorithmic studies. Staleness-matched mode (100ms interval) will show more traffic concentration than real vLLM.

### Tie-Breaking Differs

**What vLLM does:** When scores are tied, vLLM's `if score < min_score` (strict less-than) means the first engine encountered in the scan wins. The scan starts from `eng_start_index`, which is derived from `client_index`, so multiple co-located API servers start at different offsets, spreading ties round-robin.

**BLIS behavior:** BLIS uses random tie-breaking in `WeightedScoring` (introduced in #574 to fix pile-ons). When `vllm-dp` gives all instances equal scores (e.g., all zero load), BLIS picks randomly; vLLM picks deterministically.

**Impact:** Under perfectly balanced load, tie-breaking determines all routing decisions. For reproducibility studies comparing BLIS vs. vLLM routing traces, tie-breaking divergence will cause systematic differences. This is a minor issue for most workloads since ties are rare under realistic load distributions.

## Documentation Updates

### User-Facing

**File:** `docs/guide/routing.md` (if exists, else create)

**Section:** "vLLM Data-Parallel Parity"
```markdown
## vLLM Data-Parallel Routing

BLIS supports vLLM's internal load-balancing algorithm via the `vllm-dp` scorer:

```bash
./blis run --model qwen/qwen3-14b --routing-scorers "vllm-dp:1"
```

This replicates vLLM's `DPLBAsyncMPClient` behavior: selecting the instance with
the lowest `waiting × 4 + running` score. Use this for vLLM parity studies or
when comparing BLIS against production vLLM deployments.

**Note:** Do not compose `vllm-dp` with other scorers unless comparing hybrid
strategies. For vLLM parity, use `vllm-dp:1` alone.
```

### Code Comments

Add signal freshness documentation to `scoreVLLMDP()`:
```go
// scoreVLLMDP computes per-instance scores using vLLM's data-parallel formula.
// Formula: raw = QueueDepth × 4 + BatchSize, then inverted min-max normalization.
// Matches vLLM's DPLBAsyncMPClient.get_core_engine_for_request() (core_client.py:1219).
//
// Signal freshness (R17, INV-7):
//   Reads: QueueDepth (Periodic when interval>0, else Immediate)
//          BatchSize (Periodic when interval>0, else Immediate)
```

## Implementation Plan

1. **Add scorer registration** (~5 min)
   - Update `validScorerNames` map
   - Add case in `newScorerWithObserver()`

2. **Implement `scoreVLLMDP()`** (~30 min)
   - Write function (~30 lines)
   - Add signal freshness comment

3. **Write unit tests** (~45 min)
   - 4 test functions in `routing_scorers_test.go`
   - Verify behavioral contracts

4. **Run existing tests** (~5 min)
   - `go test ./sim/... -v`
   - Ensure no regressions

5. **Update documentation** (~15 min)
   - Add routing guide section
   - Update CLI help text (if needed)

**Total estimated time:** ~2 hours

## Success Criteria

- [ ] `vllm-dp` scorer passes all 4 unit tests
- [ ] All existing BLIS tests pass (no regressions)
- [ ] CLI accepts `--routing-scorers "vllm-dp:1"` without errors
- [ ] Manual parity validation shows 100% routing agreement with vLLM
- [ ] Code review approved

## References

- **Issue:** [#1075](https://github.com/inference-sim/inference-sim/issues/1075)
- **vLLM source:** `vllm/v1/engine/core_client.py:1206-1229` (DPLBAsyncMPClient.get_core_engine_for_request)
- **vLLM docs:** `docs/serving/data_parallel_deployment.md`
- **BLIS scorer framework:** `sim/routing_scorers.go`
- **BLIS invariants:** `docs/contributing/standards/invariants.md` (INV-7: Signal freshness)
