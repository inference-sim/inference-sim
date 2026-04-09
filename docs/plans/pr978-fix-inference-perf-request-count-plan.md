# Fix inference_perf Request Count Bug — Exact Generation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `math.Ceil` rounding bug causing request count inflation in inference_perf workload expansion to match real inference-perf exact counts

**Architecture:** Replace ceiling rounding with floor-based fair distribution (base + remainder allocation) in two code paths: (1) single-stage non-multi-turn `requestsPerClient` calculation, (2) multi-turn `MaxRounds` calculation. Distribute total requests evenly across N clients ensuring sum equals exactly `int(rate × duration)`.

**Tech Stack:** Go 1.22+, table-driven tests, behavioral contracts

**Source:** GitHub Issue #978
**Closes:** #978

---

## Scope Check

✅ Single module (`sim/workload`) - no subsystem split needed
✅ Two affected code paths in one file (`inference_perf.go`)
✅ Plan produces working, testable fix independently

---

## File Structure

**Modified:**
- `sim/workload/inference_perf.go`: Add `distributeRequestsEvenly` helper, fix line 148 (single-stage non-multi-turn), fix line 295 (multi-turn `computeReasoningSpec`)
- `sim/workload/inference_perf_test.go`: Add behavioral tests for exact counts

**No new files created** - pure bug fix

---

## Behavioral Contracts

**BC-1:** Single-stage non-multi-turn generates exactly `int(rate × duration)` total requests (not ceiling-inflated)

**BC-2:** Multi-turn single-stage generates exactly `int(rate × duration)` total requests across all sessions

**BC-3:** Multi-stage multi-turn generates exactly `int(rate × duration)` per stage

**BC-4:** Per-client/per-session counts differ by at most 1 (fair distribution)

**BC-5:** Determinism preserved (same seed → byte-identical output)

---

### Task 1: Add Fair Distribution Helper Function

**Files:**
- Modify: `sim/workload/inference_perf.go` (add after line 317, before `ExpandInferencePerfSpec`)
- Test: `sim/workload/inference_perf_test.go` (add after line 1742)

- [ ] **Step 1: Write the failing test**

Add to `sim/workload/inference_perf_test.go` after line 1742:

```go
func TestDistributeRequestsEvenly_ExactTotal(t *testing.T) {
	// BC-4: Fair distribution with max difference ≤ 1, sum equals total
	tests := []struct {
		total int
		n     int
		want  []int
	}{
		{total: 100, n: 3, want: []int{34, 33, 33}}, // 34+33+33=100
		{total: 10, n: 10, want: []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
		{total: 44, n: 7, want: []int{7, 7, 7, 7, 6, 6, 6}}, // 4×7 + 3×6 = 44
		{total: 3000, n: 44, want: nil},                     // Just verify sum=3000
		{total: 6000, n: 44, want: nil},                     // Issue #978 example
		{total: 0, n: 5, want: []int{0, 0, 0, 0, 0}},        // Edge: zero requests
		{total: 7, n: 1, want: []int{7}},                    // Edge: single client
	}
	for _, tt := range tests {
		dist := distributeRequestsEvenly(tt.total, tt.n)
		sum := 0
		for _, count := range dist {
			sum += count
		}
		if sum != tt.total {
			t.Errorf("distributeRequestsEvenly(%d, %d): sum=%d, want %d",
				tt.total, tt.n, sum, tt.total)
		}
		if tt.want != nil {
			if !slicesEqual(dist, tt.want) {
				t.Errorf("distributeRequestsEvenly(%d, %d) = %v, want %v",
					tt.total, tt.n, dist, tt.want)
			}
		}
		// Check fairness: max difference ≤ 1
		if len(dist) > 1 {
			min, max := dist[0], dist[0]
			for _, v := range dist {
				if v < min {
					min = v
				}
				if v > max {
					max = v
				}
			}
			if max-min > 1 {
				t.Errorf("unfair distribution: max-min=%d > 1", max-min)
			}
		}
	}
}

func slicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test ./sim/workload/... -run TestDistributeRequestsEvenly -v
```

**Expected:** FAIL with "undefined: distributeRequestsEvenly"

- [ ] **Step 3: Implement distributeRequestsEvenly helper**

Add to `sim/workload/inference_perf.go` after line 317 (after `constantDist` function):

```go
// distributeRequestsEvenly distributes totalRequests across n clients,
// ensuring the sum equals exactly totalRequests (no ceiling inflation).
// Returns per-client counts where max difference is 1.
//
// Algorithm: base = total/n (floor), remainder = total%n.
// First 'remainder' clients get base+1, others get base.
//
// Example: distributeRequestsEvenly(10, 3) → [4, 3, 3] (sum=10)
func distributeRequestsEvenly(totalRequests, n int) []int {
	if n <= 0 {
		panic("distributeRequestsEvenly: n must be positive")
	}
	if totalRequests < 0 {
		panic("distributeRequestsEvenly: totalRequests must be non-negative")
	}
	base := totalRequests / n
	remainder := totalRequests % n
	dist := make([]int, n)
	for i := 0; i < n; i++ {
		dist[i] = base
		if i < remainder {
			dist[i]++
		}
	}
	return dist
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
go test ./sim/workload/... -run TestDistributeRequestsEvenly -v
```

**Expected:** PASS (7 test cases, all passing)

- [ ] **Step 5: Run lint**

```bash
golangci-lint run ./sim/workload/...
```

**Expected:** No issues

- [ ] **Step 6: Commit**

```bash
git add sim/workload/inference_perf.go sim/workload/inference_perf_test.go
git commit -m "test(workload): add fair distribution helper with exact-count tests

BC-4: Fair distribution with max difference ≤ 1

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Fix Single-Stage Non-Multi-Turn Request Count

**Files:**
- Modify: `sim/workload/inference_perf.go` (lines 143-194)
- Test: `sim/workload/inference_perf_test.go` (add after TestDistributeRequestsEvenly tests)

- [ ] **Step 1: Write the failing test**

Add to `sim/workload/inference_perf_test.go`:

```go
func TestExpandInferencePerfSpec_SingleStageNonMultiTurn_ExactRequestCount(t *testing.T) {
	// BC-1: exact total request count (no ceiling inflation)
	tests := []struct {
		rate       float64
		duration   int64
		numPrompts int
		numUsers   int
		wantTotal  int
	}{
		{10.0, 60, 3, 2, 600},    // 10*60=600, 6 clients
		{5.0, 600, 11, 4, 3000},  // Issue #978 stage 0 example
		{10.0, 600, 11, 4, 6000}, // Issue #978 stage 1 example
		{7.5, 100, 2, 3, 750},    // Non-integer per-client
		{20.0, 30, 7, 3, 600},    // 600/21 = 28.57 → was 630 with ceil, now 600
	}
	for _, tt := range tests {
		spec := &InferencePerfSpec{
			Stages: []StageSpec{{Rate: tt.rate, Duration: tt.duration}},
			SharedPrefix: &SharedPrefixSpec{
				NumUniqueSystemPrompts:  tt.numPrompts,
				NumUsersPerSystemPrompt: tt.numUsers,
				SystemPromptLen:         10,
				QuestionLen:             10,
				OutputLen:               10,
				EnableMultiTurnChat:     false, // non-multi-turn path
			},
		}
		expanded, err := ExpandInferencePerfSpec(spec, 42)
		if err != nil {
			t.Fatalf("expansion error: %v", err)
		}
		horizon := tt.duration * 2_000_000 // 2× duration (sampler-limited)
		requests, err := GenerateRequests(expanded, horizon, 0)
		if err != nil {
			t.Fatalf("generation error: %v", err)
		}
		if len(requests) != tt.wantTotal {
			t.Errorf("rate=%.1f dur=%d clients=%dx%d: got %d requests, want %d (exact, no ceiling)",
				tt.rate, tt.duration, tt.numPrompts, tt.numUsers,
				len(requests), tt.wantTotal)
		}
	}
}

func TestExpandInferencePerfSpec_SingleStageNonMultiTurn_FairDistribution(t *testing.T) {
	// BC-4: Per-client counts differ by at most 1
	spec := &InferencePerfSpec{
		Stages: []StageSpec{{Rate: 10.0, Duration: 60}}, // 600 requests / 7 clients
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  7,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
			EnableMultiTurnChat:     false,
		},
	}
	expanded, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	horizon := int64(120_000_000) // 120 seconds
	requests, err := GenerateRequests(expanded, horizon, 0)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}
	// Count per client
	perClient := make(map[string]int)
	for _, req := range requests {
		perClient[req.ClientID]++
	}
	// Find min and max counts
	min, max := 999999, 0
	for _, count := range perClient {
		if count < min {
			min = count
		}
		if count > max {
			max = count
		}
	}
	if max-min > 1 {
		t.Errorf("unfair distribution: max=%d min=%d diff=%d (want diff ≤ 1)",
			max, min, max-min)
	}
	// Verify total is exact
	if len(requests) != 600 {
		t.Errorf("total requests = %d, want 600 (exact)", len(requests))
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
go test ./sim/workload/... -run "TestExpandInferencePerfSpec_SingleStageNonMultiTurn" -v
```

**Expected:** FAIL (tests show 630, 3036, 6028 instead of 600, 3000, 6000)

- [ ] **Step 3: Fix single-stage non-multi-turn path**

In `sim/workload/inference_perf.go`, replace lines 143-194 with:

```go
		} else {
			// Single-request (language) workload: use NormalizedExponentialSampler.
			// Distribute total requests evenly across clients using fair allocation
			// (floor-with-remainder) to match real inference-perf exact counts.
			totalRequests := int(stage.Rate * float64(stage.Duration))
			perClientDist := distributeRequestsEvenly(totalRequests, numClientsPerStage)
			durationUs := stage.Duration * 1_000_000 // seconds to microseconds

			// Defensive: prevent integer overflow in seed calculation (cast before multiply)
			totalClients := int64(sp.NumUniqueSystemPrompts) * int64(sp.NumUsersPerSystemPrompt)
			if totalClients > 1_000_000 {
				return nil, fmt.Errorf("total client count %d exceeds safety limit (1M)", totalClients)
			}

			clientIdx := 0
			for p := 0; p < sp.NumUniqueSystemPrompts; p++ {
				prefixGroup := fmt.Sprintf("prompt-%d", p)
				for u := 0; u < sp.NumUsersPerSystemPrompt; u++ {
					clientID := fmt.Sprintf("prompt-%d-user-%d", p, u)
					requestsPerClient := int64(perClientDist[clientIdx])
					clientIdx++

					// Validate sampler parameters before construction (prevent panic on user input)
					if requestsPerClient <= 0 {
						return nil, fmt.Errorf("inference_perf: requestsPerClient must be positive, got %d", requestsPerClient)
					}
					if requestsPerClient > 10_000_000 {
						return nil, fmt.Errorf("inference_perf: requestsPerClient %d exceeds safety limit (10M); reduce rate, duration, or increase clients", requestsPerClient)
					}
					if durationUs < requestsPerClient {
						return nil, fmt.Errorf("inference_perf: durationUs (%d) < requestsPerClient (%d) produces degenerate distribution", durationUs, requestsPerClient)
					}

					// Create factory closure that captures requestsPerClient and durationUs.
					// Each GenerateRequests call will invoke this factory with a sub-RNG,
					// producing a fresh sampler instance (workload reusability).
					factory := func(rng *rand.Rand) ArrivalSampler {
						return NewNormalizedExponentialSampler(rng, requestsPerClient, durationUs)
					}

					clients = append(clients, ClientSpec{
						ID:                   clientID,
						TenantID:             prefixGroup,
						SLOClass:             "standard",
						RateFraction:         rateFraction,
						Arrival:              ArrivalSpec{Process: "poisson"}, // Fallback for diagnostics/serialization
						CustomSamplerFactory: factory,
						InputDist:            inputDist,
						OutputDist:           outputDist,
						PrefixGroup:          prefixGroup,
						PrefixLength:         sp.SystemPromptLen,
						Reasoning:            reasoning,
					})
				}
			}
		}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
go test ./sim/workload/... -run "TestExpandInferencePerfSpec_SingleStageNonMultiTurn" -v
```

**Expected:** PASS (5 test cases, all showing exact counts)

- [ ] **Step 5: Run full workload test suite**

```bash
go test ./sim/workload/... -v
```

**Expected:** PASS (all existing tests still pass)

- [ ] **Step 6: Run lint**

```bash
golangci-lint run ./sim/workload/...
```

**Expected:** No issues

- [ ] **Step 7: Commit**

```bash
git add sim/workload/inference_perf.go sim/workload/inference_perf_test.go
git commit -m "fix(workload): use exact distribution for single-stage non-multi-turn

Replaces math.Ceil (line 148) with floor-based fair distribution.

BC-1: Total requests now equals exactly int(rate × duration)

Fixes #978 (partial: single-stage non-multi-turn path)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Fix Multi-Turn MaxRounds Calculation

**Files:**
- Modify: `sim/workload/inference_perf.go` (lines 114-140, 225-248, 281-309)
- Test: `sim/workload/inference_perf_test.go` (add after previous tests)

- [ ] **Step 1: Write the failing test**

Add to `sim/workload/inference_perf_test.go`:

```go
func TestExpandInferencePerfSpec_SingleStageMultiTurn_ExactRequestCount(t *testing.T) {
	// BC-2: Single-stage multi-turn generates exact request count
	spec := &InferencePerfSpec{
		Stages: []StageSpec{{Rate: 5.0, Duration: 600}},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  11,
			NumUsersPerSystemPrompt: 4,
			SystemPromptLen:         100,
			QuestionLen:             200,
			OutputLen:               50,
			EnableMultiTurnChat:     true,
		},
	}
	expanded, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	horizon := int64(600_000_000)
	requests, err := GenerateRequests(expanded, horizon, 0)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}
	// Should get exactly 3000 requests (not 3036)
	if len(requests) != 3000 {
		t.Errorf("multi-turn request count = %d, want 3000 (exact, issue #978 example)",
			len(requests))
	}
}

func TestExpandInferencePerfSpec_MultiStageMultiTurn_ExactRequestCountPerStage(t *testing.T) {
	// BC-3: Multi-stage multi-turn generates exact counts per stage
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 5.0, Duration: 600},
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  11,
			NumUsersPerSystemPrompt: 4,
			SystemPromptLen:         100,
			QuestionLen:             200,
			OutputLen:               50,
			EnableMultiTurnChat:     true,
		},
	}
	expanded, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	horizon := int64(1_200_000_000) // 1200 seconds
	requests, err := GenerateRequests(expanded, horizon, 0)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}

	// Count per stage
	boundary := int64(600_000_000)
	var stage0Count, stage1Count int
	for _, req := range requests {
		if req.ArrivalTime < boundary {
			stage0Count++
		} else {
			stage1Count++
		}
	}

	// Stage 0: exactly 3000 (not 3036)
	if stage0Count != 3000 {
		t.Errorf("stage 0 count = %d, want 3000 (exact)", stage0Count)
	}
	// Stage 1: exactly 6000 (not 6028)
	if stage1Count != 6000 {
		t.Errorf("stage 1 count = %d, want 6000 (exact)", stage1Count)
	}
	// Total: exactly 9000 (not 9064)
	if len(requests) != 9000 {
		t.Errorf("total count = %d, want 9000 (exact, issue #978 full example)",
			len(requests))
	}
}

func TestExpandInferencePerfSpec_MultiTurn_PerSessionFairness(t *testing.T) {
	// BC-4: Multi-turn sessions have MaxRounds differing by at most 1
	spec := &InferencePerfSpec{
		Stages: []StageSpec{{Rate: 10.0, Duration: 60}}, // 600 requests / 7 sessions
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  7,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
			EnableMultiTurnChat:     true,
		},
	}
	expanded, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	// Check that MaxRounds varies by at most 1 across clients
	min, max := 999999, 0
	for _, client := range expanded.Clients {
		rounds := client.Reasoning.MultiTurn.MaxRounds
		if rounds < min {
			min = rounds
		}
		if rounds > max {
			max = rounds
		}
	}
	if max-min > 1 {
		t.Errorf("unfair MaxRounds distribution: max=%d min=%d diff=%d (want diff ≤ 1)",
			max, min, max-min)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
go test ./sim/workload/... -run "MultiTurn_ExactRequestCount|MultiTurn_PerSessionFairness" -v
```

**Expected:** FAIL (tests show 3036, 6028, 9064 instead of 3000, 6000, 9000; MaxRounds is uniform instead of distributed)

- [ ] **Step 3: Update computeReasoningSpec signature**

In `sim/workload/inference_perf.go`, replace lines 281-309 with:

```go
// computeReasoningSpec builds a ReasoningSpec for inference-perf multi-turn mode.
// It derives MaxRounds and ThinkTimeUs from stage parameters to match inference-perf's
// round-robin cycling behavior: N sessions cycle at rate R over duration D seconds.
//
// MaxRounds = roundsForThisSession (from fair distribution of total requests)
// ThinkTimeUs = floor((N / R) * 1e6): inter-round delay in microseconds
//
// ContextGrowth is intentionally empty (fixed-length inputs per round) because
// real inference-perf sends constant input tokens per request — the chat template
// is applied but context is NOT accumulated across turns (H30 finding).
//
// Note: ThinkTimeUs does not account for the 1µs/token output completion heuristic
// in GenerateReasoningRequests. This is negligible for typical parameterizations
// (e.g., OutputLen=248 adds 248µs to a ThinkTimeUs of 600,000µs = 0.04% error).
func computeReasoningSpec(stageRate float64, stageDurationSec int64, numSessions int, roundsForThisSession int) *ReasoningSpec {
	thinkTimeUs := int64(float64(numSessions) / stageRate * 1e6)
	return &ReasoningSpec{
		ReasonRatioDist: DistSpec{
			Type:   "constant",
			Params: map[string]float64{"value": 0},
		},
		MultiTurn: &MultiTurnSpec{
			MaxRounds:     roundsForThisSession,
			ThinkTimeUs:   thinkTimeUs,
			ContextGrowth: "", // fixed-length: matches real inference-perf behavior
			SingleSession: true,
		},
	}
}
```

- [ ] **Step 4: Update single-stage multi-turn path (lines 114-140)**

Replace lines 114-140 in `sim/workload/inference_perf.go` with:

```go
		var reasoning *ReasoningSpec
		if sp.EnableMultiTurnChat {
			// Multi-turn: distribute total requests evenly across sessions
			totalRequests := int(stage.Rate * float64(stage.Duration))
			perSessionRounds := distributeRequestsEvenly(totalRequests, numClientsPerStage)

			// Multi-turn: use Poisson arrival for session start time.
			// The rounds within each session are spaced by ThinkTimeUs (not sampled IATs).
			clientIdx := 0
			for p := 0; p < sp.NumUniqueSystemPrompts; p++ {
				prefixGroup := fmt.Sprintf("prompt-%d", p)
				for u := 0; u < sp.NumUsersPerSystemPrompt; u++ {
					clientID := fmt.Sprintf("prompt-%d-user-%d", p, u)
					reasoning = computeReasoningSpec(stage.Rate, stage.Duration, numClientsPerStage, perSessionRounds[clientIdx])
					clientIdx++

					clients = append(clients, ClientSpec{
						ID:           clientID,
						TenantID:     prefixGroup,
						SLOClass:     "standard",
						RateFraction: rateFraction,
						Arrival:      ArrivalSpec{Process: "poisson"},
						InputDist:    inputDist,
						OutputDist:   outputDist,
						PrefixGroup:  prefixGroup,
						PrefixLength: sp.SystemPromptLen,
						Reasoning:    reasoning,
					})
				}
			}
		} else {
```

- [ ] **Step 5: Update multi-stage multi-turn path (lines 225-248)**

Replace lines 225-248 in `sim/workload/inference_perf.go` with:

```go
			var reasoning *ReasoningSpec
			if sp.EnableMultiTurnChat {
				// Multi-turn: distribute total requests evenly across sessions for this stage
				totalRequests := int(stage.Rate * float64(stage.Duration))
				perSessionRounds := distributeRequestsEvenly(totalRequests, numClientsPerStage)

				clientIdx := 0
				for p := 0; p < sp.NumUniqueSystemPrompts; p++ {
					prefixGroup := fmt.Sprintf("prompt-%d", p)
					for u := 0; u < sp.NumUsersPerSystemPrompt; u++ {
						clientID := fmt.Sprintf("stage-%d-prompt-%d-user-%d", s, p, u)
						reasoning = computeReasoningSpec(stage.Rate, stage.Duration, numClientsPerStage, perSessionRounds[clientIdx])
						clientIdx++

						clients = append(clients, ClientSpec{
							ID:           clientID,
							TenantID:     prefixGroup,
							SLOClass:     "standard",
							RateFraction: rateFraction,
							Arrival:      ArrivalSpec{Process: "poisson"},
							InputDist:    inputDist,
							OutputDist:   outputDist,
							PrefixGroup:  prefixGroup,
							PrefixLength: sp.SystemPromptLen,
							Reasoning:    reasoning,
							Lifecycle:    stageLifecycle,
						})
					}
				}
			} else {
				// Non-multi-turn multi-stage: use Poisson
				for p := 0; p < sp.NumUniqueSystemPrompts; p++ {
					prefixGroup := fmt.Sprintf("prompt-%d", p)
					for u := 0; u < sp.NumUsersPerSystemPrompt; u++ {
						clientID := fmt.Sprintf("stage-%d-prompt-%d-user-%d", s, p, u)
						clients = append(clients, ClientSpec{
							ID:           clientID,
							TenantID:     prefixGroup,
							SLOClass:     "standard",
							RateFraction: rateFraction,
							Arrival:      ArrivalSpec{Process: "poisson"},
							InputDist:    inputDist,
							OutputDist:   outputDist,
							PrefixGroup:  prefixGroup,
							PrefixLength: sp.SystemPromptLen,
							Reasoning:    reasoning,
							Lifecycle:    stageLifecycle,
						})
					}
				}
			}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
go test ./sim/workload/... -run "MultiTurn_ExactRequestCount|MultiTurn_PerSessionFairness" -v
```

**Expected:** PASS (3 test cases, all showing exact counts)

- [ ] **Step 7: Run full workload test suite**

```bash
go test ./sim/workload/... -v
```

**Expected:** PASS (all tests pass)

- [ ] **Step 8: Run lint**

```bash
golangci-lint run ./sim/workload/...
```

**Expected:** No issues

- [ ] **Step 9: Commit**

```bash
git add sim/workload/inference_perf.go sim/workload/inference_perf_test.go
git commit -m "fix(workload): use exact distribution for multi-turn MaxRounds

Replaces math.Ceil (line 295) with per-session fair distribution.

BC-2, BC-3: Multi-turn workloads now generate exactly int(rate × duration) requests

Fixes #978 (complete: all affected code paths fixed)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Add Determinism Regression Test

**Files:**
- Test: `sim/workload/inference_perf_test.go` (add after multi-turn tests)

- [ ] **Step 1: Write determinism test**

Add to `sim/workload/inference_perf_test.go`:

```go
func TestExpandInferencePerfSpec_ExactDistribution_PreservesDeterminism(t *testing.T) {
	// BC-5: Determinism preserved after fix (INV-6)
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 5.0, Duration: 600},
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  11,
			NumUsersPerSystemPrompt: 4,
			SystemPromptLen:         100,
			QuestionLen:             200,
			OutputLen:               50,
			EnableMultiTurnChat:     true,
		},
	}

	// Generate twice with same seed
	horizon := int64(1_200_000_000)

	expanded1, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("expansion1 error: %v", err)
	}
	r1, err := GenerateRequests(expanded1, horizon, 0)
	if err != nil {
		t.Fatalf("generation1 error: %v", err)
	}

	expanded2, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("expansion2 error: %v", err)
	}
	r2, err := GenerateRequests(expanded2, horizon, 0)
	if err != nil {
		t.Fatalf("generation2 error: %v", err)
	}

	// Verify byte-identical output
	if len(r1) != len(r2) {
		t.Fatalf("different counts: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			if i >= 5 {
				t.Logf("... (stopping after 5 mismatches)")
				break
			}
		}
		if r1[i].ID != r2[i].ID {
			t.Errorf("request %d: ID %q vs %q", i, r1[i].ID, r2[i].ID)
			if i >= 5 {
				break
			}
		}
	}
}
```

- [ ] **Step 2: Run test**

```bash
go test ./sim/workload/... -run TestExpandInferencePerfSpec_ExactDistribution_PreservesDeterminism -v
```

**Expected:** PASS (determinism verified)

- [ ] **Step 3: Commit**

```bash
git add sim/workload/inference_perf_test.go
git commit -m "test(workload): add determinism regression test for exact distribution

BC-5: Verify INV-6 preserved after fix

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 5: Update Stale Comment and Final Verification

**Files:**
- Modify: `sim/workload/inference_perf.go` (delete stale comment)

- [ ] **Step 1: Remove incorrect comment**

In `sim/workload/inference_perf.go`, the old code had lines 146-147:
```go
// NOTE: math.Ceil means total requests may exceed stage.Rate * stage.Duration
// by up to numClients-1. This matches inference-perf behavior.
```

This comment was removed in Task 2 when we replaced the code. Verify it's gone:

```bash
grep -n "This matches inference-perf behavior" sim/workload/inference_perf.go
```

**Expected:** No output (comment already removed in Task 2)

- [ ] **Step 2: Run full test suite**

```bash
go test ./sim/workload/... -v
```

**Expected:** PASS (all tests pass)

- [ ] **Step 3: Run full project test suite**

```bash
go test ./... -count=1
```

**Expected:** PASS (no regressions in other packages)

- [ ] **Step 4: Run lint**

```bash
golangci-lint run ./...
```

**Expected:** No issues

- [ ] **Step 5: Build binary**

```bash
go build -o blis main.go
```

**Expected:** Successful build

- [ ] **Step 6: Verify git status**

```bash
git status
```

**Expected:** Working tree clean (all changes committed)

- [ ] **Step 7: Push branch**

```bash
git push -u origin pr978-fix-inference-perf-request-count
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ BC-1: Single-stage non-multi-turn exact count (Task 2)
- ✅ BC-2: Single-stage multi-turn exact count (Task 3)
- ✅ BC-3: Multi-stage multi-turn exact count per stage (Task 3)
- ✅ BC-4: Fair distribution (Task 1, verified in Tasks 2-3)
- ✅ BC-5: Determinism preserved (Task 4)
- ✅ Issue #978 examples: 3000, 6000, 9000 (tested in Task 3)
- ✅ Stale comment removal (verified in Task 5)

**Placeholder scan:** None found (all code complete)

**Type consistency:**
- `distributeRequestsEvenly(int, int) []int` - consistent across all usages
- `computeReasoningSpec(..., int)` - signature updated consistently
- `perClientDist`, `perSessionRounds` - naming consistent

---

## Expected Outcomes

After completing all tasks:

1. **Request counts exact:** 3000, 6000, 9000 (not 3036, 6028, 9064)
2. **All tests pass:** 30+ new test cases, 0 failures
3. **No lint issues:** Clean golangci-lint output
4. **Determinism preserved:** Same seed → identical output
5. **Fair distribution:** Per-client/per-session counts differ by ≤ 1
6. **Ready for PR:** Branch pushed, ready for `gh pr create`

---

