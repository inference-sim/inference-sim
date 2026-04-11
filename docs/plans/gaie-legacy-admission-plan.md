# GAIE-Legacy Admission Policy Implementation Plan

**Goal:** Add a `gaie-legacy` admission policy that simulates production llm-d/GAIE saturation-based admission: non-sheddable requests always pass; sheddable requests are rejected when the pool-average saturation >= 1.0.

**Source:** [#1014](https://github.com/inference-sim/inference-sim/issues/1014), Part C of [#1011](https://github.com/inference-sim/inference-sim/issues/1011).

**Closes:** `Fixes #1014`

## Behavioral Contracts

**BC-1: Non-sheddable requests always admitted**
- GIVEN a GAIELegacyAdmission policy with any saturation level
- WHEN Admit() is called for a request with priority >= 0 (critical, standard, or any non-sheddable class)
- THEN admitted=true regardless of saturation

**BC-2: Sheddable requests rejected when saturated**
- GIVEN a GAIELegacyAdmission policy
- WHEN Admit() is called for a request with priority < 0 (batch, sheddable, background) AND pool-average saturation >= 1.0
- THEN admitted=false with reason containing "gaie-saturated"

**BC-3: Sheddable requests admitted when not saturated**
- GIVEN a GAIELegacyAdmission policy
- WHEN Admit() is called for a sheddable request AND pool-average saturation < 1.0
- THEN admitted=true

**BC-4: Saturation formula matches GAIE**
- GIVEN N instance snapshots with QueueDepth and KVUtilization values
- WHEN saturation is computed
- THEN result = avg across instances of max(queueDepth/qdThreshold, kvUtil/kvThreshold)

**BC-5: Empty snapshots treated as saturated**
- GIVEN zero instance snapshots (empty pool)
- WHEN saturation is computed
- THEN saturation = 1.0 (conservative, matches GAIE stale-metrics behavior)

**BC-6: shedByTier tracks rejections from all admission policies**
- GIVEN any admission policy (gaie-legacy, tier-shed, token-bucket, etc.)
- WHEN a request is rejected by admission
- THEN shedByTier[class] is incremented (no type gate)

**BC-7: INV-1 conservation holds**
- GIVEN a simulation using gaie-legacy admission
- WHEN the simulation completes
- THEN num_requests == injected_requests + rejected_requests

**BC-8: YAML configuration**
- GIVEN a policy bundle YAML with `admission.policy: "gaie-legacy"` and optional `gaie_qd_threshold`/`gaie_kv_threshold`
- WHEN the bundle is loaded and validated
- THEN the thresholds are applied (defaults: qd=5, kv=0.8)

## Deviation Log

| Source Says | Plan Does | Reason |
|---|---|---|
| Issue proposes `--gaie-qd-threshold` and `--gaie-kv-threshold` CLI flags | Configure via policy bundle YAML only | CONSISTENCY — tier-shed parameters are YAML-only too. No direct CLI flags for policy-specific thresholds. |
| Issue says "Remove TierShedAdmission type gate" | Remove type gate, track per-tier rejections for all policies | CLARIFICATION — The type gate at `cluster_event.go:150` currently restricts shedByTier to tier-shed only. Removing it means all admission rejections (gaie-legacy, tier-shed, token-bucket, reject-all) increment shedByTier. |
| GAIE `KVCacheUsagePercent` naming suggests percentage | BLIS `KVUtilization` is a fraction (0.0-1.0), same scale as GAIE despite the field name | CORRECTION — Both are fractions. No conversion needed. Verified: GAIE threshold default is 0.8, BLIS KVUtilization is in [0,1]. |
| GAIE treats stale/nil metrics as score=1.0 per pod | BLIS trusts all snapshot values unconditionally (no per-snapshot staleness) | SIMPLIFICATION — BLIS controls snapshot freshness via `SnapshotRefreshInterval`. Per-snapshot staleness is a signal-layer concern, not an admission-layer concern. Empty pool (BC-5) is the only degenerate case. |

## Tasks

### Task 1: Add GAIELegacyAdmission type and unit tests (BC-1, BC-2, BC-3, BC-4, BC-5)

**Files:** modify `sim/admission.go`, create tests in `sim/admission_tier_test.go`

**What to add in `sim/admission.go`** (after `TierShedAdmission` section, before `NewAdmissionPolicy`):

```go
// GAIELegacyAdmission simulates production llm-d/GAIE admission behavior.
// Non-sheddable requests (priority >= 0) always pass. Sheddable requests
// (priority < 0) are rejected when pool-average saturation >= 1.0.
// Saturation formula: avg across instances of max(qd/qdThreshold, kvUtil/kvThreshold).
// Empty snapshots → saturation=1.0 (conservative, matches GAIE stale-metrics behavior).
type GAIELegacyAdmission struct {
	QDThreshold float64         // queue depth threshold (GAIE default: 5)
	KVThreshold float64         // KV cache utilization threshold (GAIE default: 0.8)
	PriorityMap *SLOPriorityMap // priority mapping for IsSheddable check
}

// NewGAIELegacyAdmission creates a GAIELegacyAdmission with validated parameters.
// Panics if qdThreshold <= 0 or kvThreshold <= 0 or kvThreshold > 1.0 (R3).
// If priorityMap is nil, DefaultSLOPriorityMap() is used.
func NewGAIELegacyAdmission(qdThreshold, kvThreshold float64, priorityMap *SLOPriorityMap) *GAIELegacyAdmission {
	if qdThreshold <= 0 || math.IsNaN(qdThreshold) || math.IsInf(qdThreshold, 0) {
		panic(fmt.Sprintf("NewGAIELegacyAdmission: qdThreshold must be > 0, got %v", qdThreshold))
	}
	if kvThreshold <= 0 || kvThreshold > 1.0 || math.IsNaN(kvThreshold) || math.IsInf(kvThreshold, 0) {
		panic(fmt.Sprintf("NewGAIELegacyAdmission: kvThreshold must be in (0, 1.0], got %v", kvThreshold))
	}
	if priorityMap == nil {
		priorityMap = DefaultSLOPriorityMap()
	}
	return &GAIELegacyAdmission{
		QDThreshold: qdThreshold,
		KVThreshold: kvThreshold,
		PriorityMap: priorityMap,
	}
}

// Admit implements AdmissionPolicy. Non-sheddable requests always pass.
// Sheddable requests are rejected when pool-average saturation >= 1.0.
func (g *GAIELegacyAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	if !g.PriorityMap.IsSheddable(req.SLOClass) {
		return true, ""
	}
	sat := g.saturation(state.Snapshots)
	if sat >= 1.0 {
		return false, fmt.Sprintf("gaie-saturated: class=%s saturation=%.2f", req.SLOClass, sat)
	}
	return true, ""
}

// saturation computes pool-average saturation per GAIE formula:
// avg across instances of max(queueDepth/qdThreshold, kvUtil/kvThreshold).
// Empty snapshots → 1.0 (conservative).
func (g *GAIELegacyAdmission) saturation(snapshots []RoutingSnapshot) float64 {
	if len(snapshots) == 0 {
		return 1.0
	}
	var total float64
	for _, snap := range snapshots {
		qRatio := float64(snap.QueueDepth) / g.QDThreshold
		kvRatio := snap.KVUtilization / g.KVThreshold
		total += max(qRatio, kvRatio)
	}
	return total / float64(len(snapshots))
}
```

**Tests to add in `sim/admission_tier_test.go`** (append at end of file):

```go
// BC-1: Non-sheddable requests always admitted, even at extreme saturation.
func TestGAIELegacy_NonSheddableAlwaysAdmitted(t *testing.T) {
	policy := NewGAIELegacyAdmission(5, 0.8, nil)
	// Saturated: QueueDepth=100 → qRatio=20, KVUtil=1.0 → kvRatio=1.25 → sat=20 >> 1.0
	state := &RouterState{
		Snapshots: []RoutingSnapshot{{ID: "i0", QueueDepth: 100, KVUtilization: 1.0}},
	}
	for _, class := range []string{"critical", "standard", "", "unknown"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, _ := policy.Admit(req, state)
		if !admitted {
			t.Errorf("class=%q: non-sheddable must always be admitted, got rejected", class)
		}
	}
}

// BC-2: Sheddable requests rejected when saturation >= 1.0.
func TestGAIELegacy_SheddableRejectedWhenSaturated(t *testing.T) {
	policy := NewGAIELegacyAdmission(5, 0.8, nil)
	// qRatio=5/5=1.0, kvRatio=0.8/0.8=1.0 → sat=1.0 (exactly at boundary)
	state := &RouterState{
		Snapshots: []RoutingSnapshot{{ID: "i0", QueueDepth: 5, KVUtilization: 0.8}},
	}
	for _, class := range []string{"batch", "sheddable", "background"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, reason := policy.Admit(req, state)
		if admitted {
			t.Errorf("class=%q: sheddable must be rejected at saturation=1.0", class)
		}
		if !strings.Contains(reason, "gaie-saturated") {
			t.Errorf("class=%q: reason should contain 'gaie-saturated', got %q", class, reason)
		}
	}
}

// BC-3: Sheddable requests admitted when saturation < 1.0.
func TestGAIELegacy_SheddableAdmittedWhenNotSaturated(t *testing.T) {
	policy := NewGAIELegacyAdmission(5, 0.8, nil)
	// qRatio=2/5=0.4, kvRatio=0.3/0.8=0.375 → sat=0.4 < 1.0
	state := &RouterState{
		Snapshots: []RoutingSnapshot{{ID: "i0", QueueDepth: 2, KVUtilization: 0.3}},
	}
	for _, class := range []string{"batch", "sheddable", "background"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, _ := policy.Admit(req, state)
		if !admitted {
			t.Errorf("class=%q: sheddable should be admitted at saturation < 1.0", class)
		}
	}
}

// BC-4: Saturation formula matches GAIE: avg(max(qd/qdT, kv/kvT)).
func TestGAIELegacy_FormulaExact(t *testing.T) {
	policy := NewGAIELegacyAdmission(5, 0.8, nil)
	state := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "i0", QueueDepth: 10, KVUtilization: 0.4}, // max(10/5, 0.4/0.8) = max(2.0, 0.5) = 2.0
			{ID: "i1", QueueDepth: 1, KVUtilization: 0.9},  // max(1/5, 0.9/0.8) = max(0.2, 1.125) = 1.125
		},
	}
	// Expected: avg(2.0, 1.125) = 1.5625
	// Sheddable should be rejected (1.5625 >= 1.0)
	req := &Request{ID: "r", SLOClass: "sheddable"}
	admitted, _ := policy.Admit(req, state)
	if admitted {
		t.Error("sheddable should be rejected at saturation 1.5625")
	}

	// Now: both instances lightly loaded → saturation < 1.0
	state2 := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "i0", QueueDepth: 2, KVUtilization: 0.3}, // max(0.4, 0.375) = 0.4
			{ID: "i1", QueueDepth: 1, KVUtilization: 0.2}, // max(0.2, 0.25) = 0.25
		},
	}
	// Expected: avg(0.4, 0.25) = 0.325 < 1.0
	admitted2, _ := policy.Admit(req, state2)
	if !admitted2 {
		t.Error("sheddable should be admitted at saturation 0.325")
	}
}

// BC-5: Empty snapshots → saturation=1.0 → sheddable rejected.
func TestGAIELegacy_EmptySnapshotsSaturated(t *testing.T) {
	policy := NewGAIELegacyAdmission(5, 0.8, nil)
	state := &RouterState{Snapshots: []RoutingSnapshot{}}

	// Sheddable rejected (empty pool → saturated)
	req := &Request{ID: "r", SLOClass: "sheddable"}
	admitted, _ := policy.Admit(req, state)
	if admitted {
		t.Error("sheddable should be rejected with empty snapshots (saturation=1.0)")
	}

	// Non-sheddable still admitted
	reqCrit := &Request{ID: "r2", SLOClass: "critical"}
	admitted2, _ := policy.Admit(reqCrit, state)
	if !admitted2 {
		t.Error("non-sheddable should still be admitted with empty snapshots")
	}
}
```

**Verify:** `go test ./sim/ -run TestGAIELegacy -v`
**Lint:** `golangci-lint run ./sim/...`

---

### Task 2: Register gaie-legacy in bundle and wire config (BC-8)

**Files:** modify `sim/bundle.go`, `sim/cluster/deployment.go`, `sim/cluster/cluster.go`, `cmd/root.go`, `cmd/replay.go`

**Step 2a — `sim/bundle.go` and `sim/admission.go` (factory):**

1. Add `"gaie-legacy": true` to `validAdmissionPolicies` map (line 68).
2. In `sim/admission.go`, add `"gaie-legacy"` case to `NewAdmissionPolicy` switch that panics with a helpful message: `panic("gaie-legacy requires NewGAIELegacyAdmission; cannot use generic factory")`. This prevents silent misuse if someone calls the generic factory with "gaie-legacy" (same pattern as tier-shed which also bypasses the factory).
2. Add fields to `AdmissionConfig` struct:
   ```go
   GAIEQDThreshold *float64 `yaml:"gaie_qd_threshold"` // nil = use default (5)
   GAIEKVThreshold *float64 `yaml:"gaie_kv_threshold"` // nil = use default (0.8)
   ```
3. Add validation in `Validate()` (after tier-shed validation):
   ```go
   if b.Admission.GAIEQDThreshold != nil && *b.Admission.GAIEQDThreshold <= 0 {
       return fmt.Errorf("gaie_qd_threshold must be > 0, got %v", *b.Admission.GAIEQDThreshold)
   }
   if b.Admission.GAIEKVThreshold != nil && (*b.Admission.GAIEKVThreshold <= 0 || *b.Admission.GAIEKVThreshold > 1.0) {
       return fmt.Errorf("gaie_kv_threshold must be in (0, 1.0], got %v", *b.Admission.GAIEKVThreshold)
   }
   ```

**Step 2b — `sim/cluster/deployment.go`:**

Add fields to `DeploymentConfig` (no YAML tags — DeploymentConfig is code-constructed, not parsed from YAML):
```go
GAIEQDThreshold float64 // GAIE queue depth threshold (default 5)
GAIEKVThreshold float64 // GAIE KV cache util threshold (default 0.8)
```

**Step 2c — `sim/cluster/cluster.go`:**

In `NewClusterSimulator`, change the admission policy construction block (lines 168-176) from:
```go
if config.AdmissionPolicy == "tier-shed" {
    ...
} else {
    admissionPolicy = sim.NewAdmissionPolicy(...)
}
```
to:
```go
if config.AdmissionPolicy == "tier-shed" {
    ...
} else if config.AdmissionPolicy == "gaie-legacy" {
    admissionPolicy = sim.NewGAIELegacyAdmission(config.GAIEQDThreshold, config.GAIEKVThreshold, priorityMap)
} else {
    admissionPolicy = sim.NewAdmissionPolicy(...)
}
```

**Step 2d — `cmd/root.go`:**

1. Add package-level vars:
   ```go
   gaieQDThreshold  float64
   gaieKVThreshold  float64
   ```
   Initialize with defaults: `gaieQDThreshold = 5`, `gaieKVThreshold = 0.8`.

2. In `resolvePolicies()`, after tier-shed block, add:
   ```go
   if bundle.Admission.GAIEQDThreshold != nil {
       gaieQDThreshold = *bundle.Admission.GAIEQDThreshold
   }
   if bundle.Admission.GAIEKVThreshold != nil {
       gaieKVThreshold = *bundle.Admission.GAIEKVThreshold
   }
   ```

3. In `DeploymentConfig` construction (search for `TierShedThreshold:`), add:
   ```go
   GAIEQDThreshold:     gaieQDThreshold,
   GAIEKVThreshold:     gaieKVThreshold,
   ```

**Step 2e — `cmd/replay.go`:**

In `DeploymentConfig` construction, add:
```go
GAIEQDThreshold:     gaieQDThreshold,
GAIEKVThreshold:     gaieKVThreshold,
```

**Verify:** `go build ./...`
**Lint:** `golangci-lint run ./...`

---

### Task 3: Remove shedByTier type gate (BC-6)

**Files:** modify `sim/cluster/cluster_event.go`

**What:** Remove the `*sim.TierShedAdmission` type check at lines 148-156. Replace with unconditional shedByTier increment for ALL admission rejections:

Change from:
```go
cs.rejectedRequests++
// Populate per-tier shed counter only for TierShedAdmission rejections (S-1:
// avoids conflating token-bucket or reject-all rejections with tier-shed counts).
if _, ok := cs.admissionPolicy.(*sim.TierShedAdmission); ok {
    tier := e.request.SLOClass
    if tier == "" {
        tier = "standard"
    }
    cs.shedByTier[tier]++
}
```

To:
```go
cs.rejectedRequests++
// Track per-tier rejection counts for all admission policies.
tier := e.request.SLOClass
if tier == "" {
    tier = "standard"
}
cs.shedByTier[tier]++
```

**Verify:** `go test ./sim/cluster/ -run TestTierShed -v`
**Lint:** `golangci-lint run ./sim/cluster/...`

---

### Task 4: Add integration test (BC-7)

**Files:** add test to `sim/cluster/cluster_tier_test.go`

Add a test that runs gaie-legacy end-to-end and verifies conservation:

```go
func TestGAIELegacy_INV1_Conservation(t *testing.T) {
    const n = 60
    var requests []*sim.Request
    for _, class := range []string{"critical", "sheddable", "background"} {
        for i := 0; i < n/3; i++ {
            requests = append(requests, &sim.Request{
                ID:           fmt.Sprintf("req_%s_%d", class, i),
                ArrivalTime:  int64(i) * 5,
                SLOClass:     class,
                InputTokens:  make([]int, 50),
                OutputTokens: make([]int, 20),
                State:        sim.StateQueued,
            })
        }
    }
    cfg := newTestDeploymentConfig(2)
    cfg.AdmissionPolicy = "gaie-legacy"
    cfg.GAIEQDThreshold = 5
    cfg.GAIEKVThreshold = 0.8
    cs := NewClusterSimulator(cfg, requests, nil)
    mustRun(t, cs)

    // INV-1: num_requests == injected + rejected
    total := cs.InjectedRequests() + cs.RejectedRequests()
    if total != len(requests) {
        t.Errorf("INV-1: total=%d != num_requests=%d", total, len(requests))
    }
}
```

**Verify:** `go test ./sim/cluster/ -run TestGAIELegacy -v`

---

### Task 5: Update documentation

**Files:** modify `docs/guide/admission.md`, `docs/reference/configuration.md`, `CLAUDE.md`

**Step 5a — `docs/guide/admission.md`:**

1. Add `gaie-legacy` row to the Available Policies table:
   ```
   | **GAIE-legacy** | `--admission-policy gaie-legacy` | Simulates production llm-d/GAIE admission. Non-sheddable requests always pass. Sheddable requests rejected when pool-average saturation >= 1.0. See [GAIE-Legacy Admission](#gaie-legacy-admission) below. |
   ```

2. Add new section before "Pipeline Latency" section:
   ````markdown
   ## GAIE-Legacy Admission

   The `gaie-legacy` policy simulates production llm-d/GAIE admission behavior. It uses the same saturation formula as the Gateway API Inference Extension endpoint picker:

   **Decision tree:**
   - **Non-sheddable requests** (priority >= 0): always admitted, regardless of saturation
   - **Sheddable requests** (priority < 0): rejected when pool-average saturation >= 1.0

   **Saturation formula:**
   ```
   For each instance: score = max(queueDepth / qdThreshold, kvUtilization / kvThreshold)
   Pool saturation = average(scores across all instances)
   ```

   This matches GAIE's `utilization` saturation detector. Empty pools (no instances) are treated as saturated (score = 1.0), consistent with GAIE's stale-metrics handling.

   | YAML field | Default | Description |
   |------------|---------|-------------|
   | `admission.gaie_qd_threshold` | 5 | Queue depth threshold per instance. GAIE default. |
   | `admission.gaie_kv_threshold` | 0.8 | KV cache utilization threshold (0.0-1.0). GAIE default. |

   ```yaml
   admission:
     policy: "gaie-legacy"
     gaie_qd_threshold: 5
     gaie_kv_threshold: 0.8
   ```

   ```bash
   ./blis run --model qwen/qwen3-14b \
     --num-instances 4 --rate 500 --num-requests 2000 \
     --admission-policy gaie-legacy \
     --policy-config policies.yaml
   ```
   ````

**Step 5b — `docs/reference/configuration.md`:**

Add a GAIE-legacy section after the tier-shed section, following the same pattern:
```markdown
**GAIE-legacy admission** (`--admission-policy gaie-legacy`): Simulates production llm-d/GAIE saturation-based admission. Configured via `--policy-config` YAML only:

| YAML field | Type | Default | Description |
|------------|------|---------|-------------|
| `admission.gaie_qd_threshold` | float64 | 5 | Queue depth threshold per instance. Must be > 0. |
| `admission.gaie_kv_threshold` | float64 | 0.8 | KV cache utilization threshold. Must be in (0, 1.0]. |
| `admission.slo_priorities` | map[string]int | nil | Custom SLO class priority overrides (shared with tier-shed). |
```

**Step 5c — `CLAUDE.md`:**

Add to Recent Changes section:
```
- GAIE-legacy admission policy (#1014): `gaie-legacy` admission policy simulates production llm-d/GAIE saturation-based admission. Non-sheddable requests (priority >= 0) always pass; sheddable requests (priority < 0) rejected when pool-average saturation >= 1.0. Saturation formula: `avg(max(qd/qdThreshold, kvUtil/kvThreshold))`. Configured via policy bundle YAML: `gaie_qd_threshold` (default 5), `gaie_kv_threshold` (default 0.8). `shedByTier` counters now track rejections from all admission policies (type gate removed).
```

**Verify:** `go build ./... && go test ./... -count=1`

---

## Sanity Checklist

- [ ] No unnecessary abstractions — GAIELegacyAdmission is a single struct with one method and one helper
- [ ] No feature creep — no CLI flags for thresholds (YAML-only, consistent with tier-shed)
- [ ] R3: Constructor validates qdThreshold > 0, kvThreshold in (0, 1.0] (matches GAIE validation)
- [ ] R4: Construction sites audited — cluster.go (1 site), tests (NewGAIELegacyAdmission calls)
- [ ] R8: No exported mutable maps
- [ ] R16: Config grouped in AdmissionConfig
- [ ] INV-1: Conservation tested end-to-end
- [ ] INV-6: Determinism unaffected (no new RNG, no map iteration in output)
- [ ] Documentation: user guide, reference, CLAUDE.md all updated
