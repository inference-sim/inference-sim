# Deadline-Aware SLO Scheduling Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `DeadlineAwarePriority` and `StaticClassWeight` priority policies, then run a 6-arm hypothesis bundle comparing deadline-aware urgency against static class weights and age-only priority.

**Architecture:** Two new `PriorityPolicy` implementations added to `sim/priority.go`, registered in `sim/bundle.go`. `PriorityConfig` extended with optional deadline parameters. Experiment code in `hypotheses/h-deadline-urgency/` using the shared harness.

**Tech Stack:** Go (simulator), Bash (experiment harness), Python (analysis)

**Design doc:** `hypotheses/h-deadline-urgency/problem.md`

---

## Task 1: Implement StaticClassWeight priority policy

**Files:**
- Modify: `sim/priority.go` — add `StaticClassWeight` struct and `Compute` method
- Modify: `sim/bundle.go:63` — add `"static-class-weight"` to `validPriorityPolicies`
- Modify: `sim/priority.go:54-68` — add case to `NewPriorityPolicy` factory
- Test: `sim/priority_test.go`

- [ ] **Step 1: Write failing test for StaticClassWeight**

```go
func TestStaticClassWeight_ReturnsClassSpecificPriority(t *testing.T) {
	weights := map[string]float64{"critical": 10.0, "standard": 5.0, "sheddable": 1.0}
	p := &StaticClassWeight{ClassWeights: weights, DefaultWeight: 0.0}

	critical := &Request{SLOClass: "critical", ArrivalTime: 0}
	standard := &Request{SLOClass: "standard", ArrivalTime: 0}
	sheddable := &Request{SLOClass: "sheddable", ArrivalTime: 0}
	unknown := &Request{SLOClass: "unknown", ArrivalTime: 0}

	if got := p.Compute(critical, 1000); got != 10.0 {
		t.Errorf("critical: got %f, want 10.0", got)
	}
	if got := p.Compute(standard, 1000); got != 5.0 {
		t.Errorf("standard: got %f, want 5.0", got)
	}
	if got := p.Compute(sheddable, 1000); got != 1.0 {
		t.Errorf("sheddable: got %f, want 1.0", got)
	}
	if got := p.Compute(unknown, 1000); got != 0.0 {
		t.Errorf("unknown SLO class: got %f, want 0.0 (default)", got)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .worktrees/hypothesis-playground && go test ./sim/... -run TestStaticClassWeight -v`
Expected: FAIL — `StaticClassWeight` not defined

- [ ] **Step 3: Implement StaticClassWeight**

Add to `sim/priority.go` before `NewPriorityPolicy`:

```go
// StaticClassWeight assigns a fixed priority based on request SLO class.
// Provides class-aware ordering without time dependence.
// Used as B2 baseline to isolate deadline-awareness from class-awareness.
type StaticClassWeight struct {
	ClassWeights  map[string]float64
	DefaultWeight float64
}

func (s *StaticClassWeight) Compute(req *Request, _ int64) float64 {
	if w, ok := s.ClassWeights[req.SLOClass]; ok {
		return w
	}
	return s.DefaultWeight
}
```

- [ ] **Step 4: Register in bundle.go and factory**

In `sim/bundle.go:63`, add `"static-class-weight": true` to `validPriorityPolicies`.

In `sim/priority.go` `NewPriorityPolicy`, add case:
```go
case "static-class-weight":
	return &StaticClassWeight{
		ClassWeights:  map[string]float64{"critical": 10.0, "standard": 5.0, "sheddable": 1.0},
		DefaultWeight: 0.0,
	}
```

- [ ] **Step 5: Run tests to verify pass**

Run: `cd .worktrees/hypothesis-playground && go test ./sim/... -run TestStaticClassWeight -v`
Expected: PASS

- [ ] **Step 6: Add factory test for new policy name**

Add to `TestNewPriorityPolicy_ValidNames_ReturnsBehaviorallyCorrectPolicy` in `sim/priority_test.go`:
```go
// static-class-weight: returns class-specific weights
p4 := NewPriorityPolicy("static-class-weight")
critReq := &Request{SLOClass: "critical", ArrivalTime: 0}
shedReq := &Request{SLOClass: "sheddable", ArrivalTime: 0}
if p4.Compute(critReq, 0) <= p4.Compute(shedReq, 0) {
	t.Errorf("static-class-weight: critical priority should be > sheddable priority")
}
```

- [ ] **Step 7: Run full test suite and commit**

Run: `cd .worktrees/hypothesis-playground && go test ./sim/... -v`
Expected: All tests PASS

---

## Task 2: Implement DeadlineAwarePriority policy

**Files:**
- Modify: `sim/priority.go` — add `DeadlineAwarePriority` struct and `Compute` method
- Modify: `sim/bundle.go:63` — add `"deadline-aware"` to `validPriorityPolicies`
- Modify: `sim/priority.go` — add case to `NewPriorityPolicy` factory
- Test: `sim/priority_test.go`

- [ ] **Step 1: Write failing tests for DeadlineAwarePriority**

```go
func TestDeadlineAwarePriority_UrgencyGrowsWithElapsedTime(t *testing.T) {
	p := &DeadlineAwarePriority{
		ClassWeights: map[string]float64{"critical": 10.0, "standard": 5.0, "sheddable": 1.0},
		Deadlines:    map[string]int64{"critical": 100_000, "standard": 500_000, "sheddable": 2_000_000},
		Epsilon:      0.01,
		DefaultWeight:   0.0,
		DefaultDeadline: 500_000,
	}

	req := &Request{SLOClass: "critical", ArrivalTime: 0}

	// At t=0: urgency = 10.0 / max(0.01, 1.0 - 0/100000) = 10.0
	u0 := p.Compute(req, 0)
	if u0 != 10.0 {
		t.Errorf("t=0: got %f, want 10.0", u0)
	}

	// At t=50000 (50% of deadline): urgency = 10.0 / 0.5 = 20.0
	u50 := p.Compute(req, 50_000)
	if u50 != 20.0 {
		t.Errorf("t=50000: got %f, want 20.0", u50)
	}

	// Urgency monotonically increases
	if u50 <= u0 {
		t.Errorf("urgency should increase with elapsed time: u0=%f, u50=%f", u0, u50)
	}

	// At t=100000 (at deadline): urgency = 10.0 / 0.01 = 1000.0 (epsilon cap)
	uCap := p.Compute(req, 100_000)
	if uCap != 1000.0 {
		t.Errorf("t=100000 (at deadline): got %f, want 1000.0", uCap)
	}

	// Past deadline: urgency stays at cap
	uPast := p.Compute(req, 200_000)
	if uPast != 1000.0 {
		t.Errorf("t=200000 (past deadline): got %f, want 1000.0", uPast)
	}
}

func TestDeadlineAwarePriority_ClassOrdering(t *testing.T) {
	p := &DeadlineAwarePriority{
		ClassWeights: map[string]float64{"critical": 10.0, "standard": 5.0, "sheddable": 1.0},
		Deadlines:    map[string]int64{"critical": 100_000, "standard": 500_000, "sheddable": 2_000_000},
		Epsilon:      0.01,
		DefaultWeight:   0.0,
		DefaultDeadline: 500_000,
	}

	clock := int64(50_000)
	critical := &Request{SLOClass: "critical", ArrivalTime: 0}
	standard := &Request{SLOClass: "standard", ArrivalTime: 0}
	sheddable := &Request{SLOClass: "sheddable", ArrivalTime: 0}

	uc := p.Compute(critical, clock)
	us := p.Compute(standard, clock)
	ush := p.Compute(sheddable, clock)

	if uc <= us || us <= ush {
		t.Errorf("class ordering violated: critical=%f, standard=%f, sheddable=%f", uc, us, ush)
	}
}

func TestDeadlineAwarePriority_StarvationCrossover(t *testing.T) {
	p := &DeadlineAwarePriority{
		ClassWeights: map[string]float64{"critical": 10.0, "sheddable": 1.0},
		Deadlines:    map[string]int64{"critical": 100_000, "sheddable": 2_000_000},
		Epsilon:      0.01,
		DefaultWeight:   0.0,
		DefaultDeadline: 500_000,
	}

	// Fresh critical request
	freshCritical := &Request{SLOClass: "critical", ArrivalTime: 1_800_001}
	critUrgency := p.Compute(freshCritical, 1_800_001) // elapsed=0 → urgency=10.0

	// Old sheddable request that arrived at t=0, now at t=1800001 (past 1.8s crossover)
	oldSheddable := &Request{SLOClass: "sheddable", ArrivalTime: 0}
	shedUrgency := p.Compute(oldSheddable, 1_800_001)

	if shedUrgency <= critUrgency {
		t.Errorf("starvation crossover: sheddable at 1.8s (%f) should exceed fresh critical (%f)",
			shedUrgency, critUrgency)
	}
}

func TestDeadlineAwarePriority_INV9_DoesNotReadOutputTokens(t *testing.T) {
	p := &DeadlineAwarePriority{
		ClassWeights: map[string]float64{"critical": 10.0},
		Deadlines:    map[string]int64{"critical": 100_000},
		Epsilon:      0.01,
		DefaultWeight:   0.0,
		DefaultDeadline: 500_000,
	}

	req1 := &Request{SLOClass: "critical", ArrivalTime: 0, OutputTokens: []int{1, 2, 3}}
	req2 := &Request{SLOClass: "critical", ArrivalTime: 0, OutputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}

	if p.Compute(req1, 50_000) != p.Compute(req2, 50_000) {
		t.Error("INV-9 violation: priority differs based on OutputTokens")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .worktrees/hypothesis-playground && go test ./sim/... -run TestDeadlineAware -v`
Expected: FAIL — `DeadlineAwarePriority` not defined

- [ ] **Step 3: Implement DeadlineAwarePriority**

Add to `sim/priority.go` before `NewPriorityPolicy`:

```go
// DeadlineAwarePriority computes urgency from per-SLO-class TTFT deadlines.
// Urgency grows hyperbolically as elapsed time approaches the class deadline,
// creating stronger priority separation than linear age-weighting.
// Formula: classWeight / max(epsilon, 1.0 - elapsed / deadline)
// Per-round deadline: each Request has its own ArrivalTime, so multi-turn
// rounds get independent deadline budgets.
type DeadlineAwarePriority struct {
	ClassWeights    map[string]float64 // per SLO class base weight
	Deadlines       map[string]int64   // per SLO class TTFT target in ticks (μs)
	Epsilon         float64            // floor to prevent division by zero
	DefaultWeight   float64            // weight for unknown SLO classes
	DefaultDeadline int64              // deadline for unknown SLO classes
}

func (d *DeadlineAwarePriority) Compute(req *Request, clock int64) float64 {
	weight := d.DefaultWeight
	if w, ok := d.ClassWeights[req.SLOClass]; ok {
		weight = w
	}
	deadline := d.DefaultDeadline
	if dl, ok := d.Deadlines[req.SLOClass]; ok {
		deadline = dl
	}
	elapsed := float64(clock - req.ArrivalTime)
	fraction := elapsed / float64(deadline)
	denominator := 1.0 - fraction
	if denominator < d.Epsilon {
		denominator = d.Epsilon
	}
	return weight / denominator
}
```

- [ ] **Step 4: Register in bundle.go and factory**

In `sim/bundle.go:63`, add `"deadline-aware": true` to `validPriorityPolicies`.

In `sim/priority.go` `NewPriorityPolicy`, add case:
```go
case "deadline-aware":
	return &DeadlineAwarePriority{
		ClassWeights:    map[string]float64{"critical": 10.0, "standard": 5.0, "sheddable": 1.0},
		Deadlines:       map[string]int64{"critical": 100_000, "standard": 500_000, "sheddable": 2_000_000},
		Epsilon:         0.01,
		DefaultWeight:   0.0,
		DefaultDeadline: 500_000,
	}
```

- [ ] **Step 5: Run all tests**

Run: `cd .worktrees/hypothesis-playground && go test ./sim/... -v`
Expected: All tests PASS (including new DeadlineAware tests)

- [ ] **Step 6: Run lint**

Run: `cd .worktrees/hypothesis-playground && golangci-lint run ./...`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add sim/priority.go sim/priority_test.go sim/bundle.go
git commit -m "feat(sim): add DeadlineAwarePriority and StaticClassWeight policies"
```

---

## Task 3: Extend PriorityConfig for deadline parameters

**Files:**
- Modify: `sim/bundle.go` — extend `PriorityConfig` with optional deadline fields
- Modify: `cmd/root.go` — wire bundle priority config to `NewPriorityPolicy`
- Modify: `sim/priority.go` — add `NewPriorityPolicyFromConfig` that accepts `PriorityConfig`
- Test: `sim/bundle_test.go`

- [ ] **Step 1: Write failing test for bundle YAML loading with deadline params**

```go
func TestLoadPolicyBundle_DeadlineAwareConfig(t *testing.T) {
	yaml := `
priority:
  policy: deadline-aware
  class_weights:
    critical: 10.0
    standard: 5.0
    sheddable: 1.0
  deadlines:
    critical: 100000
    standard: 500000
    sheddable: 2000000
  epsilon: 0.01
scheduler: priority-fcfs
`
	tmpFile := filepath.Join(t.TempDir(), "bundle.yaml")
	os.WriteFile(tmpFile, []byte(yaml), 0644)

	bundle, err := LoadPolicyBundle(tmpFile)
	if err != nil {
		t.Fatalf("LoadPolicyBundle: %v", err)
	}
	if bundle.Priority.Policy != "deadline-aware" {
		t.Errorf("policy: got %q, want deadline-aware", bundle.Priority.Policy)
	}
	if bundle.Priority.ClassWeights["critical"] != 10.0 {
		t.Errorf("class_weights.critical: got %f, want 10.0", bundle.Priority.ClassWeights["critical"])
	}
	if bundle.Priority.Deadlines["critical"] != 100000 {
		t.Errorf("deadlines.critical: got %d, want 100000", bundle.Priority.Deadlines["critical"])
	}
	if bundle.Priority.Epsilon == nil || *bundle.Priority.Epsilon != 0.01 {
		t.Errorf("epsilon: want 0.01")
	}
}
```

- [ ] **Step 2: Extend PriorityConfig in bundle.go**

```go
type PriorityConfig struct {
	Policy       string             `yaml:"policy"`
	ClassWeights map[string]float64 `yaml:"class_weights,omitempty"`
	Deadlines    map[string]int64   `yaml:"deadlines,omitempty"`
	Epsilon      *float64           `yaml:"epsilon,omitempty"` // R9: pointer for zero-valid
}
```

- [ ] **Step 3: Add NewPriorityPolicyFromConfig factory**

Add to `sim/priority.go`:

```go
// NewPriorityPolicyFromConfig creates a PriorityPolicy from a PriorityConfig.
// For policies requiring parameters (deadline-aware, static-class-weight),
// reads configuration from the config struct. Falls back to defaults for missing fields.
func NewPriorityPolicyFromConfig(cfg PriorityConfig) PriorityPolicy {
	switch cfg.Policy {
	case "deadline-aware":
		weights := cfg.ClassWeights
		if weights == nil {
			weights = map[string]float64{"critical": 10.0, "standard": 5.0, "sheddable": 1.0}
		}
		deadlines := cfg.Deadlines
		if deadlines == nil {
			deadlines = map[string]int64{"critical": 100_000, "standard": 500_000, "sheddable": 2_000_000}
		}
		eps := 0.01
		if cfg.Epsilon != nil {
			eps = *cfg.Epsilon
		}
		return &DeadlineAwarePriority{
			ClassWeights:    weights,
			Deadlines:       deadlines,
			Epsilon:         eps,
			DefaultWeight:   0.0,
			DefaultDeadline: 500_000,
		}
	case "static-class-weight":
		weights := cfg.ClassWeights
		if weights == nil {
			weights = map[string]float64{"critical": 10.0, "standard": 5.0, "sheddable": 1.0}
		}
		return &StaticClassWeight{ClassWeights: weights, DefaultWeight: 0.0}
	default:
		return NewPriorityPolicy(cfg.Policy)
	}
}
```

- [ ] **Step 4: Wire in cmd/root.go**

Where `NewPriorityPolicy` is called indirectly (via `NewPolicyConfig`), update the simulator construction to pass the full `PriorityConfig` from the loaded bundle when `--policy-config` is provided. This requires `NewSimulator` to accept `PriorityConfig` instead of just the policy name string.

Alternative (simpler, experiment-scoped): In `cmd/root.go`, after loading the bundle, construct the priority policy directly and set it on the simulator config. The `PriorityPolicy` string field stays as-is for backward compat; the new factory is called when the bundle has extra fields.

- [ ] **Step 5: Run full test suite**

Run: `cd .worktrees/hypothesis-playground && go test ./... -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add sim/bundle.go sim/priority.go sim/priority_test.go sim/bundle_test.go cmd/root.go
git commit -m "feat(sim): extend PriorityConfig for deadline-aware parameters"
```

---

## Task 4: Write experiment run.sh

**Files:**
- Create: `hypotheses/h-deadline-urgency/run.sh`

- [ ] **Step 1: Write run.sh sourcing the shared harness**

The script must:
1. Source `hypotheses/lib/harness.sh`
2. Call `setup_experiment`
3. Generate workload YAML files inline (multi-turn, mixed SLO, gamma)
4. Generate policy bundle YAMLs for B2 and Treatment
5. Run all 4 configurations (B0, B1, B2, Treatment) × 3 rates × 3 seeds = 36 runs for H-main
6. Run ablation, control-negative, burst sweep, single-turn arms
7. Save results to organized directory structure

Key CLI flags for all runs:
```bash
COMMON_FLAGS="--model $MODEL --tp 2 --hardware H100 --num-instances 4 \
  --routing-scorers prefix-affinity:3,queue-depth:2 --num-requests 1500"
```

- [ ] **Step 2: Test run.sh with a quick sanity check**

Run: `cd .worktrees/hypothesis-playground && bash hypotheses/h-deadline-urgency/run.sh --dry-run` (if supported) or run with `--num-requests 10` at one rate point to verify CLI flags are accepted.

- [ ] **Step 3: Commit**

```bash
git add hypotheses/h-deadline-urgency/run.sh
git commit -m "feat(experiment): add run.sh for h-deadline-urgency"
```

---

## Task 5: Write experiment analyze.py

**Files:**
- Create: `hypotheses/h-deadline-urgency/analyze.py`

- [ ] **Step 1: Write analyze.py using shared helpers**

The script must:
1. Import `hypotheses/lib/analyze_helpers.py`
2. Parse BLIS JSON output for each arm
3. Compute per-SLO-class TTFT P99 for Treatment, B0, B1, B2
4. Compute improvement percentages (Treatment vs B2, Treatment vs B1)
5. Compute cluster-wide TTFT P99 degradation vs B0
6. Compute zero-sum weighted mean check
7. Output verdict per hypothesis arm (CONFIRMED/PARTIALLY_CONFIRMED/REFUTED)

- [ ] **Step 2: Commit**

```bash
git add hypotheses/h-deadline-urgency/analyze.py
git commit -m "feat(experiment): add analyze.py for h-deadline-urgency"
```

---

## Task 6: Execute experiments and analyze results

- [ ] **Step 1: Build the binary**

Run: `cd .worktrees/hypothesis-playground && go build -o blis main.go`

- [ ] **Step 2: Run the full experiment suite**

Run: `cd .worktrees/hypothesis-playground && bash hypotheses/h-deadline-urgency/run.sh`
Expected: All runs complete without timeouts. Results in `hypotheses/h-deadline-urgency/results/`.

- [ ] **Step 3: Run analysis**

Run: `cd .worktrees/hypothesis-playground && python3 hypotheses/h-deadline-urgency/analyze.py`
Expected: Per-arm verdicts printed. Record in FINDINGS.md.

- [ ] **Step 4: Write FINDINGS.md**

Create `hypotheses/h-deadline-urgency/FINDINGS.md` using the template at `docs/contributing/templates/hypothesis.md`. Include:
- Prediction vs outcome for each arm
- Per-seed data tables
- Status classification per arm
- Issues to file

- [ ] **Step 5: Commit all results**

```bash
git add hypotheses/h-deadline-urgency/
git commit -m "feat(experiment): h-deadline-urgency results and FINDINGS"
```
