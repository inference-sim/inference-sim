package sim

import (
	"fmt"
	"math"
	"testing"
)

func TestConstantPriority_ReturnsFixedScore(t *testing.T) {
	// BC-6: ConstantPriority returns configured score regardless of request/clock
	policy := &ConstantPriority{Score: 5.0}
	req := &Request{ID: "r1", ArrivalTime: 100, InputTokens: make([]int, 50)}

	got := policy.Compute(req, 1000)
	if got != 5.0 {
		t.Errorf("ConstantPriority.Compute: got %f, want 5.0", got)
	}

	// Different request and clock — same result
	req2 := &Request{ID: "r2", ArrivalTime: 500, InputTokens: make([]int, 200)}
	got2 := policy.Compute(req2, 9999)
	if got2 != 5.0 {
		t.Errorf("ConstantPriority.Compute (different req): got %f, want 5.0", got2)
	}
}

func TestConstantPriority_DefaultZero(t *testing.T) {
	// BC-6: Zero-value ConstantPriority returns 0.0
	policy := &ConstantPriority{}
	req := &Request{ID: "r1", ArrivalTime: 0}
	got := policy.Compute(req, 0)
	if got != 0.0 {
		t.Errorf("ConstantPriority (zero): got %f, want 0.0", got)
	}
}

func TestSLOBasedPriority_OlderRequestGetsHigherPriority(t *testing.T) {
	// BC-7: With AgeWeight > 0, older requests get higher priority
	policy := &SLOBasedPriority{BaseScore: 0.0, AgeWeight: 1e-6}
	clock := int64(2000000) // 2 seconds in ticks

	older := &Request{ID: "old", ArrivalTime: 0}      // age = 2s
	newer := &Request{ID: "new", ArrivalTime: 1000000} // age = 1s

	pOlder := policy.Compute(older, clock)
	pNewer := policy.Compute(newer, clock)

	if pOlder <= pNewer {
		t.Errorf("SLOBasedPriority: older=%f should be > newer=%f", pOlder, pNewer)
	}
}

func TestSLOBasedPriority_MonotonicPriorityWithAge(t *testing.T) {
	// BC-6: Priority increases monotonically with request age
	policy := &SLOBasedPriority{BaseScore: 1.0, AgeWeight: 0.5}
	clock := int64(1000)

	// Requests at different ages (arrival times descending = ages ascending)
	arrivalTimes := []int64{900, 700, 500, 200, 0} // newest to oldest
	var prevPriority float64
	for i, arrival := range arrivalTimes {
		req := &Request{ID: fmt.Sprintf("r%d", i), ArrivalTime: arrival}
		priority := policy.Compute(req, clock)
		if i > 0 && priority <= prevPriority {
			t.Errorf("priority not monotonically increasing with age: arrival=%d priority=%f <= prev priority=%f",
				arrival, priority, prevPriority)
		}
		prevPriority = priority
	}

	// Also verify: same-age requests get same priority (determinism)
	reqA := &Request{ID: "a", ArrivalTime: 500}
	reqB := &Request{ID: "b", ArrivalTime: 500}
	if policy.Compute(reqA, clock) != policy.Compute(reqB, clock) {
		t.Error("same-age requests should get identical priority")
	}
}

func TestNewPriorityPolicy_ValidNames_ReturnsBehaviorallyCorrectPolicy(t *testing.T) {
	// BC-1: empty string and "constant" return a policy that produces 0.0 for any input
	req := &Request{ID: "r1", ArrivalTime: 100, InputTokens: make([]int, 50)}

	p1 := NewPriorityPolicy("")
	if got := p1.Compute(req, 1000); got != 0.0 {
		t.Errorf("NewPriorityPolicy(\"\").Compute: got %f, want 0.0", got)
	}

	p2 := NewPriorityPolicy("constant")
	if got := p2.Compute(req, 1000); got != 0.0 {
		t.Errorf("NewPriorityPolicy(\"constant\").Compute: got %f, want 0.0", got)
	}

	// BC-2: "slo-based" returns a policy where older requests get higher priority
	p3 := NewPriorityPolicy("slo-based")
	olderReq := &Request{ID: "old", ArrivalTime: 0}
	newerReq := &Request{ID: "new", ArrivalTime: 500000}
	clock := int64(1000000)

	olderPriority := p3.Compute(olderReq, clock)
	newerPriority := p3.Compute(newerReq, clock)
	if olderPriority <= newerPriority {
		t.Errorf("NewPriorityPolicy(\"slo-based\"): older priority (%f) should be > newer priority (%f)",
			olderPriority, newerPriority)
	}

	// BC-8: "static-class-weight" returns class-differentiated priority
	p4 := NewPriorityPolicy("static-class-weight")
	critReq := &Request{ID: "crit", ArrivalTime: 0, SLOClass: "critical"}
	shedReq := &Request{ID: "shed", ArrivalTime: 0, SLOClass: "sheddable"}
	critP := p4.Compute(critReq, clock)
	shedP := p4.Compute(shedReq, clock)
	if critP <= shedP {
		t.Errorf("NewPriorityPolicy(\"static-class-weight\"): critical (%f) should be > sheddable (%f)",
			critP, shedP)
	}

	// BC-9: "deadline-aware" returns urgency that grows with elapsed time
	p5 := NewPriorityPolicy("deadline-aware")
	earlyReq := &Request{ID: "early", ArrivalTime: 900_000, SLOClass: "standard"}
	lateReq := &Request{ID: "late", ArrivalTime: 100_000, SLOClass: "standard"}
	earlyP := p5.Compute(earlyReq, clock)
	lateP := p5.Compute(lateReq, clock)
	if lateP <= earlyP {
		t.Errorf("NewPriorityPolicy(\"deadline-aware\"): older request (%f) should have > urgency than newer (%f)",
			lateP, earlyP)
	}
}

func TestNewPriorityPolicy_UnknownName_Panics(t *testing.T) {
	// EH-1: unknown name panics
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("NewPriorityPolicy(\"unknown\"): expected panic, got nil")
		}
	}()
	NewPriorityPolicy("unknown")
}

func TestPriorityPolicy_Compute_NoSideEffects(t *testing.T) {
	// NC-3: Compute must not modify the request
	policies := []struct {
		name   string
		policy PriorityPolicy
	}{
		{"constant", &ConstantPriority{Score: 5.0}},
		{"slo-based", &SLOBasedPriority{BaseScore: 1.0, AgeWeight: 1e-6}},
		{"static-class-weight", &StaticClassWeight{
			ClassWeights:  map[string]float64{"critical": 10.0},
			DefaultWeight: 1.0,
		}},
		{"deadline-aware", &DeadlineAwarePriority{
			ClassWeights:    map[string]float64{"critical": 10.0},
			Deadlines:       map[string]int64{"critical": 100_000},
			Epsilon:         0.01,
			DefaultWeight:   1.0,
			DefaultDeadline: 500_000,
		}},
	}
	for _, tc := range policies {
		t.Run(tc.name, func(t *testing.T) {
			req := &Request{
				ID: "r1", ArrivalTime: 100, InputTokens: make([]int, 50),
				Priority: 0.0, State: StateQueued,
			}
			tc.policy.Compute(req, 1000)
			if req.Priority != 0.0 {
				t.Errorf("Compute modified req.Priority: got %f, want 0.0", req.Priority)
			}
			if req.State != StateQueued {
				t.Errorf("Compute modified req.State: got %q, want %q", req.State, StateQueued)
			}
			if req.ID != "r1" {
				t.Errorf("Compute modified req.ID: got %q, want %q", req.ID, "r1")
			}
		})
	}
}

// TestInvertedSLO_OlderRequestsGetLowerPriority verifies BC-5.
func TestInvertedSLO_OlderRequestsGetLowerPriority(t *testing.T) {
	policy := NewPriorityPolicy("inverted-slo")

	oldReq := &Request{ID: "old", ArrivalTime: 0}
	newReq := &Request{ID: "new", ArrivalTime: 900_000}
	clock := int64(1_000_000)

	oldPriority := policy.Compute(oldReq, clock)
	newPriority := policy.Compute(newReq, clock)

	// THEN older request MUST have lower priority than newer request
	if oldPriority >= newPriority {
		t.Errorf("expected older request priority (%f) < newer request priority (%f)", oldPriority, newPriority)
	}
}

// --- StaticClassWeight tests ---

func TestStaticClassWeight_ReturnsClassSpecificPriority(t *testing.T) {
	// GIVEN a StaticClassWeight policy with per-class weights
	policy := &StaticClassWeight{
		ClassWeights:  map[string]float64{"critical": 10.0, "standard": 5.0, "sheddable": 1.0},
		DefaultWeight: 0.0,
	}
	clock := int64(1_000_000)

	tests := []struct {
		name     string
		sloClass string
		want     float64
	}{
		{"critical class", "critical", 10.0},
		{"standard class", "standard", 5.0},
		{"sheddable class", "sheddable", 1.0},
		{"unknown class gets default", "batch", 0.0},
		{"empty class gets default", "", 0.0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := &Request{ID: "r1", ArrivalTime: 0, SLOClass: tc.sloClass}
			got := policy.Compute(req, clock)
			if got != tc.want {
				t.Errorf("StaticClassWeight(%q): got %f, want %f", tc.sloClass, got, tc.want)
			}
		})
	}
}

func TestStaticClassWeight_IgnoresClock(t *testing.T) {
	// GIVEN a StaticClassWeight policy
	// WHEN computing priority at different clock values
	// THEN the result is identical (class-only, not time-aware)
	policy := &StaticClassWeight{
		ClassWeights:  map[string]float64{"critical": 10.0},
		DefaultWeight: 0.0,
	}
	req := &Request{ID: "r1", ArrivalTime: 0, SLOClass: "critical"}

	p1 := policy.Compute(req, 0)
	p2 := policy.Compute(req, 1_000_000)
	p3 := policy.Compute(req, 999_999_999)

	if p1 != p2 || p2 != p3 {
		t.Errorf("StaticClassWeight should be clock-independent: got %f, %f, %f", p1, p2, p3)
	}
}

// --- DeadlineAwarePriority tests ---

func TestDeadlineAwarePriority_UrgencyGrowsWithElapsedTime(t *testing.T) {
	// GIVEN a DeadlineAwarePriority with 500ms deadline for standard class
	policy := &DeadlineAwarePriority{
		ClassWeights:    map[string]float64{"standard": 5.0},
		Deadlines:       map[string]int64{"standard": 500_000},
		Epsilon:         0.01,
		DefaultWeight:   0.0,
		DefaultDeadline: 500_000,
	}

	req := &Request{ID: "r1", ArrivalTime: 0, SLOClass: "standard"}

	// WHEN at t=0 (fresh arrival), urgency = weight / (1.0 - 0/500000) = 5.0 / 1.0 = 5.0
	p0 := policy.Compute(req, 0)
	// WHEN at t=250000 (50% elapsed), urgency = 5.0 / (1.0 - 0.5) = 10.0
	p50 := policy.Compute(req, 250_000)
	// WHEN at t=500000 (100% elapsed), fraction=1.0, denom=0.0 < epsilon=0.01, urgency = 5.0 / 0.01 = 500.0
	p100 := policy.Compute(req, 500_000)
	// WHEN at t=750000 (150% elapsed), fraction=1.5, denom=-0.5 < epsilon=0.01, urgency = 5.0 / 0.01 = 500.0
	pPast := policy.Compute(req, 750_000)

	// THEN urgency grows monotonically with elapsed time
	if p0 >= p50 {
		t.Errorf("urgency at t=0 (%f) should be < urgency at t=50%% (%f)", p0, p50)
	}
	if p50 >= p100 {
		t.Errorf("urgency at t=50%% (%f) should be < urgency at t=100%% (%f)", p50, p100)
	}
	// At and past deadline, urgency is capped at weight/epsilon
	if p100 != pPast {
		t.Errorf("urgency at deadline (%f) should equal urgency past deadline (%f)", p100, pPast)
	}
	// Verify the cap value
	expectedCap := 5.0 / 0.01
	if p100 != expectedCap {
		t.Errorf("urgency at deadline: got %f, want %f", p100, expectedCap)
	}
}

func TestDeadlineAwarePriority_ClassOrdering(t *testing.T) {
	// GIVEN deadline-aware policy with critical > standard > sheddable weights
	// AND all requests at the same elapsed fraction (50%)
	// THEN critical > standard > sheddable in urgency
	policy := &DeadlineAwarePriority{
		ClassWeights:    map[string]float64{"critical": 10.0, "standard": 5.0, "sheddable": 1.0},
		Deadlines:       map[string]int64{"critical": 100_000, "standard": 500_000, "sheddable": 2_000_000},
		Epsilon:         0.01,
		DefaultWeight:   0.0,
		DefaultDeadline: 500_000,
	}

	// Each request is at 50% of its class deadline
	critReq := &Request{ID: "crit", ArrivalTime: 0, SLOClass: "critical"}
	stdReq := &Request{ID: "std", ArrivalTime: 0, SLOClass: "standard"}
	shedReq := &Request{ID: "shed", ArrivalTime: 0, SLOClass: "sheddable"}

	// At 50% elapsed: urgency = weight / (1.0 - 0.5) = weight * 2
	critPriority := policy.Compute(critReq, 50_000)    // 50% of 100_000
	stdPriority := policy.Compute(stdReq, 250_000)      // 50% of 500_000
	shedPriority := policy.Compute(shedReq, 1_000_000)  // 50% of 2_000_000

	if critPriority <= stdPriority {
		t.Errorf("critical urgency (%f) should be > standard (%f) at 50%% elapsed", critPriority, stdPriority)
	}
	if stdPriority <= shedPriority {
		t.Errorf("standard urgency (%f) should be > sheddable (%f) at 50%% elapsed", stdPriority, shedPriority)
	}
}

func TestDeadlineAwarePriority_StarvationCrossover(t *testing.T) {
	// S5: A sheddable request near its 2s deadline should overtake a fresh critical request
	// This is the key behavioral property that differentiates deadline-aware from static-class
	policy := &DeadlineAwarePriority{
		ClassWeights:    map[string]float64{"critical": 10.0, "standard": 5.0, "sheddable": 1.0},
		Deadlines:       map[string]int64{"critical": 100_000, "standard": 500_000, "sheddable": 2_000_000},
		Epsilon:         0.01,
		DefaultWeight:   0.0,
		DefaultDeadline: 500_000,
	}

	// At 95% of sheddable deadline (1.9M / 2M), the sheddable request's urgency
	// should overtake a fresh critical request.
	// Fresh critical: elapsed=0, urgency = 10.0 / 1.0 = 10.0
	// Old sheddable: elapsed=1.9M, fraction=0.95, denom=0.05, urgency = 1.0 / 0.05 = 20.0
	clock := int64(1_900_000)
	freshCrit := &Request{ID: "fresh-crit", ArrivalTime: 1_900_000, SLOClass: "critical"}
	oldShed := &Request{ID: "old-shed", ArrivalTime: 0, SLOClass: "sheddable"}

	critPriority := policy.Compute(freshCrit, clock)
	shedPriority := policy.Compute(oldShed, clock)

	// Fresh critical: elapsed=0, urgency = 10.0 / 1.0 = 10.0
	// Old sheddable: elapsed=1.9M, fraction=0.95, denom=0.05, urgency = 1.0 / 0.05 = 20.0
	if shedPriority <= critPriority {
		t.Errorf("starvation crossover: old sheddable urgency (%f) should overtake fresh critical (%f)",
			shedPriority, critPriority)
	}
}

// --- NewPriorityPolicyFromConfig tests ---

func TestNewPriorityPolicyFromConfig_DeadlineAware_CustomParams(t *testing.T) {
	// GIVEN a PriorityConfig with custom deadline-aware parameters
	eps := 0.05
	cfg := PriorityConfig{
		Policy:       "deadline-aware",
		ClassWeights: map[string]float64{"critical": 20.0, "standard": 8.0},
		Deadlines:    map[string]int64{"critical": 50_000, "standard": 250_000},
		Epsilon:      &eps,
	}

	policy := NewPriorityPolicyFromConfig(cfg)

	// THEN the policy uses custom weights: critical request gets priority = 20.0 / 1.0 = 20.0 at t=0
	critReq := &Request{ID: "crit", ArrivalTime: 0, SLOClass: "critical"}
	got := policy.Compute(critReq, 0)
	if got != 20.0 {
		t.Errorf("expected critical priority 20.0 at t=0, got %f", got)
	}

	// THEN standard request gets 8.0 / 1.0 = 8.0 at t=0
	stdReq := &Request{ID: "std", ArrivalTime: 0, SLOClass: "standard"}
	got = policy.Compute(stdReq, 0)
	if got != 8.0 {
		t.Errorf("expected standard priority 8.0 at t=0, got %f", got)
	}

	// THEN at deadline, urgency caps at weight/epsilon = 20.0/0.05 = 400.0
	got = policy.Compute(critReq, 50_000) // 100% of critical deadline
	expected := 20.0 / 0.05
	if got != expected {
		t.Errorf("expected critical urgency at deadline = %f, got %f", expected, got)
	}
}

func TestNewPriorityPolicyFromConfig_DeadlineAware_Defaults(t *testing.T) {
	// GIVEN a PriorityConfig with only policy name (no custom params)
	cfg := PriorityConfig{Policy: "deadline-aware"}

	policy := NewPriorityPolicyFromConfig(cfg)

	// THEN it uses default weights (same as NewPriorityPolicy("deadline-aware"))
	critReq := &Request{ID: "crit", ArrivalTime: 0, SLOClass: "critical"}
	got := policy.Compute(critReq, 0)
	if got != 10.0 {
		t.Errorf("expected default critical priority 10.0 at t=0, got %f", got)
	}
}

func TestNewPriorityPolicyFromConfig_StaticClassWeight_CustomParams(t *testing.T) {
	// GIVEN a PriorityConfig with custom class weights
	cfg := PriorityConfig{
		Policy:       "static-class-weight",
		ClassWeights: map[string]float64{"critical": 100.0, "batch": 0.5},
	}

	policy := NewPriorityPolicyFromConfig(cfg)

	// THEN custom weights are used
	critReq := &Request{ID: "crit", ArrivalTime: 0, SLOClass: "critical"}
	batchReq := &Request{ID: "batch", ArrivalTime: 0, SLOClass: "batch"}
	unknownReq := &Request{ID: "unk", ArrivalTime: 0, SLOClass: "unknown"}

	if got := policy.Compute(critReq, 0); got != 100.0 {
		t.Errorf("expected critical=100.0, got %f", got)
	}
	if got := policy.Compute(batchReq, 0); got != 0.5 {
		t.Errorf("expected batch=0.5, got %f", got)
	}
	// Unknown class gets default weight = 0.0
	if got := policy.Compute(unknownReq, 0); got != 0.0 {
		t.Errorf("expected unknown=0.0, got %f", got)
	}
}

func TestNewPriorityPolicyFromConfig_FallsBackToNameFactory(t *testing.T) {
	// GIVEN configs for policies that don't use extended params
	tests := []struct {
		name   string
		config PriorityConfig
	}{
		{"constant", PriorityConfig{Policy: "constant"}},
		{"slo-based", PriorityConfig{Policy: "slo-based"}},
		{"inverted-slo", PriorityConfig{Policy: "inverted-slo"}},
		{"empty", PriorityConfig{Policy: ""}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			policy := NewPriorityPolicyFromConfig(tc.config)
			if policy == nil {
				t.Fatal("expected non-nil policy")
			}
			// Just verify it doesn't panic and returns a valid policy
			req := &Request{ID: "r1", ArrivalTime: 0}
			_ = policy.Compute(req, 1000)
		})
	}
}

func TestNewPriorityPolicyFromConfig_EpsilonZero(t *testing.T) {
	// R9: zero epsilon is a valid value (not "unset")
	eps := 0.0
	cfg := PriorityConfig{
		Policy:  "deadline-aware",
		Epsilon: &eps,
	}

	policy := NewPriorityPolicyFromConfig(cfg)

	// With epsilon=0, at exactly the deadline, denom=0.0, eps=0.0:
	// the clamp condition (denom < eps) is false (0.0 < 0.0 is false),
	// so urgency = weight / 0.0 = +Inf.
	// This confirms R9: zero epsilon is used (not treated as "unset" = 0.01).
	req := &Request{ID: "r1", ArrivalTime: 0, SLOClass: "critical"}
	got := policy.Compute(req, 100_000) // exactly at default critical deadline
	if !math.IsInf(got, 1) {
		t.Errorf("expected +Inf urgency with epsilon=0 at deadline, got %f", got)
	}
}

func TestDeadlineAwarePriority_INV9_DoesNotReadOutputTokens(t *testing.T) {
	// INV-9: Priority policies must not read Request.OutputTokens
	// Verify that different OutputTokens produce the same priority
	policy := &DeadlineAwarePriority{
		ClassWeights:    map[string]float64{"critical": 10.0},
		Deadlines:       map[string]int64{"critical": 100_000},
		Epsilon:         0.01,
		DefaultWeight:   0.0,
		DefaultDeadline: 500_000,
	}

	req1 := &Request{ID: "r1", ArrivalTime: 0, SLOClass: "critical", OutputTokens: make([]int, 10)}
	req2 := &Request{ID: "r2", ArrivalTime: 0, SLOClass: "critical", OutputTokens: make([]int, 1000)}
	req3 := &Request{ID: "r3", ArrivalTime: 0, SLOClass: "critical", OutputTokens: nil}

	clock := int64(50_000)
	p1 := policy.Compute(req1, clock)
	p2 := policy.Compute(req2, clock)
	p3 := policy.Compute(req3, clock)

	if p1 != p2 || p2 != p3 {
		t.Errorf("INV-9 violation: priority depends on OutputTokens: %f, %f, %f", p1, p2, p3)
	}
}
