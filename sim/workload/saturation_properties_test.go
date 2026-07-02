// sim/workload/saturation_properties_test.go
//
// Property tests for BacklogClassifier implementations (#1394). Each test runs
// against EVERY classifier registered via NewBacklogClassifier — new classifier
// implementations automatically inherit this suite without modification.
//
// Six properties (each one trace below failed for the pre-fix slope-based
// classifier and motivated this PR):
//
//   P1 — Determinism: Classify(x) == Classify(x) on a fresh run (no hidden state).
//   P2 — Monotonicity in true ρ: at ρ extremes, all classifiers return the
//        expected category (UNSATURATED for ρ ≤ 0.85; PERSISTENTLY_SATURATED
//        for ρ ≥ 1.10). Borderline (0.95 ≤ ρ ≤ 1.05) is unconstrained.
//   P3 — Drain-phase invariance: same inject phase + 0/30s/60s drain → same
//        classification. The drain phase is not part of saturation.
//   P4 — Warmup-phase robustness: a warmup ramp does not change steady-state
//        verdict at ρ extremes.
//   P5 — Conservation (RequestsToIntervals): every injected request appears in
//        the interval set unless it timed out. This is the property that
//        catches #1389; classifier-agnostic.
//   P6 — Cross-classifier agreement: at ρ extremes, all registered classifiers
//        return the same classification string.

package workload

import (
	"fmt"
	"testing"
	"time"

	sim "blis/sim"
)

// ────────────────────────────────────────────────────────────────────────────
// SyntheticTrace — deterministic generator of *sim.Request slices with a known
// utilization ρ = λ/μ. Used as the input distribution for property tests.

// SyntheticTrace describes a workload with a controllable utilization regime.
// Build() returns a slice of *sim.Request and the simEndUs to feed into
// AnalyzeBacklogDriftWithClassifier.
type SyntheticTrace struct {
	LambdaPerSec   float64       // arrival rate during inject phase
	MuPerSec       float64       // engine service rate (completions/sec)
	InjectDuration time.Duration // length of inject phase
	DrainDuration  time.Duration // length of drain phase (0 = no drain)
	WarmupDuration time.Duration // ramp-up at start of inject (linearly grows from 0 to λ)
	Seed           int64
}

// Build constructs the request slice and simEndUs.
//
// The model is intentionally simple:
//   - Arrivals are sampled at jittered intervals averaging 1/λ during the
//     inject phase. Warmup ramps the rate linearly from 0 to λ.
//   - Each request's TTFT is set to a deterministic constant (1ms) for
//     completed requests; service time = 1/μ on average.
//   - At each instant during the simulation, completions are scheduled to
//     drain backlog at rate μ. If λ > μ, queue grows — late arrivals end up
//     queued at horizon, with State = StateQueued.
//   - simEndUs = inject + drain duration.
//
// Determinism: identical fields → identical output. (Seed is reserved for future
// extensions such as Poisson-jittered traces; the baseline implementation uses
// constant-rate spacing so the slope-based and drain-ratio classifiers see the
// same low-variance signal.)
func (st SyntheticTrace) Build() (requests []*sim.Request, simEndUs int64, expectedRho float64) {
	expectedRho = st.LambdaPerSec / st.MuPerSec

	injectUs := int64(st.InjectDuration / time.Microsecond)
	drainUs := int64(st.DrainDuration / time.Microsecond)
	warmupUs := int64(st.WarmupDuration / time.Microsecond)
	simEndUs = injectUs + drainUs

	const oneMs = int64(1_000)
	const ttftUs = oneMs
	muIATUs := int64(1e6 / st.MuPerSec) // service time in µs

	// Constant-rate (deterministic) arrival generation. During warmup, instantaneous
	// rate ramps linearly from 0 to λ; outside warmup, rate = λ. Walk forward in
	// time, computing the next arrival's gap from the instantaneous rate.
	t := int64(0)
	arrivals := []int64{}
	for t < injectUs {
		rate := st.LambdaPerSec
		if warmupUs > 0 && t < warmupUs {
			rate = st.LambdaPerSec * float64(t) / float64(warmupUs)
			if rate < 1e-3 {
				rate = 1e-3
			}
		}
		gapUs := int64(1e6 / rate)
		t += gapUs
		if t >= injectUs {
			break
		}
		arrivals = append(arrivals, t)
	}

	// Assign completion/queued state. The engine processes at μ, so a request
	// scheduled to finish before simEndUs is "completed"; the rest are "queued".
	// For simplicity, FIFO completion: the i-th arrival completes at arrival[i] +
	// queueing_delay + service_time, where queueing_delay reflects backlog at arrival.
	requests = make([]*sim.Request, 0, len(arrivals))
	nextFreeUs := int64(0) // earliest time the engine can start the next request
	for _, arrUs := range arrivals {
		startUs := arrUs
		if nextFreeUs > startUs {
			startUs = nextFreeUs
		}
		completionUs := startUs + muIATUs
		nextFreeUs = startUs + muIATUs

		req := &sim.Request{
			ArrivalTime: arrUs,
		}
		if completionUs <= simEndUs {
			// Completed. RequestsToIntervals derives completion as
			//   ArrivalTime + FirstTokenTime + Σ ITL.
			// To make that equal startUs + muIATUs, we encode queueing delay
			// in FirstTokenTime: FirstTokenTime = (startUs - arrUs) + ttftUs.
			req.TTFTSet = true
			req.FirstTokenTime = (startUs - arrUs) + ttftUs
			body := muIATUs - ttftUs
			if body < 0 {
				body = 0
			}
			req.ITL = []int64{body}
			req.State = sim.StateCompleted
		} else if startUs < simEndUs {
			// Started but not finished by simEnd — Running
			req.TTFTSet = false
			req.State = sim.StateRunning
		} else {
			// Never started — Queued
			req.TTFTSet = false
			req.State = sim.StateQueued
		}
		requests = append(requests, req)
	}

	return requests, simEndUs, expectedRho
}

// ────────────────────────────────────────────────────────────────────────────
// allClassifiers returns every BacklogClassifier registered in the factory.
// Used by property tests so adding a new classifier auto-inherits the suite.

func allClassifiers(t *testing.T) map[string]BacklogClassifier {
	t.Helper()
	out := map[string]BacklogClassifier{}
	for _, name := range sim.ValidBacklogClassifierNames() {
		c := NewBacklogClassifier(name)
		if c == nil {
			t.Fatalf("NewBacklogClassifier(%q) returned nil", name)
		}
		out[name] = c
	}
	if len(out) == 0 {
		t.Fatal("No registered classifiers — registry is empty?")
	}
	return out
}

// ────────────────────────────────────────────────────────────────────────────
// P1: Determinism

func TestProperty_P1_Determinism(t *testing.T) {
	classifiers := allClassifiers(t)
	trace := SyntheticTrace{
		LambdaPerSec:   80,
		MuPerSec:       100,
		InjectDuration: 60 * time.Second,
		DrainDuration:  10 * time.Second,
		Seed:           42,
	}
	reqs, simEnd, _ := trace.Build()
	cfg := DefaultBacklogDriftConfig()
	cfg.WindowSize = 5 * time.Second
	cfg.MinWindows = 3

	for name, c := range classifiers {
		t.Run(name, func(t *testing.T) {
			r1 := AnalyzeBacklogDriftWithClassifier(reqs, simEnd, cfg, c)
			r2 := AnalyzeBacklogDriftWithClassifier(reqs, simEnd, cfg, c)
			if r1.Classification != r2.Classification {
				t.Errorf("Non-deterministic classification: %s vs %s", r1.Classification, r2.Classification)
			}
			if r1.Note != r2.Note {
				t.Errorf("Non-deterministic note for %s", name)
			}
		})
	}
}

// ────────────────────────────────────────────────────────────────────────────
// P2: Monotonicity in true ρ at extremes

func TestProperty_P2_MonotonicityExtremes(t *testing.T) {
	classifiers := allClassifiers(t)
	cfg := DefaultBacklogDriftConfig()
	cfg.WindowSize = 5 * time.Second
	cfg.MinWindows = 3

	cases := []struct {
		name        string
		lambda, mu  float64
		expectClass string
	}{
		{"deeply_unsaturated_rho_0.5", 50, 100, "UNSATURATED"},
		{"unsaturated_rho_0.85", 85, 100, "UNSATURATED"},
		{"saturated_rho_1.15", 115, 100, "PERSISTENTLY_SATURATED"},
		{"deeply_saturated_rho_1.5", 150, 100, "PERSISTENTLY_SATURATED"},
	}

	for _, tc := range cases {
		trace := SyntheticTrace{
			LambdaPerSec:   tc.lambda,
			MuPerSec:       tc.mu,
			InjectDuration: 60 * time.Second,
			DrainDuration:  20 * time.Second,
			Seed:           42,
		}
		reqs, simEnd, rho := trace.Build()
		_ = rho // for debug

		for name, c := range classifiers {
			t.Run(fmt.Sprintf("%s/%s", tc.name, name), func(t *testing.T) {
				report := AnalyzeBacklogDriftWithClassifier(reqs, simEnd, cfg, c)
				if report.Classification != tc.expectClass {
					t.Errorf("classifier=%s ρ=%.2f: expected %s, got %s. Note: %s",
						name, tc.lambda/tc.mu, tc.expectClass, report.Classification, report.Note)
				}
			})
		}
	}
}

// ────────────────────────────────────────────────────────────────────────────
// P3: Drain-phase invariance

func TestProperty_P3_DrainPhaseInvariance(t *testing.T) {
	classifiers := allClassifiers(t)
	cfg := DefaultBacklogDriftConfig()
	cfg.WindowSize = 5 * time.Second
	cfg.MinWindows = 3

	// Saturated case — verify that drain length doesn't flip the classification
	// (the historical bug was tent-shape masking).
	for _, drain := range []time.Duration{0, 30 * time.Second, 60 * time.Second} {
		trace := SyntheticTrace{
			LambdaPerSec:   115,
			MuPerSec:       100,
			InjectDuration: 60 * time.Second,
			DrainDuration:  drain,
			Seed:           42,
		}
		reqs, simEnd, _ := trace.Build()

		for name, c := range classifiers {
			t.Run(fmt.Sprintf("drain=%s/%s", drain, name), func(t *testing.T) {
				report := AnalyzeBacklogDriftWithClassifier(reqs, simEnd, cfg, c)
				if report.Classification != "PERSISTENTLY_SATURATED" {
					t.Errorf("classifier=%s drain=%s: drain length should not change classification, got %s. Note: %s",
						name, drain, report.Classification, report.Note)
				}
			})
		}
	}
}

// ────────────────────────────────────────────────────────────────────────────
// P4: Warmup-phase robustness

func TestProperty_P4_WarmupRobustness(t *testing.T) {
	classifiers := allClassifiers(t)
	cfg := DefaultBacklogDriftConfig()
	cfg.WindowSize = 5 * time.Second
	cfg.MinWindows = 3

	// At ρ=0.5 (clearly unsaturated), neither warmup nor abrupt-start should change verdict.
	// Same trace, different warmup durations.
	for _, warmup := range []time.Duration{0, 10 * time.Second} {
		trace := SyntheticTrace{
			LambdaPerSec:   50,
			MuPerSec:       100,
			InjectDuration: 60 * time.Second,
			DrainDuration:  10 * time.Second,
			WarmupDuration: warmup,
			Seed:           42,
		}
		reqs, simEnd, _ := trace.Build()

		for name, c := range classifiers {
			t.Run(fmt.Sprintf("warmup=%s/%s", warmup, name), func(t *testing.T) {
				report := AnalyzeBacklogDriftWithClassifier(reqs, simEnd, cfg, c)
				if report.Classification != "UNSATURATED" {
					t.Errorf("classifier=%s warmup=%s: should remain UNSATURATED at ρ=0.5, got %s",
						name, warmup, report.Classification)
				}
			})
		}
	}
}

// ────────────────────────────────────────────────────────────────────────────
// P5: Conservation — every injected request shows up in the interval set
//     unless it timed out. This is the invariant that catches #1389.
//     Classifier-agnostic.

func TestProperty_P5_Conservation(t *testing.T) {
	cases := []struct {
		name           string
		completed      int
		running        int
		queued         int
		timedOut       int
	}{
		{"only_completed", 100, 0, 0, 0},
		{"completed_plus_running", 80, 20, 0, 0},
		{"completed_plus_queued", 50, 0, 50, 0},
		{"all_states_mixed", 40, 20, 30, 10},
		{"only_timed_out", 0, 0, 0, 50},
		{"all_queued", 0, 0, 100, 0},
	}

	const simEndUs = int64(60_000_000)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reqs := []*sim.Request{}
			for i := 0; i < tc.completed; i++ {
				reqs = append(reqs, &sim.Request{
					ArrivalTime: int64(i * 100), TTFTSet: true,
					FirstTokenTime: 1000, ITL: []int64{1000}, State: sim.StateCompleted,
				})
			}
			for i := 0; i < tc.running; i++ {
				reqs = append(reqs, &sim.Request{
					ArrivalTime: int64(50_000_000 + i*100), TTFTSet: false, State: sim.StateRunning,
				})
			}
			for i := 0; i < tc.queued; i++ {
				reqs = append(reqs, &sim.Request{
					ArrivalTime: int64(55_000_000 + i*100), TTFTSet: false, State: sim.StateQueued,
				})
			}
			for i := 0; i < tc.timedOut; i++ {
				reqs = append(reqs, &sim.Request{
					ArrivalTime: int64(i * 100), TTFTSet: false, State: sim.StateTimedOut,
				})
			}
			intervals := RequestsToIntervals(reqs, simEndUs)
			expected := tc.completed + tc.running + tc.queued
			if len(intervals) != expected {
				t.Errorf("Conservation violation: got %d intervals, want %d (timed_out=%d excluded)",
					len(intervals), expected, tc.timedOut)
			}
		})
	}
}

// ────────────────────────────────────────────────────────────────────────────
// P6: Cross-classifier agreement at ρ extremes

func TestProperty_P6_CrossClassifierAgreement(t *testing.T) {
	classifiers := allClassifiers(t)
	if len(classifiers) < 2 {
		t.Skip("Need ≥ 2 registered classifiers for cross-agreement test")
	}
	cfg := DefaultBacklogDriftConfig()
	cfg.WindowSize = 5 * time.Second
	cfg.MinWindows = 3

	// At ρ=0.5 and ρ=1.5, every classifier should agree on the verdict.
	for _, lambda := range []float64{50, 150} {
		trace := SyntheticTrace{
			LambdaPerSec:   lambda,
			MuPerSec:       100,
			InjectDuration: 60 * time.Second,
			DrainDuration:  20 * time.Second,
			Seed:           42,
		}
		reqs, simEnd, _ := trace.Build()

		verdicts := map[string]string{}
		for name, c := range classifiers {
			r := AnalyzeBacklogDriftWithClassifier(reqs, simEnd, cfg, c)
			verdicts[name] = r.Classification
		}
		// Check all verdicts equal
		var first string
		for _, v := range verdicts {
			first = v
			break
		}
		for name, v := range verdicts {
			if v != first {
				t.Errorf("ρ=%.2f: classifiers disagree at extremes — %v", lambda/100.0, verdicts)
				_ = name
				break
			}
		}
	}
}
