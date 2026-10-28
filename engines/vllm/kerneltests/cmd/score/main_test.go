// Moved from blis-latency-kernel/cmd/score/main_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package main

import (
	"math"
	"testing"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-vllm/kerneltests/internal/artifacttest"
	"github.com/inference-sim/blis-vllm/kerneltests/internal/harness"
)

// The scorer decides which modelling changes ship. A roofline fit that looked better than
// everything else on the component sweeps was rejected because this harness said it made the
// end-to-end error worse — so a broken harness would have let it through, and every later
// decision would rest on it.
//
// These tests check the two things a harness can get wrong without any symptom: admitting a
// point it should not, and computing a prediction from the wrong inputs.

func f(v float64) *float64 { return &v }

// --- Scope: which points the harness agrees to score ---------------------------

func TestPreemptedPointsAreOutOfScope(t *testing.T) {
	// A preempted request's recompute arrives as ordinary scheduled tokens the simulator
	// owns, so measured ITL there is a scheduler outcome and not a step time.
	r := resolve(point{Concurrency: 1024, ITLms: 254, ISL: f(8015), OSL: f(486),
		Preempted: f(45)}, 1)
	if r.inScope {
		t.Error("a point with 45 preemptions was accepted for scoring")
	}
	if r.why == "" {
		t.Error("an out-of-scope point must say why")
	}
}

func TestQueuedPointsAreOutOfScope(t *testing.T) {
	// A non-empty queue means prefill chunks share decode steps.
	r := resolve(point{Concurrency: 128, ITLms: 78, Running: f(60), Waiting: f(9.75),
		InTPS: f(1000), OutTPS: f(10), EffConc: f(60), E2Ems: f(5000)}, 1)
	if r.inScope {
		t.Error("a point with 9.75 queued requests was accepted")
	}
}

func TestHighTTFTWithoutAStatedBatchIsOutOfScope(t *testing.T) {
	// Without a stated resident batch, client concurrency stands in for it — which holds
	// only while nothing queues. A TTFT well past its flat band says otherwise.
	r := resolve(point{Concurrency: 256, ITLms: 84, ISL: f(8015), OSL: f(489),
		Preempted: f(0), TTFTms: f(996)}, 1)
	if r.inScope {
		t.Error("a point with 996 ms TTFT and no stated batch was accepted")
	}
}

func TestAPointWithNoDerivableContextIsOutOfScope(t *testing.T) {
	r := resolve(point{Concurrency: 8, ITLms: 6.7, Running: f(1.1), Waiting: f(0),
		Preempted: f(0)}, 1)
	if r.inScope {
		t.Error("a point with no context stated or derivable was accepted")
	}
}

func TestACleanPointIsInScope(t *testing.T) {
	// The negative tests above prove nothing unless something passes.
	r := resolve(point{Concurrency: 1, ITLms: 6.02, ISL: f(8015), OSL: f(484),
		Preempted: f(0), TTFTms: f(41.83)}, 1)
	if !r.inScope {
		t.Errorf("a clean low-concurrency point was rejected: %s", r.why)
	}
	if r.context != 8015+484/2 {
		t.Errorf("context resolved to %d, expected input plus half the output", r.context)
	}
	if !r.contextStated {
		t.Error("a stated input length must be recorded as stated")
	}
}

// --- Derivation: the batch and the context ------------------------------------

func TestAStatedResidentBatchIsPreferredOverConcurrency(t *testing.T) {
	// `running` is the engine's own count; client concurrency is an upper bound on it.
	// Preferring the wrong one silently prices a step that never ran.
	r := resolve(point{Concurrency: 128, ITLms: 18, Running: f(21.1), Waiting: f(0),
		Preempted: f(0), InTPS: f(1030), OutTPS: f(10), EffConc: f(21.1),
		E2Ems: f(10381)}, 3)
	if !r.batchStated {
		t.Error("a stated resident batch was not used")
	}
	// Summed across three replicas, so a per-step batch is a third of it.
	if r.batch != 7 {
		t.Errorf("batch resolved to %d; 21.1 running over 3 replicas is 7", r.batch)
	}
	// And it must differ from what client concurrency would have given, or this test
	// passes whether or not `running` was consulted. Concurrency 128 over 3 replicas is
	// 42; the engine held 21.1, so 7. A harness that ignored `running` would score a step
	// six times larger than the one that ran, and nothing else here would notice.
	fromConcurrency := 128 / 3
	if r.batch == fromConcurrency {
		t.Errorf("batch %d equals what client concurrency alone gives; the stated "+
			"resident batch is not being preferred", r.batch)
	}
}

func TestEffectiveConcurrencyCountsAsAStatedBatch(t *testing.T) {
	// "Requests genuinely in flight" is the resident batch by another name, and unlike the
	// client level it already excludes what is waiting.
	r := resolve(point{Concurrency: 32, ITLms: 135, EffConc: f(14.2), Preempted: f(0),
		InTPS: f(11469), OutTPS: f(89.8), E2Ems: f(41549)}, 4)
	if !r.batchStated {
		t.Error("effective concurrency was not treated as a stated batch")
	}
	if r.batch != 4 {
		t.Errorf("batch resolved to %d; 14.2 in flight over 4 replicas is 4", r.batch)
	}
}

func TestContextIsDerivedFromRatesWhenNotStated(t *testing.T) {
	// Neither length is given, but both follow from rates the report does give — without
	// assuming either. This is what let the Nemotron and Kimi arms be scored at all.
	//
	// 256 out-tok/s over 8 in flight is 32 tok/s per request; over 6238 ms that is 200
	// output tokens; the 103:1 input:output rate ratio makes the input 20,600.
	r := resolve(point{Concurrency: 32, ITLms: 11.76, Running: f(9.5), Waiting: f(0.41),
		Preempted: f(0), InTPS: f(26400), OutTPS: f(256), EffConc: f(8),
		E2Ems: f(6238)}, 3)
	if r.contextStated {
		t.Error("a derived context must not be recorded as stated")
	}
	if r.context < 5000 || r.context > 40000 {
		t.Errorf("derived context %d is outside the range these rates imply", r.context)
	}
	// And the derivation must be marked in the note, so a reader of the table knows.
	if r.why == "" {
		t.Error("a derived input must be marked")
	}
}

func TestDerivedContextTracksTheInputOutputRatio(t *testing.T) {
	// The whole point of the derivation is that it follows the data. A ten-fold longer
	// prompt at the same output rate must give a ten-fold longer context.
	short := resolve(point{Concurrency: 8, ITLms: 7, Running: f(1), Waiting: f(0),
		Preempted: f(0), InTPS: f(1000), OutTPS: f(100), EffConc: f(1),
		E2Ems: f(3000)}, 1)
	long := resolve(point{Concurrency: 8, ITLms: 7, Running: f(1), Waiting: f(0),
		Preempted: f(0), InTPS: f(10000), OutTPS: f(100), EffConc: f(1),
		E2Ems: f(3000)}, 1)
	if ratio := float64(long.context) / float64(short.context); ratio < 8 || ratio > 11 {
		t.Errorf("a 10x input:output ratio gave a %.1fx context change", ratio)
	}
}

// --- Prediction ---------------------------------------------------------------

func TestSpeculationDividesTheStepByAcceptedTokens(t *testing.T) {
	// An accepted draft token costs no extra step, so a step serves more than one token per
	// request and the observed interval falls below the step time. Omitting this would make
	// every speculative arm look slow.
	k := scoreFixture(t, "minimax-m25-h200-tp8.yaml")
	if k == nil {
		return
	}
	base := resolved{point: point{ITLms: 10}, batch: 8, context: 4096, replicas: 1}
	spec := base
	spec.AcceptPct = f(90)
	plain, withSpec := predict(k, base), predict(k, spec)
	if withSpec >= plain {
		t.Errorf("90%% acceptance did not lower the interval: %.3f against %.3f",
			withSpec, plain)
	}
	// 90% acceptance means 1.9 tokens per step on average.
	if ratio := plain / withSpec; math.Abs(ratio-1.9) > 0.01 {
		t.Errorf("interval fell by %.3fx; 90%% acceptance gives 1.9 tokens per step",
			ratio)
	}
}

func TestPredictionIncludesThePerTokenHostCost(t *testing.T) {
	// The output processor runs off the forward pass but its cost still lands on the
	// interval between tokens, so a prediction that omitted it would understate every point.
	k := scoreFixture(t, "minimax-m25-h200-tp8.yaml")
	if k == nil {
		return
	}
	r := resolved{point: point{ITLms: 10}, batch: 1, context: 1024, replicas: 1}
	// NoOverlap, matching predict(): that is the edge the measured evidence selects
	// (blis-registry docs/band-selection.md). This test checks the HOST term is included,
	// so it must agree with the production path on which edge it starts from.
	step := k.StepTime(harness.DecodeBatch(r.batch, r.context)).NoOverlap.Seconds() * 1e3
	got := predict(k, r)
	host := k.OutputTokenOverhead().Seconds() * 1e3
	if math.Abs(got-(step+host)) > 1e-9 {
		t.Errorf("prediction %.6f is not the step %.6f plus the per-token host cost %.6f",
			got, step, host)
	}
}

func TestBatchForBuildsOneRequestPerResidentRequest(t *testing.T) {
	k := scoreFixture(t, "minimax-m25-h200-tp8.yaml")
	if k == nil {
		return
	}
	r := resolved{point: point{}, batch: 37, context: 2048, replicas: 1}
	b := harness.DecodeBatch(r.batch, r.context)
	if len(b.Reqs) != 37 {
		t.Errorf("built %d requests for a batch of 37", len(b.Reqs))
	}
	for _, req := range b.Reqs {
		if req.Computed+req.Scheduled != 2048 {
			t.Errorf("a request carries context %d, expected 2048",
				req.Computed+req.Scheduled)
		}
	}
}

// scoreFixture builds a kernel from the sibling repositories, or skips.
func scoreFixture(t *testing.T, scenario string) *latencykernel.Kernel {
	t.Helper()
	k, err := harness.Open(scenario, harness.Repos{
		Scenarios: "../../testdata",
		Catalog:   harness.DefaultCatalog(),
		Registry:  harness.DefaultRegistry(),
	})
	artifacttest.RequireArtifact(t, harness.DefaultCatalog(),
		harness.DefaultCatalog(), "catalog", err)
	return k
}
