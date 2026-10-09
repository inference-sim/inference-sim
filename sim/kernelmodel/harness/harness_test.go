package harness

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/kernelmodel/internal/artifacts"
)

// corpusPath is the AISimulate end-to-end corpus, under BLIS_MEASUREMENTS.
func corpusPath(t testing.TB) string { return artifacts.Measurement(t, "aisimulate_e2e.json") }

func cfg() Config {
	return Config{
		Repos:            kernelmodel.DefaultRepos(),
		Admission:        AdmissionKernelKV,
		SessionsPerPoint: 24,
		Seed:             42,
	}
}

func corpus(t *testing.T) *Corpus {
	t.Helper()
	c, err := LoadCorpus(corpusPath(t))
	if err != nil {
		t.Fatalf("LoadCorpus: %v", err)
	}
	return c
}

// A run must actually complete requests and observe inter-token latency. A harness that
// produced a zero mean would make every MAPE meaningless, and that is exactly what happened
// when requests entered through EnqueueRequest instead of InjectArrival: 300 simulated
// seconds, zero completions, and a score that looked like a number.
func TestARunCompletesRequestsAndObservesLatency(t *testing.T) {
	c := corpus(t)
	sw := c.Sweeps[0]
	obs, err := Run(sw, sw.Points[0].Concurrency, cfg())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if obs.Completed == 0 {
		t.Fatal("no requests completed, so no inter-token latency was observed and any " +
			"score derived from this run would be vacuous")
	}
	if obs.MeanITLUs <= 0 {
		t.Fatalf("%d requests completed but mean ITL is %v", obs.Completed, obs.MeanITLUs)
	}
	if obs.KVBlocks <= 0 {
		t.Errorf("KV budget is %d blocks", obs.KVBlocks)
	}
}

// Mean inter-token latency must rise with client concurrency at a fixed deployment. This is
// the property the entire comparison rests on: a flat curve would give a meaningless shape
// score, and a falling one would mean the closed loop is not applying load.
func TestInterTokenLatencyRisesWithConcurrency(t *testing.T) {
	c := corpus(t)
	sw := c.Sweeps[0]
	var prev float64
	for _, p := range sw.Points {
		obs, err := Run(sw, p.Concurrency, cfg())
		if err != nil {
			t.Fatalf("c=%d: %v", p.Concurrency, err)
		}
		if obs.MeanITLUs < prev {
			t.Errorf("concurrency %d gave mean ITL %.0f us, below the %.0f at the lower "+
				"level; the closed loop is not applying more load",
				p.Concurrency, obs.MeanITLUs, prev)
		}
		prev = obs.MeanITLUs
	}
}

// A run must be deterministic at a fixed seed. Without it, a MAPE difference between two
// runs could be noise rather than a modelling change, and no conclusion would be defensible.
func TestARunIsDeterministicAtAFixedSeed(t *testing.T) {
	c := corpus(t)
	sw := c.Sweeps[0]
	a, err := Run(sw, 8, cfg())
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	b, err := Run(sw, 8, cfg())
	if err != nil {
		t.Fatalf("second run: %v", err)
	}
	if a.MeanITLUs != b.MeanITLUs {
		t.Errorf("two runs at seed %d gave %.6f and %.6f us", cfg().Seed,
			a.MeanITLUs, b.MeanITLUs)
	}
	if a.Completed != b.Completed {
		t.Errorf("two runs completed %d and %d requests", a.Completed, b.Completed)
	}
}

// The workload must carry the sweep's stated ISL and OSL. A harness that used its own
// lengths would be measuring a different workload than the one the snapshot measured, and
// the comparison would not be apples to apples however good the number looked.
//
// Behavioural: it runs two sweeps whose ONLY difference is the workload identity and asserts
// the observed latencies differ. 1k1k and 8k1k on one deployment must not agree.
func TestTheSweepsWorkloadReachesTheSimulation(t *testing.T) {
	c := corpus(t)
	byWorkload := map[string]Sweep{}
	for _, sw := range c.Sweeps {
		if sw.Scenario == "gpt-oss-120b-h200-fp4-vllm-tp4.yaml" {
			byWorkload[sw.Label] = sw
		}
	}
	short, okShort := byWorkload["1k1k"]
	long, okLong := byWorkload["8k1k"]
	if !okShort || !okLong {
		t.Skip("this deployment does not carry both workloads in the corpus")
	}
	const concurrency = 8
	a, err := Run(short, concurrency, cfg())
	if err != nil {
		t.Fatalf("1k1k: %v", err)
	}
	b, err := Run(long, concurrency, cfg())
	if err != nil {
		t.Fatalf("8k1k: %v", err)
	}
	if a.MeanITLUs == b.MeanITLUs {
		t.Errorf("a 1024-token prompt and an 8192-token prompt both gave %.0f us; the "+
			"sweep's workload is not reaching the simulation", a.MeanITLUs)
	}
	// An 8192-token prompt costs more prefill work per request, which at equal concurrency
	// raises the time between output tokens.
	if b.MeanITLUs <= a.MeanITLUs {
		t.Errorf("8k1k gave %.0f us, not above 1k1k's %.0f", b.MeanITLUs, a.MeanITLUs)
	}
}

// Every sweep in the corpus must name a workload this harness can parse. A silent parse
// failure would drop points from the score without saying so.
func TestEverySweepStatesAParsableWorkload(t *testing.T) {
	for _, sw := range corpus(t).Sweeps {
		isl, osl, err := sw.ISLOSL()
		if err != nil {
			t.Errorf("%s %s: %v", sw.Scenario, sw.Label, err)
			continue
		}
		if isl <= 0 || osl <= 0 {
			t.Errorf("%s %s: isl=%d osl=%d", sw.Scenario, sw.Label, isl, osl)
		}
	}
}

// MAPE and Median must behave as their names claim. Cheap to check and load-bearing: the
// headline figures are these two functions applied to per-point errors.
func TestTheErrorStatisticsAreWhatTheyClaim(t *testing.T) {
	if got := MAPE([]float64{110, 90}, []float64{100, 100}); math.Abs(got-10) > 1e-9 {
		t.Errorf("MAPE of +10%% and -10%% is %v, expected 10", got)
	}
	if got := MAPE2([]float64{10, 20, 30}); math.Abs(got-20) > 1e-9 {
		t.Errorf("MAPE2 of 10,20,30 is %v, expected 20", got)
	}
	if got := Median([]float64{3, 1, 2}); got != 2 {
		t.Errorf("median of 3,1,2 is %v, expected 2", got)
	}
	if got := Median([]float64{4, 1, 3, 2}); got != 2.5 {
		t.Errorf("median of 1,2,3,4 is %v, expected 2.5", got)
	}
	// An empty input must be NaN rather than 0: a zero would read as a perfect score.
	if !math.IsNaN(MAPE2(nil)) {
		t.Error("MAPE2 of nothing must be NaN, not a number that reads as perfect")
	}
}

// The measured mean must not depend on how many requests the run happened to complete. That
// dependence is harness noise, and before the warm-up discard and the cycle-scaled budget it
// was worth 1.3 points of MAPE with no trend -- larger than several of the modelling effects
// this comparison reports.
//
// Behavioural: it runs the same point under budgets that differ by a factor of five and
// requires the observed mean to agree within a tolerance far below the effects being claimed.
func TestTheObservedMeanDoesNotDependOnTheSessionBudget(t *testing.T) {
	c := corpus(t)
	var sw Sweep
	for _, s := range c.Sweeps {
		if s.Scenario == "minimax-m2.5-b200-fp8-vllm-tp4.yaml" && s.Label == "1k1k" {
			sw = s
		}
	}
	if sw.Scenario == "" {
		t.Skip("reference sweep absent from the corpus")
	}
	// Concurrency 32 is where the dependence was worst: a 40-completion budget covered
	// barely one pool cycle there.
	const concurrency = 32
	base := cfg()
	var first float64
	for _, floor := range []int{24, 60, 120} {
		c := base
		c.SessionsPerPoint = floor
		obs, err := Run(sw, concurrency, c)
		if err != nil {
			t.Fatalf("floor %d: %v", floor, err)
		}
		if obs.Measured <= 0 {
			t.Fatalf("floor %d: no requests contributed to the mean", floor)
		}
		if first == 0 {
			first = obs.MeanITLUs
			continue
		}
		rel := math.Abs(obs.MeanITLUs-first) / first
		if rel > 0.05 {
			t.Errorf("a %d-completion floor gave %.0f us against %.0f at the smallest "+
				"floor, a %.1f%% difference; the mean is still tracking the budget rather "+
				"than the deployment", floor, obs.MeanITLUs, first, 100*rel)
		}
	}
}

// The budget must scale with concurrency, or a high-concurrency point measures fewer pool
// cycles than a low one and its mean is correspondingly noisier.
func TestTheCompletionBudgetScalesWithConcurrency(t *testing.T) {
	c := corpus(t)
	sw := c.Sweeps[0]
	low, err := Run(sw, 4, cfg())
	if err != nil {
		t.Fatalf("c=4: %v", err)
	}
	high, err := Run(sw, 64, cfg())
	if err != nil {
		t.Fatalf("c=64: %v", err)
	}
	if high.Completed <= low.Completed {
		t.Errorf("concurrency 64 completed %d requests and concurrency 4 completed %d; the "+
			"budget is not scaling, so the wider pool measures fewer cycles",
			high.Completed, low.Completed)
	}
	// And both must measure several cycles, not one.
	if got := float64(high.Measured) / 64; got < 1.5 {
		t.Errorf("concurrency 64 measured %d requests, only %.1f pool cycles after the "+
			"warm-up discard", high.Measured, got)
	}
}

// The warm-up discard must actually discard, and must never discard everything: a run that
// measured nothing would report a zero mean, which reads as a perfect score.
func TestTheWarmupDiscardLeavesSomethingMeasured(t *testing.T) {
	c := corpus(t)
	sw := c.Sweeps[0]
	conf := cfg()
	conf.WarmupFraction = 0.5
	obs, err := Run(sw, 8, conf)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if obs.WarmupDiscarded == 0 {
		t.Error("a 50% warm-up fraction discarded nothing")
	}
	if obs.Measured == 0 {
		t.Fatal("nothing was measured, so the reported mean is zero and would read as a " +
			"perfect prediction")
	}
	if obs.WarmupDiscarded+obs.Measured > obs.Completed {
		t.Errorf("discarded %d + measured %d exceeds %d completions",
			obs.WarmupDiscarded, obs.Measured, obs.Completed)
	}
	// An absurd fraction must fall back rather than measure nothing.
	conf.WarmupFraction = 0.999999
	obs2, err := Run(sw, 8, conf)
	if err != nil {
		t.Fatalf("Run at an extreme warmup: %v", err)
	}
	if obs2.Measured == 0 || obs2.MeanITLUs <= 0 {
		t.Error("an extreme warm-up fraction left nothing measured instead of falling back")
	}
}
