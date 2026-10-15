package harness

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/kernelmodel/internal/artifacts"
)

// corpusPath is the AISimulate end-to-end corpus, under BLIS_MEASUREMENTS.
func corpusPath(t testing.TB) string { return artifacts.Measurement(t, "aisimulate_e2e.json") }

func cfg(t testing.TB) Config {
	return Config{
		Repos: kernelmodel.DefaultRepos(),
		Blis:  testBlis(t),
		Seed:  42,
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

// A run simulates exactly the benchmark's main phase: 10 x concurrency requests, every one
// completed, every one in the TPOT mean (no target output length is 1), and both means
// positive. A harness that produced a zero mean would make every MAPE meaningless, and that is
// exactly what happened when requests entered through EnqueueRequest instead of InjectArrival:
// 300 simulated seconds, zero completions, and a score that looked like a number.
func TestARunIsTheBenchmarksMainPhase(t *testing.T) {
	c := corpus(t)
	sw := c.Sweeps[0]
	for _, conc := range []int{sw.Points[0].Concurrency, 32} {
		obs, err := Run(sw, conc, cfg(t))
		if err != nil {
			t.Fatalf("c=%d: %v", conc, err)
		}
		// Literal 10, restated from --num-prompts $((CONC * 10)).
		if obs.Completed != 10*conc || obs.TPOTRequests != 10*conc {
			t.Errorf("c=%d completed %d and averaged TPOT over %d; the benchmark measures %d",
				conc, obs.Completed, obs.TPOTRequests, 10*conc)
		}
		if obs.MeanTPOTUs <= 0 || obs.MeanTTFTUs <= 0 {
			t.Errorf("c=%d: mean TPOT %v us, mean TTFT %v us", conc, obs.MeanTPOTUs, obs.MeanTTFTUs)
		}
	}
}

// Mean time per output token must rise with client concurrency at a fixed deployment. This is
// the property the entire comparison rests on: a flat curve would give a meaningless shape
// score, and a falling one would mean the closed loop is not applying load.
func TestInterTokenLatencyRisesWithConcurrency(t *testing.T) {
	c := corpus(t)
	sw := c.Sweeps[0]
	var prev float64
	for _, p := range sw.Points {
		obs, err := Run(sw, p.Concurrency, cfg(t))
		if err != nil {
			t.Fatalf("c=%d: %v", p.Concurrency, err)
		}
		if obs.MeanTPOTUs < prev {
			t.Errorf("concurrency %d gave mean ITL %.0f us, below the %.0f at the lower "+
				"level; the closed loop is not applying more load",
				p.Concurrency, obs.MeanTPOTUs, prev)
		}
		prev = obs.MeanTPOTUs
	}
}

// A run must be deterministic at a fixed seed. Without it, a MAPE difference between two
// runs could be noise rather than a modelling change, and no conclusion would be defensible.
func TestARunIsDeterministicAtAFixedSeed(t *testing.T) {
	c := corpus(t)
	sw := c.Sweeps[0]
	a, err := Run(sw, 8, cfg(t))
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	b, err := Run(sw, 8, cfg(t))
	if err != nil {
		t.Fatalf("second run: %v", err)
	}
	if a.MeanTPOTUs != b.MeanTPOTUs {
		t.Errorf("two runs at seed %d gave %.6f and %.6f us", cfg(t).Seed,
			a.MeanTPOTUs, b.MeanTPOTUs)
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
	a, err := Run(short, concurrency, cfg(t))
	if err != nil {
		t.Fatalf("1k1k: %v", err)
	}
	b, err := Run(long, concurrency, cfg(t))
	if err != nil {
		t.Fatalf("8k1k: %v", err)
	}
	if a.MeanTPOTUs == b.MeanTPOTUs {
		t.Errorf("a 1024-token prompt and an 8192-token prompt both gave %.0f us; the "+
			"sweep's workload is not reaching the simulation", a.MeanTPOTUs)
	}
	// An 8192-token prompt costs more prefill work per request, which at equal concurrency
	// raises the time between output tokens.
	if b.MeanTPOTUs <= a.MeanTPOTUs {
		t.Errorf("8k1k gave %.0f us, not above 1k1k's %.0f", b.MeanTPOTUs, a.MeanTPOTUs)
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
