// Package harness runs BLIS against the InferenceX measurements AISimulate's end-to-end
// accuracy corpus is scored on, with the latency model supplied exclusively by
// blis-latency-kernel.
//
// # What is reproduced
//
// Each measured point is one run of InferenceX's benchmark client (benchmark_serving.py) at
// one concurrency. This package reproduces that run's MAIN phase exactly as the client drove
// it, and computes the client's own metrics from it:
//
//	workload     the 10*CONC target input/output lengths numpy drew at seed 0, in request
//	             order (workload.go; testdata/inferencex-workloads/seed0.json)
//	gating       a semaphore of CONC at --request-rate inf: the first CONC start at once,
//	             each later one when a slot frees, in order (blis replay pool mode)
//	output       --ignore-eos: exactly the target output length
//	deployment   the sweep's scenario, with every run setting the measured engine was
//	             launched with written into it (enginesettings.go)
//	TTFT, TPOT   calculate_metrics's definitions over the main-phase requests only
//
// The warm-up phase is not replayed as its own requests, but the state a drained warm-up leaves
// behind IS reproduced: Run scores the main phase from an empty engine where prefix caching is
// off, and from the primed state (request 0's prompt cached per replica) where it is on. Run says
// why each is exact. What is NOT reproduced is listed in docs/kernel-exclusive/APPLES-TO-APPLES.md
// and in the coverage report (gaps.go).
//
// # It does not simulate
//
// Every point is simulated by the `blis` binary (Config.Blis), invoked as `blis replay` on a
// trace of the point's requests and a scenario carrying its settings, exactly as a user would
// replay a trace (#1902). This package only states the point and reads back the
// --metrics-path file; see Run.
//
// # What it deliberately does not do
//
// It does not tune. No coefficient, no engine setting and no scheduler parameter is fitted
// to reduce error against the corpus. Every value either comes from the scenario file, from
// the measured run, from vLLM's own default for a setting the run did not pass, or from the
// benchmark client's source.
package harness

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

// Corpus is the evaluation set as blis-latency-kernel's extractor writes it.
type Corpus struct {
	Source            string `json:"source"`
	MeasurementSource string `json:"measurement_source"`
	AISimulateTotals  struct {
		TPOTMapePct       float64 `json:"tpot_mape_pct"`
		TPOTShapeErrorPct float64 `json:"tpot_shape_error_pct"`
		Points            int     `json:"points"`
	} `json:"aisimulate_totals"`
	Sweeps []Sweep `json:"sweeps"`
}

// Sweep is one deployment at one workload, measured across concurrency levels.
type Sweep struct {
	Scenario    string         `json:"scenario"`
	Model       string         `json:"model"`
	GPU         string         `json:"gpu"`
	Workload    string         `json:"workload"`
	Label       string         `json:"label"`
	Framework   string         `json:"framework"`
	Precision   string         `json:"precision"`
	Serving     string         `json:"serving"`
	SpecMethod  string         `json:"spec_method"`
	Parallelism map[string]int `json:"parallelism"`
	Points      []Point        `json:"points"`
}

// Point is one concurrency level. Every latency is normalised to the sweep's lowest
// concurrency, because the snapshot does not disclose absolute times.
type Point struct {
	Concurrency int `json:"concurrency"`
	// MeasuredRelative and AISimulateRelative keep their names for TPOT, which every figure
	// published before TTFT scoring existed was computed from.
	MeasuredRelative   float64 `json:"measured_tpot_relative"`
	AISimulateRelative float64 `json:"aisimulate_tpot_relative"`
	Status             string  `json:"status"`

	// TTFT, and AIC's predictions for both metrics. The artifact carries all of these per point;
	// an earlier revision of the extractor kept only TPOT under AISimulate.
	//
	// AIC and AISimulate are the same project -- snapshot.aic_source names ai-dynamo/aisimulate
	// for both -- AIC being the analytic configurator path now absorbed into the simulator. They
	// are two modes of one product, not two baselines, and are reported that way.
	MeasuredTTFTRelative   float64 `json:"measured_ttft_relative"`
	AISimulateTTFTRelative float64 `json:"aisimulate_ttft_relative"`
	AICRelative            float64 `json:"aic_tpot_relative"`
	AICTTFTRelative        float64 `json:"aic_ttft_relative"`
}

// MetricOf selects one metric's measured and predicted relatives from a point, so a scorer can
// loop over metrics instead of duplicating its arithmetic per metric.
func (p Point) MetricOf(m Metric) (measured, aisimulate, aic float64) {
	if m == MetricTTFT {
		return p.MeasuredTTFTRelative, p.AISimulateTTFTRelative, p.AICTTFTRelative
	}
	return p.MeasuredRelative, p.AISimulateRelative, p.AICRelative
}

// Metric names a published latency metric.
type Metric string

const (
	// MetricTPOT is time per output token: the steady-state decode metric.
	MetricTPOT Metric = "TPOT"
	// MetricTTFT is time to first token: prefill, queueing and admission.
	MetricTTFT Metric = "TTFT"
)

// AllMetrics is the reporting order.
var AllMetrics = []Metric{MetricTPOT, MetricTTFT}

// ISLOSL parses a "<isl>:<osl>" workload identity.
func (s Sweep) ISLOSL() (isl, osl int, err error) {
	parts := strings.Split(s.Workload, ":")
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("workload %q is not <isl>:<osl>", s.Workload)
	}
	if isl, err = strconv.Atoi(parts[0]); err != nil {
		return 0, 0, fmt.Errorf("workload %q: %w", s.Workload, err)
	}
	if osl, err = strconv.Atoi(parts[1]); err != nil {
		return 0, 0, fmt.Errorf("workload %q: %w", s.Workload, err)
	}
	return isl, osl, nil
}

// LoadCorpus reads the extracted corpus.
func LoadCorpus(path string) (*Corpus, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Corpus
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, err
	}
	if len(c.Sweeps) == 0 {
		return nil, fmt.Errorf("%s holds no sweeps", path)
	}
	return &c, nil
}

// Observation is what one (sweep, concurrency) point produced, as InferenceX's client
// computes it from the same requests, and the counts a reader needs to interpret it.
type Observation struct {
	// MeanTPOTUs is calculate_metrics's mean_tpot: the mean over main-phase requests with more
	// than one output token of (latency - ttft) / (output_len - 1), in microseconds.
	MeanTPOTUs float64
	// MeanTTFTUs is calculate_metrics's mean_ttft: the mean time to first token over every
	// main-phase request.
	MeanTTFTUs float64
	// Completed is how many main-phase requests completed: always 10 x concurrency, because
	// Run refuses anything else.
	Completed int
	// TPOTRequests is how many of them entered the TPOT mean (output length above one).
	TPOTRequests int
	// Preemptions is the run's preemption_count, as `blis replay` reports it.
	Preemptions int64

	// Settings is the configuration this point ran with and where each field came from, so
	// a report can distinguish a measured setting from a resolved default.
	Settings PointConfig

	// Primed reports that the main phase ran after the warm-up's state -- request 0's prompt
	// cached -- because the point's engine caches prefixes (see Run).
	Primed bool
}

// Config carries what a run needs beyond the sweep itself.
type Config struct {
	Repos kernelmodel.Repos
	// Blis is the path of the `blis` binary every point is simulated with. Required: the
	// harness does not simulate anything itself (#1902) -- it writes the point's scenario and
	// trace and runs `blis replay` on them the way any user would.
	Blis string
	// Seed is `blis replay`'s --seed. It draws the token IDs of the synthetic prompts, and
	// nothing else a point reads: the lengths come from Workloads, and with no shared prefix
	// a token ID changes no latency.
	Seed int64

	// Workloads holds every point's target length vectors. Nil uses the committed seed-0
	// fixture -- the benchmark's own draw. cmd/sensitivity passes the seed-1..4 fixtures to
	// show how much the score moves with the draw that exact reproduction removes.
	Workloads *BenchmarkWorkloads

	// MaxNumSeqsScale and TokenBudgetScale multiply the point's resolved caps. They exist
	// for cmd/sensitivity, which measures how much the score depends on settings the
	// snapshot does not publish. Zero means "use the resolved value unchanged", which is
	// what every scoring run does -- these are never set to tune a result, and choosing the
	// best value would be fitting to the evaluation set.
	MaxNumSeqsScale  float64
	TokenBudgetScale float64

	// EngineSettings carries the settings each measured run was launched with. When nil,
	// every point falls back to the scenario's values -- which is what this experiment did
	// before the settings were extracted, and is reported as such rather than assumed.
	EngineSettings *EngineSettingSet

	// PromptLengths is the evidence that each point's sent prompt lengths are the targets the
	// fixture replays (promptlengths.go). Nil uses the committed file.
	PromptLengths *PromptLengthEvidence

	// cold starts a prefix-caching point's main phase from an empty engine (RunCold).
	cold bool
}

var (
	defaultWorkloadsOnce sync.Once
	defaultWorkloads     *BenchmarkWorkloads
	defaultWorkloadsErr  error
)

// workloads is cfg's fixture, or the committed seed-0 one.
func (cfg Config) workloads() (*BenchmarkWorkloads, error) {
	if cfg.Workloads != nil {
		return cfg.Workloads, nil
	}
	defaultWorkloadsOnce.Do(func() {
		defaultWorkloads, defaultWorkloadsErr = LoadBenchmarkWorkloads(DefaultWorkloadsPath(0))
	})
	return defaultWorkloads, defaultWorkloadsErr
}

// scaled applies a sensitivity multiplier, with zero meaning "unchanged".
func scaled(v int64, factor float64) int64 {
	if factor <= 0 {
		return v
	}
	out := int64(float64(v) * factor)
	if out < 1 {
		return 1
	}
	return out
}

// Run drives one point by running `blis replay` on it.
//
// # Why the harness does not simulate
//
// An earlier revision built a single-instance simulator in-process, scaled max_num_seqs and the
// token budget by dp, and priced each step through a rank-splitting wrapper -- a second model
// of data parallelism beside the one the CLI ships (DP-as-placement: dp independent per-rank
// replicas behind the router). The two disagreed (#1902), and the score measured the
// harness's model rather than the simulator's. So the harness states the point -- a scenario
// carrying the point's engine settings and a trace of its requests -- and runs the same binary,
// with the same flags, a user would.
//
// # The warm-up
//
// The client sends 2*CONC copies of request 0 under its own semaphore of CONC, awaits all of
// them, and only then starts the main phase; the warm-up requests are in no metric. So the main
// phase starts against an idle engine whose only memory of the warm-up is whatever state
// outlives a drained request. In this simulator, on the configuration every point runs
// (single-tier GPU KV cache, round-robin routing, FCFS scheduling, always-admit), that is:
//
//   - the KV cache's block-hash map, which keeps the hashes of released blocks
//     (sim/kv/cache.go ReleaseKVBlocks). It is read for cross-request reuse only when prefix
//     caching is enabled (sim/batch_formation.go skips GetCachedBlocks when
//     PrefixCachingDisabled); there is no offload tier to read it otherwise. The only main
//     request whose prompt it holds is request 0: the others share no prefix with it;
//   - the round-robin router's counter (sim/routing.go), advanced by the number of copies.
//
// The scheduler, batch formation and the kernel adapter carry no state between requests, no
// step price reads the clock, and a drained engine holds no KV block.
//
// So where prefix caching is DISABLED, the main phase from an empty engine (mainPhase) is the
// main phase after the warm-up, and Run scores it. Where it is ENABLED, Run scores the main phase
// with request 0's prompt cached (primedPhase): dp copies of request 0 run to completion first,
// one per replica, so every replica holds request 0's prompt and the router counter has moved by
// a multiple of dp, and the main phase then starts on an idle engine. That is the drained
// warm-up's state with dp copies in place of 2*CONC: the copies are token-for-token the same
// prompt, so how many ran changes no block the main phase can read
// (TestThePrimedMainPhaseDoesNotDependOnHowManyCopiesRan). RunCold gives the empty-engine run of
// such a point, as information only. The exact two-phase protocol, with drain-gated phases in
// the trace format, is tracked as blis-schemas#56 and inference-sim#1926.
func Run(sw Sweep, concurrency int, cfg Config) (Observation, error) {
	if err := RequireBlis(cfg.Blis); err != nil {
		return Observation{}, err
	}
	isl, osl, err := sw.ISLOSL()
	if err != nil {
		return Observation{}, err
	}
	ws, err := cfg.workloads()
	if err != nil {
		return Observation{}, err
	}
	pl, err := ws.At(isl, osl, concurrency)
	if err != nil {
		return Observation{}, fmt.Errorf("%s: %w", sw.Scenario, err)
	}
	admit, dp, err := pointSettings(sw, concurrency, cfg)
	if err != nil {
		return Observation{}, err
	}
	engine := admit
	engine.MaxNumSeqs = int(scaled(int64(admit.MaxNumSeqs), cfg.MaxNumSeqsScale))
	engine.MaxNumBatchedTokens = int(scaled(int64(admit.MaxNumBatchedTokens), cfg.TokenBudgetScale))

	primed := !admit.PrefixCachingDisabled && !cfg.cold
	run, ignore := mainPhase(pl), func(string) bool { return false }
	if primed {
		copies := max(dp, 1)
		run = primedPhase(pl, copies)
		ignore = func(id string) bool {
			i, err := requestIndex(id)
			return err != nil || i >= len(pl.InputLens) // a warm-up copy or its holder round
		}
	}
	out, err := runBlis(cfg, sw.Scenario, engine, run)
	if err != nil {
		return Observation{}, fmt.Errorf("%s c=%d: %w", sw.Scenario, concurrency, err)
	}
	if primed {
		if err := checkPrimed(out.Requests, len(pl.InputLens)); err != nil {
			return Observation{}, fmt.Errorf("%s c=%d: %w", sw.Scenario, concurrency, err)
		}
	}
	obs, err := summarize(out.Requests, pl, ignore)
	if err != nil {
		return Observation{}, fmt.Errorf("%s c=%d: %w", sw.Scenario, concurrency, err)
	}
	obs.Preemptions = out.PreemptionCount
	obs.Settings = admit
	obs.Primed = primed
	return obs, nil
}

// pointSettings is the configuration a point runs with -- each setting from the strongest source
// available for THIS point, with its provenance -- and the deployment's data-parallel width. It
// is decided before any simulation, and Run calls it, so a caller that needs only the settings
// reads exactly what Run would run without simulating:
//
//	measured  the run's own command line, from its engine log
//	resolved  vLLM's own default, for a setting the run did not pass
//	scenario  the value in the scenario file, when neither exists
//
// The settings are PER RANK -- one vLLM EngineCore's -- because that is what the engine was
// launched with and what `blis replay` reads from a scenario: a dp > 1 deployment runs as dp
// replicas, each with these. Nothing here multiplies by dp.
func pointSettings(sw Sweep, concurrency int, cfg Config) (PointConfig, int, error) {
	m, err := kernelmodel.Open(sw.Scenario, cfg.Repos)
	if err != nil {
		return PointConfig{}, 0, fmt.Errorf("%s: %w", sw.Scenario, err)
	}
	eng, err := m.Engine()
	if err != nil {
		return PointConfig{}, 0, err
	}
	d := m.Deployment()
	return resolveAdmission(sw, concurrency, eng, d, cfg.EngineSettings), d.DP, nil
}

// RunCold is Run with the main phase started from an empty engine whatever the point's prefix
// caching -- for a prefix-caching point, the protocol WITHOUT the warm-up's cached prompt. It
// is information only: no scorer scores it.
func RunCold(sw Sweep, concurrency int, cfg Config) (Observation, error) {
	cfg.cold = true
	return Run(sw, concurrency, cfg)
}

// checkPrimed refuses a primed run that is not "the main phase after a drained warm-up":
//
//   - a warm-up copy that had not completed before the main phase started;
//   - a first main request that did not start with the main phase;
//   - a holder round (which keeps a copy's pool slot) that arrived before the last main
//     request completed, so that something besides the main phase was in the engine.
func checkPrimed(reqs []sim.RequestMetrics, n int) error {
	const usPerS, usPerMs = 1e6, 1e3
	var lastMainEnd, firstHolder float64
	firstHolder = math.Inf(1)
	for _, r := range reqs {
		start, end := r.ArrivedAt*usPerS, r.ArrivedAt*usPerS+r.E2E*usPerMs
		i, err := requestIndex(r.ID)
		switch {
		case err != nil: // a holder round: "session_warmup-<j>_round_1_<k>"
			firstHolder = math.Min(firstHolder, start)
		case i >= n:
			if end >= float64(primeMainStartUs) {
				return fmt.Errorf("warm-up copy %s ended at %.0f us, not before the main phase at %d us",
					r.ID, end, primeMainStartUs)
			}
		default:
			if i == 0 && math.Abs(start-float64(primeMainStartUs)) > 1 {
				return fmt.Errorf("main request 0 arrived at %.0f us, not with the main phase at %d us",
					start, primeMainStartUs)
			}
			lastMainEnd = math.Max(lastMainEnd, end)
		}
	}
	if firstHolder <= lastMainEnd {
		return fmt.Errorf("a warm-up holder round arrived at %.0f us, before the main phase ended at %.0f us",
			firstHolder, lastMainEnd)
	}
	return nil
}

// WarmupDifference is, for a prefix-caching point, how far the scored observation (after the
// warm-up: request 0's prompt cached) lies from the same main phase started from an empty
// engine. Information only: it is reported, never used to score or to place a point in a tier.
type WarmupDifference struct {
	Scored, Cold       Observation
	DTTFTPct, DTPOTPct float64 // 100 * (scored - cold) / cold
}

// RunWarmupDifference runs a prefix-caching point cold and reports its difference from the
// scored (primed) observation.
func RunWarmupDifference(sw Sweep, concurrency int, cfg Config, scored Observation) (WarmupDifference, error) {
	cold, err := RunCold(sw, concurrency, cfg)
	if err != nil {
		return WarmupDifference{}, err
	}
	return WarmupDifference{
		Scored: scored, Cold: cold,
		DTTFTPct: 100 * (scored.MeanTTFTUs/cold.MeanTTFTUs - 1),
		DTPOTPct: 100 * (scored.MeanTPOTUs/cold.MeanTPOTUs - 1),
	}, nil
}

// LoadDrawFixtures loads the numpy seed-1..4 fixtures cmd/sensitivity's draw sweep uses: other
// workloads from the same intervals, never repeated observations of the benchmark's seed 0.
func LoadDrawFixtures() ([]*BenchmarkWorkloads, error) {
	var out []*BenchmarkWorkloads
	for seed := 1; seed <= 4; seed++ {
		ws, err := LoadBenchmarkWorkloads(DefaultWorkloadsPath(seed))
		if err != nil {
			return nil, err
		}
		out = append(out, ws)
	}
	return out, nil
}

// summarize reduces a finished run's per-request metrics to InferenceX's two means, after
// checking that the run simulated exactly the point's requests.
//
// # The metric definitions, and how BLIS's fields map onto them
//
// calculate_metrics in benchmark_serving.py, over the main-phase outputs:
//
//	mean_ttft = mean(ttft)                                   over successful requests
//	mean_tpot = mean((latency - ttft) / (output_len - 1))    over those with output_len > 1
//
// where the client's ttft is the time from sending the request to its first content chunk,
// and latency the time from sending it to its LAST content chunk (most_recent_timestamp - st;
// the usage-only chunk and [DONE] that follow do not move it).
//
// BLIS's request clock starts when the pool admits the request, which is when the client
// would send it. Its ttft_ms (FirstTokenTime) runs to the first token's emission, including
// tokenization and output processing. Its e2e_ms is FirstTokenTime + every inter-token
// interval + PostDecodeFixedOverhead -- the fixed completion work AFTER the last token, which
// the client's latency does not include. So the client's (latency - ttft) is BLIS's sum of
// inter-token intervals, and BLIS's per-request itl_ms is exactly that sum over
// (output_len - 1) for a request that completed normally (sim/simulator.go
// recordRequestCompletion). TPOT is therefore read from itl_ms, not derived from e2e_ms,
// which would add PostDecodeFixedOverhead / (output_len - 1) to every request.
//
// Both means are over requests, not pooled over tokens, as the client's are.
//
// `blis replay` reports per-request latencies in milliseconds; they are converted back to
// microseconds (the simulator's tick) here. The round trip is exact to within one float64 ulp,
// far below anything the score prints.
func summarize(reqs []sim.RequestMetrics, pl PointLengths, ignore func(id string) bool) (Observation, error) {
	n := len(pl.InputLens)
	seen := make([]bool, n)
	const usPerMs = 1e3
	var ttftSum, tpotSum float64
	var tpotN, completed int
	for _, r := range reqs {
		if r.CompletionIndex == 0 || (ignore != nil && ignore(r.ID)) {
			continue
		}
		i, err := requestIndex(r.ID)
		if err != nil || i < 0 || i >= n || seen[i] {
			return Observation{}, fmt.Errorf("completed request %q is not one of the point's %d "+
				"main-phase requests, or completed twice", r.ID, n)
		}
		seen[i] = true
		// The trace states each request's lengths; a mismatch would mean the simulation ran
		// a different workload from the fixture.
		if r.NumPrefillTokens != pl.InputLens[i] || r.NumDecodeTokens != pl.OutputLens[i] {
			return Observation{}, fmt.Errorf("request %d ran %d input and %d output tokens; the "+
				"benchmark's targets are %d and %d", i, r.NumPrefillTokens, r.NumDecodeTokens,
				pl.InputLens[i], pl.OutputLens[i])
		}
		completed++
		ttftSum += r.TTFT * usPerMs
		if r.NumDecodeTokens > 1 {
			tpotSum += r.ITL * usPerMs
			tpotN++
		}
	}
	if completed != n {
		return Observation{}, fmt.Errorf("completed %d of the point's %d main-phase requests", completed, n)
	}
	obs := Observation{Completed: completed, TPOTRequests: tpotN, MeanTTFTUs: ttftSum / float64(completed)}
	if tpotN > 0 {
		obs.MeanTPOTUs = tpotSum / float64(tpotN)
	}
	return obs, nil
}

// requestIndex recovers the trace record index from the request ID `blis replay` assigns a
// one-round session's request, "request_<request_id>" (workload.LoadTraceV2SessionBlueprints).
func requestIndex(id string) (int, error) {
	const prefix = "request_"
	if !strings.HasPrefix(id, prefix) {
		return 0, fmt.Errorf("request id %q has no %q prefix", id, prefix)
	}
	return strconv.Atoi(strings.TrimPrefix(id, prefix))
}

// MAPE is the mean absolute percentage error of predicted against measured.
func MAPE(predicted, measured []float64) float64 {
	if len(predicted) != len(measured) || len(predicted) == 0 {
		return math.NaN()
	}
	var sum float64
	for i := range predicted {
		sum += math.Abs(predicted[i]-measured[i]) / measured[i]
	}
	return 100 * sum / float64(len(predicted))
}

// Median of a copy of xs.
func Median(xs []float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	c := append([]float64(nil), xs...)
	sort.Float64s(c)
	if len(c)%2 == 1 {
		return c[len(c)/2]
	}
	return (c[len(c)/2-1] + c[len(c)/2]) / 2
}

// MAPE2 is the mean of a slice of per-point percentage errors.
//
// Distinct from MAPE, which takes predicted and measured and computes the errors itself.
// Both exist because the caller here has already computed per-point errors on both sides and
// must average them identically -- averaging two differently-derived quantities is the error
// class this whole comparison is guarding against.
func MAPE2(errs []float64) float64 {
	if len(errs) == 0 {
		return math.NaN()
	}
	var s float64
	for _, e := range errs {
		s += e
	}
	return s / float64(len(errs))
}

// MonotoneMeasurement reports whether this sweep's MEASURED time per output token rises with
// client concurrency at every step.
//
// # Why this predicate exists, and why it is not cherry-picking
//
// Time per output token cannot fall when a fixed deployment is given more concurrent work:
// a wider resident batch shares the same GPU, so each token waits longer. A sweep where the
// measurement falls is recording something other than the steady-state response -- a
// different resident batch between the two runs, a scheduler regime change, or measurement
// noise -- and no monotone cost model can reproduce it.
//
// Three properties keep this honest:
//
//   - It is a property of the MEASUREMENT alone. It never reads either model's prediction, so
//     it cannot be tuned to favour one.
//   - It was defined from the physics before either side's error on the subsets was known.
//   - Excluding these sweeps HELPS the baseline: AISimulate's error on the vLLM subset falls
//     from 7.15% to 6.68% when they are removed, because its error on them (11.17%) is worse
//     than its average too. So this raises the bar rather than lowering it.
//
// Both figures are reported, always. The excluded subset is small and named: 4 sweeps and 25
// of 238 vLLM points, listed by cmd/kernelscore.
func (s Sweep) MonotoneMeasurement() bool {
	for i := 1; i < len(s.Points); i++ {
		if s.Points[i].MeasuredRelative <= s.Points[i-1].MeasuredRelative {
			return false
		}
	}
	return true
}

// TTFTSpikeIndices returns the indices of points whose MEASURED time to first token is an
// isolated excursion -- more than spikeFactor times both of its neighbours.
//
// These are measurement artefacts in the ground truth, not deployment behaviour. A first token
// cannot take 52 times longer at concurrency 16 than its neighbours at 8 and 32 and then
// recover; the neighbours bracket what the deployment actually does.
//
// The threshold is not a tuning knob, and the data says so. Over the 281 interior points of the
// vLLM corpus the excursion ratio cur/max(prev,next) has median 0.738 and p90 0.903, and its
// four largest values are 52.31x, 22.30x, 11.18x and 10.71x. The fifth largest is 1.74x. Every
// threshold from 2x to 10x therefore selects exactly the same four points -- the decision is
// made by a gap of nearly an order of magnitude in the data, not by the constant.
//
// The four, all on gpt-oss-120b/h200: tp4 1k1k at concurrency 16, tp2 8k1k at concurrency 8 and
// 32, and tp1 8k1k at concurrency 16.
//
// Why a POINT and not the whole sweep. The TPOT path excludes a non-monotone sweep entirely
// (MonotoneMeasurement), because a curve that falls is not recording a steady-state response
// anywhere along it. A TTFT spike is the opposite: one point is wrong and the rest of the sweep
// is sound, so dropping the sweep would discard good measurements to remove a bad one. The
// anchor is never excluded -- it defines the normalisation, and a sweep whose FIRST point is
// the artefact cannot be repaired by dropping a later one.
//
// Scale: the spikes cost AISimulate only 1.6 percentage points of TTFT shape error (18.13% to
// 16.47% over the vLLM subset), so this is not where any model's TTFT error lives. It is
// excluded because it is not a measurement, and reported rather than silently dropped.
func (s Sweep) TTFTSpikeIndices(spikeFactor float64) []int {
	var out []int
	for i := 1; i < len(s.Points)-1; i++ {
		cur := s.Points[i].MeasuredTTFTRelative
		prev, next := s.Points[i-1].MeasuredTTFTRelative, s.Points[i+1].MeasuredTTFTRelative
		if prev <= 0 || next <= 0 || cur <= 0 {
			continue
		}
		bound := prev
		if next > bound {
			bound = next
		}
		if cur > spikeFactor*bound {
			out = append(out, i)
		}
	}
	return out
}

// DefaultTTFTSpikeFactor is the excursion ratio above which a measured TTFT point is treated as
// an artefact. Ten sits inside the 1.74x-to-10.71x gap documented on TTFTSpikeIndices, so every
// value from 2 to 10 selects the same four points. Reported with the figures it affects.
const DefaultTTFTSpikeFactor = 10.0

// SignedStats summarises a slice of SIGNED per-point percentage errors, where a positive
// value means the model predicted a LARGER relative rise than was measured.
//
// MAPE2 answers "how big is the error"; these fields answer "which way does it run, and is
// that consistent". A model whose errors are symmetric scatter has Mean near zero and
// FractionOver near 0.5. A model with a systematic direction has |Mean| approaching MeanAbs
// and FractionOver approaching 0 or 1. The two cases call for different fixes -- rescaling a
// coefficient removes bias but cannot remove scatter -- which is why the sign is retained
// through aggregation rather than discarded at the point of computation.
type SignedStats struct {
	N            int
	Mean         float64 // signed; cancellation is the point
	MeanAbs      float64 // identical to MAPE2 over the same slice
	Median       float64 // signed
	FractionOver float64 // share with error > 0
	P10, P90     float64 // signed, to expose asymmetric tails
}

// Signed computes SignedStats over signed per-point percentage errors.
func Signed(errs []float64) SignedStats {
	if len(errs) == 0 {
		return SignedStats{Median: math.NaN(), Mean: math.NaN(), MeanAbs: math.NaN(),
			FractionOver: math.NaN(), P10: math.NaN(), P90: math.NaN()}
	}
	s := SignedStats{N: len(errs)}
	var sum, sumAbs float64
	over := 0
	for _, e := range errs {
		sum += e
		sumAbs += math.Abs(e)
		if e > 0 {
			over++
		}
	}
	s.Mean = sum / float64(len(errs))
	s.MeanAbs = sumAbs / float64(len(errs))
	s.Median = Median(errs)
	s.FractionOver = float64(over) / float64(len(errs))
	s.P10 = Percentile(errs, 10)
	s.P90 = Percentile(errs, 90)
	return s
}

// Percentile returns the p-th percentile of xs by linear interpolation between the two
// bracketing order statistics.
//
// Interpolation rather than nearest-rank because nearest-rank is not symmetric under
// p -> 100-p at small n: with six points, Go's half-away-from-zero rounding sends both the
// 10th and the 90th percentile UP a rank, so the pair no longer mirrors and a two-sided tail
// report acquires a direction that is not in the data. Per-concurrency buckets here run as
// small as one point, so the small-n behaviour is the common case, not the edge case.
func Percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	c := append([]float64(nil), xs...)
	sort.Float64s(c)
	if p <= 0 {
		return c[0]
	}
	if p >= 100 {
		return c[len(c)-1]
	}
	pos := p / 100 * float64(len(c)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return c[lo]
	}
	return c[lo] + (pos-float64(lo))*(c[hi]-c[lo])
}

// Abs returns a copy of errs with every element replaced by its magnitude, so a caller
// holding signed errors can obtain the absolute aggregate without mutating its own slice.
func Abs(errs []float64) []float64 {
	out := make([]float64, len(errs))
	for i, e := range errs {
		out[i] = math.Abs(e)
	}
	return out
}

// LogStats summarises per-point errors in LOG space, which is the natural space for a ratio.
//
// Mean and SD of log(predicted/measured) decompose a model's error into the part a single
// rescaling could remove (bias: the mean) and the part it could not (scatter: the SD). That
// decomposition is what tells a reader whether chasing a coefficient is worthwhile, and it is
// not recoverable from a percentage mean -- mean(log r) is the log of the GEOMETRIC mean of
// the ratios, while the arithmetic mean of (r-1) upweights over-predictions, so the two differ
// and the arithmetic one is always the larger (Jensen).
//
// ResidualFloor is the mean absolute percentage error that would remain if the bias were
// removed perfectly and the scatter left untouched: the floor on what recalibration can buy.
//
// ResidualFloor is NOT guaranteed to be below the error it decomposes. De-biasing scales every
// ratio by one constant, and in percentage space that can cost more than it saves when a point
// sits near a predicted value of zero: measured over random inputs, a pair like
// {-99.97%, +41.6%} yields a floor 3419 percentage points ABOVE its own mean absolute error,
// because |exp(d)-1| is convex and log compresses the two measures differently near -100%.
// The quantity is therefore informative for a model whose points are not near-total
// under-predictions -- which the kernel's are not, its p10 being -10.76% -- and misleading for
// one whose points are. Read it with the spread, never alone.
type LogStats struct {
	N             int
	MeanLog       float64 // bias, in nats
	SDLog         float64 // scatter, in nats
	GeoMeanPct    float64 // 100*(exp(MeanLog)-1), the bias as a percentage
	ResidualFloor float64 // mean |e| after perfect bias removal, in percent
}

// Logs computes LogStats from SIGNED percentage errors, where e is 100*(predicted/measured-1).
//
// A point with e <= -100 implies a non-positive predicted value, which cannot arise from a
// positive latency ratio; such a point is excluded and N reports how many were used.
func Logs(errs []float64) LogStats {
	var logs []float64
	for _, e := range errs {
		if r := 1 + e/100; r > 0 {
			logs = append(logs, math.Log(r))
		}
	}
	if len(logs) == 0 {
		return LogStats{MeanLog: math.NaN(), SDLog: math.NaN(),
			GeoMeanPct: math.NaN(), ResidualFloor: math.NaN()}
	}
	var sum float64
	for _, l := range logs {
		sum += l
	}
	mean := sum / float64(len(logs))
	var ss, resid float64
	for _, l := range logs {
		d := l - mean
		ss += d * d
		// The error this point would still carry with the bias removed.
		resid += math.Abs(math.Exp(d) - 1)
	}
	sd := 0.0
	if len(logs) > 1 {
		sd = math.Sqrt(ss / float64(len(logs)-1))
	}
	return LogStats{
		N:             len(logs),
		MeanLog:       mean,
		SDLog:         sd,
		GeoMeanPct:    100 * (math.Exp(mean) - 1),
		ResidualFloor: 100 * resid / float64(len(logs)),
	}
}
