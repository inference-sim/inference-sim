// Package harness runs BLIS closed-loop simulations against the AISimulate evaluation
// corpus, with the latency model supplied exclusively by blis-latency-kernel.
//
// # What makes this apples-to-apples
//
// The corpus states the deployment and the workload; this package reproduces both rather
// than choosing anything. Its fields come from the snapshot:
//
//	latency_scope      client_observed   -> the score reads per-request ITL, not step time
//	measurement_scope  end_to_end        -> a full request lifecycle through the simulator
//	workload           <isl>:<osl>       -> fixed input and output lengths per sweep
//	concurrency        N                 -> a fixed pool of N closed-loop sessions
//	serving            aggregated        -> one pool, no disaggregation
//	spec_method        none              -> speculation off
//	parallelism        tp/pp/dp/ep       -> from the scenario file the sweep names
//
// Concurrency is a CLIENT level, so it is driven as a closed loop: N sessions in flight, a
// new request admitted when one finishes. BLIS's own seam does this --
// `Simulator.OnRequestDone` returns follow-up requests -- so the resident batch is decided
// by BLIS's admission control rather than assumed equal to N. That decoupling is the entire
// point: blis-latency-kernel's 13.67% shape error against this corpus comes from assuming
// resident batch equals concurrency, and here it does not have to.
//
// # What it deliberately does not do
//
// It does not tune. No coefficient, no engine setting and no scheduler parameter is fitted
// to reduce error against the corpus. Every value either comes from the scenario file, from
// the snapshot, or is a BLIS default recorded in Conditions.
package harness

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"

	// Registers the KV-store constructor behind sim.MustNewKVStoreFromConfig. BLIS breaks
	// the import cycle between sim/ and sim/kv/ with a registration variable, so a package
	// that builds a simulator must import sim/kv for its side effect.
	_ "github.com/inference-sim/inference-sim/sim/kv"
	"github.com/inference-sim/inference-sim/sim/workload"
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

// Admission selects how the resident batch is bounded.
type Admission int

const (
	// AdmissionKernelKV derives the KV budget from the kernel's memory methods, so KV
	// pressure and preemption bound the resident batch alongside max_num_seqs.
	AdmissionKernelKV Admission = iota
	// AdmissionSeqsOnly sets a KV budget large enough never to bind, leaving max_num_seqs
	// and the token budget as the only bounds. Used where the kernel's FixedBytes cannot
	// supply a usable budget; see kernelmodel.UpstreamExpertWeightDefect.
	AdmissionSeqsOnly
)

// Run simulates one (sweep, concurrency) point and returns the mean ITL in microseconds.
//
// The returned Observation carries the resident batch actually achieved, because that is the
// quantity this experiment is about and a mean latency with no batch beside it cannot be
// interpreted.
type Observation struct {
	MeanITLUs float64
	// MeanTTFTUs is the mean time to first token over EVERY measured completion -- deliberately
	// not the post-warm-up subset MeanITLUs uses. See summarize for why the cut belongs to one
	// metric and not the other.
	MeanTTFTUs     float64
	MeasuredTTFT   int
	Completed      int
	MeanResident   float64
	PeakResident   int
	Preemptions    int
	KVBlocks       int64
	AdmissionUsed  Admission
	StepsSimulated int64

	// WarmupDiscarded is how many leading completions were dropped, and Measured how many
	// contributed to MeanITLUs. Reported so a reader can see the mean was not taken over a
	// transient, and so a run that discarded everything is visible rather than silent.
	WarmupDiscarded int
	Measured        int

	// Settings is the configuration this point ran with and where each field came from, so
	// a report can distinguish a measured setting from a resolved default.
	Settings PointConfig
}

// Config carries what a run needs beyond the sweep itself.
type Config struct {
	Repos     kernelmodel.Repos
	Admission Admission
	// SessionsPerPoint is how many requests each concurrency level completes before the
	// run stops. Larger is steadier and slower; the value is recorded in the report.
	SessionsPerPoint int
	Seed             int64

	// MaxNumSeqsScale and TokenBudgetScale multiply the scenario's stated caps. They exist
	// for cmd/sensitivity, which measures how much the score depends on settings the
	// snapshot does not publish. Zero means "use the scenario's value unchanged", which is
	// what every scoring run does -- these are never set to tune a result, and choosing the
	// best value would be fitting to the evaluation set.
	MaxNumSeqsScale  float64
	TokenBudgetScale float64

	// LengthRangeRatio overrides the prompt-length sampling interval, as a FRACTION of
	// the labelled length. It exists because the replayed workload and the measured one
	// are not the same distribution, and which one to replay is a question the data
	// cannot settle.
	//
	// AISimulate's replay draws one-sided, [ratio*len, len], so at the default 0.8 the
	// mean prompt is 0.9*len. vLLM's own benchmark client draws SYMMETRICALLY,
	// [floor(len*(1-r)), ceil(len*(1+r))], with r defaulting to "0.0"
	// (vllm/benchmarks/datasets/datasets.py:1944 and datasets/utils.py:72-74) -- so a run
	// that passed no --random-range-ratio used CONSTANT lengths of exactly len, with a
	// mean 11% longer than what is replayed here.
	//
	// InferenceX's rows carry isl and osl as plain integers with no ratio, dataset name
	// or seed, and the engine-settings extraction finds no ratio either, so neither
	// reading can be confirmed from the data. This knob makes the alternative testable
	// rather than asserted. A value of 0 keeps AISimulate's default; 1.0 means constant
	// lengths, which is vLLM's documented default behaviour.
	//
	// Like the two scales above, this is a sensitivity control. Picking whichever value
	// scores best would be fitting to the evaluation set.
	LengthRangeRatio float64

	// Estimator selects which latency model supplies step time. The zero value is
	// EstimatorKernel, so a caller that does not set it gets blis-latency-kernel and is
	// byte-identical to a build without this field (INV-6).
	//
	// Only STEP TIME changes with this field. KV blocks still come from the kernel's memory
	// methods, and the host per-token cost is taken from the kernel and given to every arm,
	// so an arm differs from another on the forward-pass model alone. See backends.go.
	Estimator Estimator
	// Backends locates the inputs the analytic arms need. Required only when Estimator is
	// not the kernel.
	Backends BackendPaths

	// EngineSettings carries the settings each measured run was launched with. When nil,
	// every point falls back to the scenario's values -- which is what this experiment did
	// before the settings were extracted, and is reported as such rather than assumed.
	EngineSettings *EngineSettingSet

	// WarmupFraction is the leading fraction of completed requests whose inter-token
	// latency is discarded before the mean is taken.
	//
	// A closed-loop pool does not start at its steady state: at t=0 all N users submit at
	// once, so the first requests see a resident batch that is still filling and a KV cache
	// that is still cold. Pooling their ITL with the steady-state ones makes the reported
	// mean depend on how many requests the run completed, which is a property of the
	// harness rather than of the deployment. Measured before this existed: the same score
	// read 9.92%, 11.20%, 9.89%, 10.72% and 10.64% at 20, 30, 40, 60 and 100 sessions per
	// point -- a 1.3-point swing with no trend, which is transient contamination.
	//
	// Zero disables it. The default used by cmd/kernelscore is stated there.
	WarmupFraction float64

	// ThinkTimeUs is per-round client think time. Zero -- the default and what every
	// scoring run uses -- models a saturating client that re-submits the instant its
	// previous request finishes, which is what a closed-loop concurrency sweep measures.
	// Non-zero exists so cmd/sensitivity can show whether the assumption is load-bearing.
	ThinkTimeUs int64
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

// Run drives one point.
func Run(sw Sweep, concurrency int, cfg Config) (Observation, error) {
	isl, osl, err := sw.ISLOSL()
	if err != nil {
		return Observation{}, err
	}
	m, err := kernelmodel.Open(sw.Scenario, cfg.Repos)
	if err != nil {
		return Observation{}, fmt.Errorf("%s: %w", sw.Scenario, err)
	}
	eng, err := m.Engine()
	if err != nil {
		return Observation{}, err
	}

	// The KV budget: from the kernel where it can be derived, else large enough not to
	// bind. Which one was used is reported, never silently chosen.
	admission := cfg.Admission
	var blocks int64
	if admission == AdmissionKernelKV {
		b, err := m.KVBudget()
		if err != nil {
			return Observation{}, fmt.Errorf("%s: %w", sw.Scenario, err)
		}
		blocks = b.TotalBlocks
	} else {
		// A ceiling that cannot bind: every session's whole context, plus slack.
		perSeq := int64(math.Ceil(float64(isl+osl)/float64(eng.BlockSize))) + 2
		blocks = perSeq * int64(concurrency) * 2
	}

	// The workload the REAL harness drives, from its own source constants, which AISimulate's
	// replay also matches. See workload.go for the citations and for why the distribution is
	// matched rather than the individual draws.
	//
	// The request budget is the real harness's too: InferenceX runs `--num-warmups
	// $((2 * CONC))` and then measures `--num-prompts $((CONC * 10))`, so a point is
	// 12 x concurrency requests of which the leading 2 x concurrency are discarded. An earlier
	// version used max(24, 4 x concurrency) with the leading half discarded -- a criterion
	// derived here from the shape of the transient. It converged, but it measured about
	// 2 x concurrency requests where the harness measures 10 x, and it discarded a fraction
	// rather than a fixed phase. SessionsPerPoint survives only as a floor.
	w := AISimulateWorkload(isl, osl, concurrency)
	// Applied to the INPUT length only. Output-length variance is what makes requests
	// retire at different times, so the resident batch churns rather than finishing in
	// lockstep; removing it changes the queueing regime rather than the prompt shape,
	// and a first version of this knob that flattened both took TTFT's shape error from
	// 16.22% to 53.73%.
	if cfg.LengthRangeRatio > 0 {
		w.ISLLow = int(float64(isl) * cfg.LengthRangeRatio)
	}
	warmup := w.WarmupCount
	total := w.WarmupCount + w.RequestCount
	if floor := cfg.SessionsPerPoint; floor > total {
		total = floor
	}

	// Built as a BLIS WorkloadSpec and expanded by BLIS's own generator rather than by
	// assembling SessionBlueprints here. Three reasons, all about fidelity:
	//
	//   - A Concurrency client is BLIS's native closed-loop primitive. GenerateWorkload emits
	//     N seed requests plus unlimited-round blueprints, and SessionManager.OnComplete
	//     supplies a follow-up per completion, holding N users in flight. That is what the
	//     snapshot's concurrency means, and it leaves the RESIDENT batch to BLIS's admission
	//     control rather than fixing it at N.
	//   - Concurrency and RateFraction are mutually exclusive in BLIS and Validate() enforces
	//     it, so that invariant is checked rather than assumed.
	//   - Every other knob (prefix sharing, multimodal, reasoning, LoRA, network, lifecycle,
	//     SLO) is left at its zero value, so the spec records exactly what this comparison
	//     does and does not exercise.
	//
	// Lengths are discrete uniform over AISimulate's one-sided interval, expressed through
	// BLIS's existing `empirical` sampler. The label is an upper bound: at "1024:1024" the
	// interval is [819, 1024] on both axes. An earlier version used a constant at the label,
	// which drove a mean context about 10% longer than the baseline's and removed the length
	// variance entirely -- and variance matters here beyond its mean, because variable output
	// lengths make requests retire at different times, so the resident batch churns instead
	// of finishing in lockstep.
	spec := &workload.WorkloadSpec{
		Version:  "v1",
		Seed:     cfg.Seed,
		Category: "language", // plain text generation: the snapshot states no multimodal or reasoning workload
		Clients: []workload.ClientSpec{{
			ID:          "closed-loop",
			Concurrency: concurrency,
			// Zero think time by default: the snapshot's replay sets no request rate or
			// arrival interval, so every arrival is at t=0 and concurrency alone gates
			// execution -- a saturating client. Varied only by cmd/sensitivity.
			ThinkTimeUs: cfg.ThinkTimeUs,
			// RateFraction stays 0, which is what makes this a concurrency client.
			InputDist: workload.DistSpec{
				Type: "empirical", Params: pdfParams(uniformPDF(w.ISLLow, w.ISLHigh)),
			},
			OutputDist: workload.DistSpec{
				Type: "empirical", Params: pdfParams(uniformPDF(w.OSLLow, w.OSLHigh)),
			},
			// No shared prefix: a prefix-cache hit would change the work per request, and
			// AISimulate's spec sets cached_prefix_tokens to zero.
			PrefixLength: 0,
			Streaming:    true,
		}},
		NumRequests: int64(total),
	}
	if err := spec.Validate(); err != nil {
		return Observation{}, fmt.Errorf("workload spec: %w", err)
	}
	wl, err := workload.GenerateWorkload(spec, math.MaxInt64, int64(total))
	if err != nil {
		return Observation{}, err
	}
	if len(wl.Sessions) == 0 {
		return Observation{}, fmt.Errorf(
			"a concurrency-%d client produced no session blueprints, so the run would be "+
				"open-loop and the resident batch would not be bounded by the client level",
			concurrency)
	}
	mgr := workload.NewSessionManager(wl.Sessions)
	if wl.FollowUpBudget >= 0 {
		mgr.SetFollowUpBudget(wl.FollowUpBudget)
	}

	// Data parallelism runs dp independent EngineCores, each with its OWN max_num_seqs and
	// token budget, and requests split disjointly across them. This single-instance
	// simulator models the aggregate, so both caps scale by dp -- the same rule KVBudget
	// applies to blocks, and vLLM's own (see latency.CalculateKVBlocks' dp scaling).
	//
	// Getting this wrong is not a small inaccuracy. The ep4-dp2 sweeps in this corpus begin
	// at concurrency 256 against a stated max_num_seqs of 256: without the dp factor the
	// resident batch saturates at the first point and the predicted curve is FLAT
	// (1.002, 1.003 across a four-fold concurrency rise) while the measurement doubles.
	dp := int64(m.DataParallelWidth())
	if dp < 1 {
		dp = 1
	}
	// The three settings that decide admission, each taken from the strongest source
	// available for THIS point and reported so a reader knows which was used:
	//
	//   measured  the run's own command line, from its engine log
	//   resolved  vLLM's device-memory resolution, for a setting the run did not pass
	//   scenario  the value in the scenario file, when no measurement exists at all
	//
	// The ordering matters because the settings are not uniform. Across the 238 scored
	// points the runs passed max_num_seqs equal to the client concurrency on 92, a fixed
	// value on 69, and nothing on 77, so neither a single value nor a single rule is right
	// anywhere near everywhere.
	admit := resolveAdmission(sw, concurrency, eng, m.Deployment(), cfg.EngineSettings)
	obsSettings := admit

	cfgSim := sim.SimConfig{
		Horizon:       math.MaxInt64,
		Seed:          cfg.Seed,
		KVCacheConfig: sim.NewKVCacheConfig(blocks, int64(admit.BlockSize), 0, 0, 0, 0),
		BatchConfig: sim.NewBatchConfig(
			scaled(int64(admit.MaxNumSeqs)*dp, cfg.MaxNumSeqsScale),
			scaled(int64(admit.MaxNumBatchedTokens)*dp, cfg.TokenBudgetScale), 0,
			// InferenceX launched 233 of the 238 scored points with
			// --no-enable-prefix-caching, so a comparison that cached unconditionally
			// charged less prefill work than the engine did (#1867).
			sim.WithPrefixCachingDisabled(admit.PrefixCachingDisabled)),
	}
	// The latency model under test. The KERNEL supplied everything above -- the KV budget,
	// the engine settings, the dp width -- so swapping only this leaves the resident batch
	// and the admission behaviour decided identically for every arm, which is what makes the
	// comparison a step-time comparison.
	var lm sim.LatencyModel = m
	if cfg.Estimator != "" && cfg.Estimator != EstimatorKernel {
		alt, err := altModel(cfg.Estimator, m.Deployment(), cfg.Backends,
			hostCosts{perOutputTokenUs: m.OutputTokenProcessingTime()})
		if err != nil {
			return Observation{}, fmt.Errorf("%s: %w", sw.Scenario, err)
		}
		lm = alt
	}
	kvStore := sim.MustNewKVStoreFromConfig(cfgSim.KVCacheConfig)
	s, err := sim.NewSimulator(cfgSim, kvStore, lm)
	if err != nil {
		return Observation{}, err
	}
	// The closed loop: BLIS's own seam. Each completion returns the follow-up request that
	// keeps the pool at N in flight, so the RESIDENT batch is whatever admission control
	// allows rather than N by construction.
	// Completion order, for the warm-up discard. OnRequestDone fires once per request
	// reaching a terminal state, in completion order, which is exactly the sequence the
	// steady-state cut needs.
	var order []string
	s.OnRequestDone = func(req *sim.Request, tick int64) []*sim.Request {
		order = append(order, req.ID)
		return mgr.OnComplete(req, tick)
	}
	// InjectArrival, not EnqueueRequest. EnqueueRequest puts a request in the wait queue and
	// schedules only its timeout; it is the tail of the arrival path, called BY QueuedEvent,
	// which is what triggers the first StepEvent. Calling it directly leaves the simulator
	// with a populated wait queue and no step scheduled, so Run() processes the timeouts and
	// exits having executed zero steps -- which is exactly what happened before this
	// comment existed.
	for _, r := range wl.Requests {
		s.InjectArrival(r)
	}
	s.Run()

	obs := summarize(s, order, warmup, cfg.WarmupFraction, blocks, admission)
	obs.Settings = obsSettings
	return obs, nil
}

// summarize reduces a finished simulation to the observation the score needs.
//
// The mean is taken over the per-request mean inter-token latencies of the requests that
// completed AFTER the warm-up prefix, in completion order. Two reasons it is per request
// rather than over BLIS's pooled AllITLs: a pooled mean weights a long request more heavily
// than a short one, and only a per-request view can drop a warm-up prefix at all.
func summarize(s *sim.Simulator, order []string, warmupCount int, warmupFraction float64,
	blocks int64, a Admission) Observation {
	obs := Observation{
		Completed:     len(order),
		KVBlocks:      blocks,
		AdmissionUsed: a,
	}
	// The real harness's fixed warm-up phase. WarmupFraction overrides it for
	// cmd/sensitivity, which varies it to show the score does not turn on the choice.
	cut := warmupCount
	if warmupFraction > 0 && warmupFraction < 1 {
		cut = int(float64(len(order)) * warmupFraction)
	}
	// Never discard everything: a run that completed few requests still has to report
	// something, and silently returning zero would read as a perfect score.
	if cut >= len(order) {
		cut = 0
	}
	var sum float64
	var n int
	var ttftSum float64
	var ttftN int
	for _, id := range order[cut:] {
		if itl, ok := s.Metrics.RequestITLs[id]; ok && itl > 0 {
			sum += itl
			n++
		}
	}
	// TTFT is averaged over EVERY measured completion, not over order[cut:].
	//
	// The warm-up cut and the metric it serves are not the same question. Time per output token
	// is a steady-state quantity: a closed-loop pool starts with a filling batch and a cold
	// cache, so pooling the transient into a TPOT mean makes the figure depend on how many
	// requests the run completed. Time to FIRST token is the opposite -- the transient is the
	// phenomenon. When N requests are launched at once against an empty queue, a 920-token
	// prompt against an 8192-token budget admits 8 prefills per step, so the 64th request waits
	// 8 waves and its TTFT is 8 prefills deep. Discarding the first 2N completions discards
	// exactly that queueing.
	//
	// This is what the real harness measures, not a preference. InferenceX's benchmark_serving.py
	// awaits its warm-up requests and throws them away entirely (`_ = await
	// asyncio.gather(warmup_tasks)`), then starts the MEASURED phase against an empty queue and,
	// at --request-rate inf, launches all 10*concurrency of them at once ("Infinite request_rate
	// disables waiting"). The start-up transient is inside the measured set on their side, so
	// removing it on ours compares two different quantities.
	//
	// Measured on gpt-oss-120b-h200-fp4-vllm-tp4 1k1k, TTFT relatives across concurrency 4 to 64:
	// 1.000 1.032 1.098 1.212 1.337 with the cut applied, 1.000 1.115 1.343 1.699 2.325 without,
	// against a measurement of 1.000 1.131 (spike) 1.865 2.871. The cut, not the model, was the
	// reason the TTFT curve was flat.
	for _, id := range order {
		// Counted independently of the ITL branch: a request that emitted exactly one token has
		// a TTFT and no inter-token interval, so requiring both would silently drop it.
		if t, ok := s.Metrics.RequestTTFTs[id]; ok && t > 0 {
			ttftSum += t
			ttftN++
		}
	}
	if n > 0 {
		obs.MeanITLUs = sum / float64(n)
	}
	if ttftN > 0 {
		obs.MeanTTFTUs = ttftSum / float64(ttftN)
	}
	obs.MeasuredTTFT = ttftN
	obs.WarmupDiscarded = cut
	obs.Measured = n
	return obs
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
