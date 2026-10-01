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
	Source             string `json:"source"`
	MeasurementSource  string `json:"measurement_source"`
	AISimulateTotals   struct {
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
	Concurrency        int     `json:"concurrency"`
	MeasuredRelative   float64 `json:"measured_tpot_relative"`
	AISimulateRelative float64 `json:"aisimulate_tpot_relative"`
	Status             string  `json:"status"`
}

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
	MeanITLUs      float64
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

	// CyclesPerPoint is how many full pool cycles of completions each point must gather, on
	// top of the SessionsPerPoint floor. A "cycle" is `concurrency` completions: one pass of
	// every user in the pool. Zero means defaultCyclesPerPoint.
	CyclesPerPoint int

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
	cfgSim := sim.SimConfig{
		Horizon:       math.MaxInt64,
		Seed:          cfg.Seed,
		KVCacheConfig: sim.NewKVCacheConfig(blocks, int64(eng.BlockSize), 0, 0, 0, 0),
		BatchConfig: sim.NewBatchConfig(
			scaled(int64(eng.MaxNumSeqs)*dp, cfg.MaxNumSeqsScale),
			scaled(int64(eng.MaxNumBatchedTokens)*dp, cfg.TokenBudgetScale), 0),
	}
	kvStore := sim.MustNewKVStoreFromConfig(cfgSim.KVCacheConfig)
	s, err := sim.NewSimulator(cfgSim, kvStore, m)
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

	return summarize(s, order, warmup, cfg.WarmupFraction, blocks, admission), nil
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
	for _, id := range order[cut:] {
		if itl, ok := s.Metrics.RequestITLs[id]; ok && itl > 0 {
			sum += itl
			n++
		}
	}
	if n > 0 {
		obs.MeanITLUs = sum / float64(n)
	}
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

// defaultCyclesPerPoint is how many full pool cycles a point gathers when the caller states
// no preference: four, of which the first half is discarded as warm-up, leaving two settled
// cycles. Chosen from the observed convergence of the per-point mean rather than from a score.
const defaultCyclesPerPoint = 4
