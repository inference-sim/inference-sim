package harness

// workload.go reproduces the workload AISimulate's end-to-end accuracy replay drives, so that
// BLIS and AISimulate are simulating the same thing against the same measurements.
//
// # Why this file exists
//
// The snapshot labels each sweep with a workload identity such as "1024:1024". That label is
// NOT the request length. It is the UPPER BOUND of a sampling interval, and reading it as a
// value made this harness drive a different workload from the baseline it was being compared
// against -- a mean context about 10% too long, with no variance at all.
//
// # What AISimulate actually does, with citations
//
// `scripts/run_e2e_accuracy.py` builds each replay spec with
//
//	"isl": request.workload.isl,
//	"osl": request.workload.osl,
//	"request_count": request.workload.concurrency * 10,
//	"random_range_ratio": 0.8,
//	"random_seed": 0,
//
// and `python/aisimulate/src/aisimulate/runner.py` samples lengths with
//
//	if random_range_ratio == 1.0:
//	    return [upper] * count
//	lower = int(upper * random_range_ratio)
//	return [rng.randint(lower, upper) for _ in range(count)]
//
// Python's `randint` is inclusive at both ends. So at the "1024:1024" label, input length is
// uniform on [819, 1024] and output length is uniform on [819, 1024], sampled independently
// per request, and the run consists of `concurrency * 10` requests.
//
// # Why the DISTRIBUTION is matched and not the individual draws
//
// AISimulate draws from Python's Mersenne Twister at seed 0. Go's generator is different, so
// reproducing the sequence draw-for-draw would mean precomputing 83 sweeps' worth of length
// vectors in Python and shipping them as fixtures. The metric does not need that: each point
// averages inter-token latency over ten pool cycles, so what enters the comparison is the
// distribution, not the order. Matching the distribution exactly and the draws not at all is
// the smaller, checkable claim -- and parity_test.go asserts the distribution.
//
// # One thing that cannot be matched
//
// The REAL benchmark's workload is not recoverable. InferenceX's `benchmark_results` rows
// carry `isl` and `osl` as plain integers with no range ratio, dataset name or seed. And
// vLLM's own `--random-range-ratio` is SYMMETRIC -- `[len*(1-r), len*(1+r)]`, per
// `vllm/benchmarks/datasets/utils.py` -- where AISimulate's is one-sided. So AISimulate
// simulates a mean context about 10% shorter than the real runs under either reading of the
// real one. That is a property of the baseline, it bounds how closely ANY simulator can match
// these measurements, and it is stated in the results rather than absorbed.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

// AISimulate's replay constants, from the source cited above. Named rather than inlined so
// parity_test.go asserts against the same values the harness uses.
const (
	aisimulateRangeRatio      = 0.8
	aisimulateRequestsPerUser = 10
	aisimulateLengthSeed      = 0
	aisimulateArrivalSeed     = 42

	// Warm-up requests per user, from the real harness rather than from AISimulate's replay.
	// InferenceX's own client runs `--num-warmups $((2 * CONC))` BEFORE the measured
	// `--num-prompts $((CONC * 10))`, so warm-up requests are excluded from the measurement
	// entirely rather than being a fraction of it
	// (inferencex-e2e/benchmarks/single_node/srt_fixed_sequence.sh and benchmark_lib.sh).
	inferencexWarmupsPerUser = 2
)

// Workload is the sampling interval and request count for one (label, concurrency) point.
type Workload struct {
	ISLLow, ISLHigh int
	OSLLow, OSLHigh int
	// RequestCount is the MEASURED request count: concurrency x 10.
	RequestCount int
	// WarmupCount is the leading requests to run and discard: concurrency x 2. The real
	// harness runs these as a separate phase before the measured ones, so the measured count
	// is not reduced by them.
	WarmupCount int
}

// AISimulateWorkload returns the workload AISimulate's replay drives for a sweep labelled
// isl:osl at this concurrency.
func AISimulateWorkload(isl, osl, concurrency int) Workload {
	return Workload{
		ISLLow:  int(float64(isl) * aisimulateRangeRatio),
		ISLHigh: isl,
		OSLLow:  int(float64(osl) * aisimulateRangeRatio),
		OSLHigh: osl,
		// concurrency * 10, matching run_e2e_accuracy.py. This supersedes the harness's own
		// convergence criterion of four pool cycles; ten is more, so convergence still holds
		// and the budget is now the baseline's rather than this project's choice.
		RequestCount: concurrency * aisimulateRequestsPerUser,
		WarmupCount:  concurrency * inferencexWarmupsPerUser,
	}
}

// uniformPDF builds the discrete uniform distribution over [low, high] inclusive, in the form
// BLIS's `empirical` sampler takes: a map from token count to probability.
//
// The empirical sampler is used rather than a new uniform one because it already exists and is
// reachable from a WorkloadSpec, so the harness stays declarative and nothing is added to
// BLIS for this experiment.
func uniformPDF(low, high int) map[int]float64 {
	if high < low {
		low, high = high, low
	}
	n := high - low + 1
	w := 1.0 / float64(n)
	pdf := make(map[int]float64, n)
	for v := low; v <= high; v++ {
		pdf[v] = w
	}
	return pdf
}

// pdfParams renders a PDF as the string-keyed map a DistSpec carries.
func pdfParams(pdf map[int]float64) map[string]float64 {
	out := make(map[string]float64, len(pdf))
	for length, weight := range pdf {
		out[itoa(length)] = weight
	}
	return out
}

// itoa avoids a strconv import in a file that needs nothing else from it.
func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// --- The InferenceX benchmark's exact per-request target lengths ---
//
// scripts/gen_inferencex_workloads.py writes, per (isl, osl, concurrency), the length vectors
// InferenceX's client draws with numpy at its default seed 0; fixture_test.go cross-checks them.

// The benchmark protocol's constants, from the source cited above. Named so the fixture loader
// and the tests check the file against the same values the protocol states.
const (
	// inferencexRangeRatio is RANDOM_RANGE_RATIO, fixed at 0.8 in every benchmark workflow.
	inferencexRangeRatio = 0.8
	// inferencexRequestsPerConcurrency is --num-prompts / --max-concurrency.
	inferencexRequestsPerConcurrency = 10
	// inferencexWarmupsPerConcurrency is --num-warmups / --max-concurrency. The warm-up phase
	// is NOT simulated; see Run for why that is exact on prefix-caching-disabled points and
	// flagged on the others.
	inferencexWarmupsPerConcurrency = 2
)

// PointLengths is the target length of every main-phase request of one point, in request
// order: request i of the benchmark asked for InputLens[i] prompt tokens and OutputLens[i]
// output tokens.
type PointLengths struct {
	ISL         int   `json:"isl"`
	OSL         int   `json:"osl"`
	Concurrency int   `json:"concurrency"`
	InputLens   []int `json:"input_lens"`
	OutputLens  []int `json:"output_lens"`
}

// BenchmarkWorkloads is the fixture: every scored point's length vectors under one numpy seed.
type BenchmarkWorkloads struct {
	Seed                   int            `json:"seed"`
	RangeRatio             float64        `json:"range_ratio"`
	RequestsPerConcurrency int            `json:"requests_per_concurrency"`
	PrefixLen              int            `json:"prefix_len"`
	Points                 []PointLengths `json:"points"`

	path  string
	byKey map[[3]int]int
}

// DefaultWorkloadsPath is the committed fixture for a numpy seed. Seed 0 is the benchmark's;
// seeds 1 to 4 exist for cmd/sensitivity's draw-sensitivity sweep.
func DefaultWorkloadsPath(seed int) string {
	return filepath.Join(repoRoot(), "testdata", "inferencex-workloads", "seed"+strconv.Itoa(seed)+".json")
}

// LoadBenchmarkWorkloads reads a fixture and refuses one that does not state the benchmark's
// protocol: a different range ratio or request count, a prefix, or a point whose vectors have
// the wrong length or leave the sampling interval.
func LoadBenchmarkWorkloads(path string) (*BenchmarkWorkloads, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("benchmark workloads: %w", err)
	}
	var w BenchmarkWorkloads
	if err := json.Unmarshal(raw, &w); err != nil {
		return nil, fmt.Errorf("benchmark workloads %s: %w", path, err)
	}
	if w.RangeRatio != inferencexRangeRatio || w.RequestsPerConcurrency != inferencexRequestsPerConcurrency ||
		w.PrefixLen != 0 {
		return nil, fmt.Errorf("benchmark workloads %s state range ratio %v, %d requests per concurrency "+
			"and prefix %d; the benchmark's are %v, %d and 0", path, w.RangeRatio,
			w.RequestsPerConcurrency, w.PrefixLen, inferencexRangeRatio, inferencexRequestsPerConcurrency)
	}
	if len(w.Points) == 0 {
		return nil, fmt.Errorf("benchmark workloads %s hold no points", path)
	}
	w.path = path
	w.byKey = make(map[[3]int]int, len(w.Points))
	for i, p := range w.Points {
		if err := p.validate(); err != nil {
			return nil, fmt.Errorf("benchmark workloads %s: %w", path, err)
		}
		k := [3]int{p.ISL, p.OSL, p.Concurrency}
		if _, dup := w.byKey[k]; dup {
			return nil, fmt.Errorf("benchmark workloads %s: point %v appears twice", path, k)
		}
		w.byKey[k] = i
	}
	return &w, nil
}

// validate checks one point's shape against the protocol: 10 x concurrency requests, each
// target inside [int(len*0.8), len].
func (p PointLengths) validate() error {
	n := inferencexRequestsPerConcurrency * p.Concurrency
	if p.Concurrency < 1 || len(p.InputLens) != n || len(p.OutputLens) != n {
		return fmt.Errorf("point %d:%d c=%d carries %d input and %d output lengths, not %d",
			p.ISL, p.OSL, p.Concurrency, len(p.InputLens), len(p.OutputLens), n)
	}
	for _, axis := range []struct {
		name  string
		upper int
		lens  []int
	}{{"input", p.ISL, p.InputLens}, {"output", p.OSL, p.OutputLens}} {
		lower := int(float64(axis.upper) * inferencexRangeRatio)
		for i, v := range axis.lens {
			if v < lower || v > axis.upper {
				return fmt.Errorf("point %d:%d c=%d: %s length %d of request %d is outside [%d, %d]",
					p.ISL, p.OSL, p.Concurrency, axis.name, v, i, lower, axis.upper)
			}
		}
	}
	return nil
}

// At returns one point's vectors. A point the fixture does not hold is an error, never a
// fallback to sampled lengths: scoring it would compare a different workload.
func (w *BenchmarkWorkloads) At(isl, osl, concurrency int) (PointLengths, error) {
	i, ok := w.byKey[[3]int{isl, osl, concurrency}]
	if !ok {
		return PointLengths{}, fmt.Errorf("benchmark workloads %s hold no point %d:%d at concurrency %d; "+
			"regenerate with scripts/gen_inferencex_workloads.py", w.path, isl, osl, concurrency)
	}
	return w.Points[i], nil
}
