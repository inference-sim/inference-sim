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

// AISimulate's replay constants, from the source cited above. Named rather than inlined so
// parity_test.go asserts against the same values the harness uses.
const (
	aisimulateRangeRatio      = 0.8
	aisimulateRequestsPerUser = 10
	aisimulateLengthSeed      = 0
	aisimulateArrivalSeed     = 42
)

// Workload is the sampling interval and request count for one (label, concurrency) point.
type Workload struct {
	ISLLow, ISLHigh int
	OSLLow, OSLHigh int
	RequestCount    int
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
