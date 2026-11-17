// sim/saturation/settle.go: the one steady-state detector (DEV-10, design 8.8).
//
// A steady-state run is classified SETTLED, OVERLOADED or INCOMPLETE over its measured window,
// and only a settled run gives a latency figure for steady state. The detector replaces the
// roster (composite, threshold, backlog-drift, peak-rate; inference-sim#1616, #1617, #1618,
// #1693, #1927), whose members disagreed on the same run and answered a different question each.
//
// The statistic is the trend of the number of requests in the system. A system that has
// settled has no trend in its backlog; an open system with load above one has a backlog that
// grows without bound, whatever its service times. A closed system (fixed concurrency) holds
// its backlog at the concurrency, so it reads settled unless its backlog itself drifts.
//
//  1. N(t), the requests in the system, is swept from every request's arrival and departure.
//     A request still in the system at the end of the run stays in it: counting only the
//     requests that finished would make an overloaded run's backlog fall at its end.
//  2. The window is cut into Batches equal slices, and each slice's time-average of N is its
//     batch mean. The trend is read from these Batches numbers, not from N's every event, so
//     the autocorrelation of a queue is spread over the batches and removed by pre-whitening.
//  3. The trend test is the Mann-Kendall test on the batch means after trend-free
//     pre-whitening of lag-1 autocorrelation (Yue and Pilon 2002), one-sided upward. The run
//     is OVERLOADED when the test rejects at Alpha and the growth Sen's slope implies across
//     the window is also at least GrowthFloor requests and a quarter of the mean backlog: a
//     significant drift of a hundredth of a request is not an overload.
//  4. Two conditions make the window INCOMPLETE, unless the backlog visibly grows: too few
//     completions to fill the batches (MinPerBatch each; a server that completes so few while
//     its backlog grows is overloaded, and says so), and a latency tail so heavy that
//     its variance is infinite. The test is Hill's estimate of the tail index of the
//     completion latencies, from their largest tenth (below 2.2) or largest twentieth (below
//     2.0); a tail index below 2 means an infinite variance, the batch means of a window of
//     practical length do not settle, and its excursions look like trends. The estimate is
//     noisy: the table in settle_test.go states how often a Pareto tail of index 1.5 is
//     caught at each window length, and how seldom an exponential one is mistaken for it.
//
// What the detector promises is measured, not assumed: settle_test.go simulates single-server
// queues whose behaviour is known exactly (open M/M/1 and M/G/1 at loads 0.5 to 1.5, a closed
// queue, Pareto service) and states the false-alarm rate (a settled system read as overloaded)
// and the power (an overloaded one read as overloaded) at each window length. the design states
// the table; the test fails if the detector no longer meets it.
package saturation

import (
	"fmt"
	"math"
	"os"
	"sort"
)

// State is the classification of a measured window.
type State string

const (
	Settled    State = "SETTLED"
	Overloaded State = "OVERLOADED"
	Incomplete State = "INCOMPLETE"
)

// Job is one request's time in the system.
type Job struct {
	// ArrivalUs is when it entered the system.
	ArrivalUs int64
	// DepartUs is when it left, or, for a request still in the system at the end of the run,
	// the end of the run (Open).
	DepartUs int64
	Open     bool
	// LatencyMs is a completed request's end-to-end latency; zero for a request that did not
	// complete.
	LatencyMs float64
}

// Options are the detector's constants. Zero values take the defaults below.
type Options struct {
	Batches     int     // slices of the window; default 20
	Alpha       float64 // one-sided level of the trend test; default 0.01
	MinPerBatch int     // completions each batch needs on average; default 10
	GrowthFloor float64 // least backlog growth across the window to call overload; default 2
	TailIndex   float64 // Hill index of the largest tenth below which the tail is too heavy; default 2.2 (2.0 of the largest twentieth)
}

func (o Options) withDefaults() Options {
	if o.Batches == 0 {
		o.Batches = 20
	}
	if o.Alpha == 0 {
		o.Alpha = 0.01
	}
	if o.MinPerBatch == 0 {
		o.MinPerBatch = 10
	}
	if o.GrowthFloor == 0 {
		o.GrowthFloor = 2
	}
	if o.TailIndex == 0 {
		o.TailIndex = 2.2
	}
	return o
}

// Verdict is the detector's answer and the numbers behind it.
type Verdict struct {
	State  State  `json:"state"`
	Reason string `json:"reason"`
	// Completions is the completed requests that departed inside the window.
	Completions int `json:"completions"`
	// MeanInSystem is the time-average of the requests in the system over the window.
	MeanInSystem float64 `json:"mean_in_system"`
	// GrowthRequests is the growth of the backlog across the window Sen's slope implies; P is
	// the one-sided p-value of the trend test. Both zero when the trend was not read.
	GrowthRequests float64 `json:"growth_requests"`
	P              float64 `json:"p_value"`
	// TailIndex is Hill's index of the completion latencies' tail, from the largest tenth.
	TailIndex float64 `json:"tail_index"`
}

// Settle classifies the window [startUs, endUs) of a run whose requests are jobs.
func Settle(jobs []Job, startUs, endUs int64, opt Options) Verdict {
	o := opt.withDefaults()
	if endUs <= startUs {
		return Verdict{State: Incomplete, Reason: "the measured window has no length"}
	}
	var lat []float64
	for _, j := range jobs {
		if !j.Open && j.LatencyMs > 0 && j.DepartUs >= startUs && j.DepartUs < endUs {
			lat = append(lat, j.LatencyMs)
		}
	}
	v := Verdict{Completions: len(lat)}
	means := batchMeans(jobs, startUs, endUs, o.Batches)
	var sum float64
	for _, m := range means {
		sum += m
	}
	v.MeanInSystem = sum / float64(len(means))
	z, slope := pwMannKendall(means)
	v.P = 1 - normalCDF(z)
	v.GrowthRequests = slope * float64(len(means)-1)
	grows := v.P < o.Alpha && v.GrowthRequests >= math.Max(o.GrowthFloor, 0.25*v.MeanInSystem)
	overloaded := func() Verdict {
		v.State, v.Reason = Overloaded, fmt.Sprintf("the backlog grows by %.1f requests across the window (one-sided p %.2g)", v.GrowthRequests, v.P)
		return v
	}
	if len(lat) < o.MinPerBatch*o.Batches {
		// Too few completions to read the latency tail. A backlog that visibly grows is still
		// an overload: the server is not keeping up, which is why so few completed.
		if grows {
			return overloaded()
		}
		v.State, v.Reason = Incomplete, fmt.Sprintf("%d completions in the window, fewer than the %d the %d batches need", len(lat), o.MinPerBatch*o.Batches, o.Batches)
		return v
	}
	v.TailIndex = hill(lat, 0.10)
	if v.TailIndex < o.TailIndex || hill(lat, 0.05) < o.TailIndex*2.0/2.2 {
		v.State, v.Reason = Incomplete, fmt.Sprintf("the completion latencies have tail index %.2f, near or below 2: their variance is infinite and the window is too short to settle", v.TailIndex)
		return v
	}
	if grows {
		return overloaded()
	}
	v.State, v.Reason = Settled, "the backlog shows no significant growth across the window"
	return v
}

// batchMeans are the time-averages of the requests in the system over k equal slices of
// [startUs, endUs).
func batchMeans(jobs []Job, startUs, endUs int64, k int) []float64 {
	type edge struct {
		at int64
		d  int
	}
	edges := make([]edge, 0, 2*len(jobs))
	for _, j := range jobs {
		if j.DepartUs <= startUs || j.ArrivalUs >= endUs {
			continue
		}
		edges = append(edges, edge{j.ArrivalUs, 1}, edge{j.DepartUs, -1})
	}
	sort.Slice(edges, func(a, b int) bool { return edges[a].at < edges[b].at })
	area := make([]float64, k)
	width := float64(endUs-startUs) / float64(k)
	// add accrues n requests held over [a, b) to the slices it covers.
	add := func(a, b int64, n int) {
		a, b = max(a, startUs), min(b, endUs)
		if b <= a || n == 0 {
			return
		}
		for i := int(float64(a-startUs) / width); i < k; i++ {
			lo, hi := float64(startUs)+float64(i)*width, float64(startUs)+float64(i+1)*width
			s, e := math.Max(float64(a), lo), math.Min(float64(b), hi)
			if e > s {
				area[i] += float64(n) * (e - s)
			}
			if hi >= float64(b) {
				break
			}
		}
	}
	n, prev := 0, startUs
	for _, e := range edges {
		add(prev, e.at, n)
		n += e.d
		prev = max(prev, e.at)
	}
	add(prev, endUs, n)
	for i := range area {
		area[i] /= width
	}
	return area
}

// hill is Hill's estimate of the tail index of xs from its largest frac (at least ten values).
func hill(xs []float64, frac float64) float64 {
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	k := max(10, int(float64(len(s))*frac))
	if k >= len(s) {
		return math.Inf(1)
	}
	ref := s[len(s)-k-1]
	var sum float64
	for _, x := range s[len(s)-k:] {
		sum += math.Log(x / ref)
	}
	if sum <= 0 {
		return math.Inf(1)
	}
	return float64(k) / sum
}

// pwMannKendall is the z statistic of the Mann-Kendall test on xs after trend-free
// pre-whitening of its lag-1 autocorrelation, and Sen's slope of the original series.
func pwMannKendall(xs []float64) (z, slope float64) {
	slope = senSlope(xs)
	n := len(xs)
	detr := make([]float64, n)
	for i, x := range xs {
		detr[i] = x - slope*float64(i)
	}
	r1 := lag1(detr)
	series := xs
	if r1 > 0.1 {
		series = make([]float64, 0, n-1)
		for i := 1; i < n; i++ {
			series = append(series, detr[i]-r1*detr[i-1]+slope*float64(i))
		}
	}
	return mannKendallZ(series), slope
}

func senSlope(xs []float64) float64 {
	var sl []float64
	for i := range xs {
		for j := i + 1; j < len(xs); j++ {
			sl = append(sl, (xs[j]-xs[i])/float64(j-i))
		}
	}
	sort.Float64s(sl)
	if len(sl) == 0 {
		return 0
	}
	return sl[len(sl)/2]
}

func lag1(xs []float64) float64 {
	var mean float64
	for _, x := range xs {
		mean += x
	}
	mean /= float64(len(xs))
	var num, den float64
	for i, x := range xs {
		den += (x - mean) * (x - mean)
		if i > 0 {
			num += (x - mean) * (xs[i-1] - mean)
		}
	}
	if den == 0 {
		return 0
	}
	return num / den
}

// mannKendallZ is the normal score of the Mann-Kendall S, with the tie correction.
func mannKendallZ(xs []float64) float64 {
	n := len(xs)
	var s float64
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			switch {
			case xs[j] > xs[i]:
				s++
			case xs[j] < xs[i]:
				s--
			}
		}
	}
	counts := map[float64]int{}
	for _, x := range xs {
		counts[x]++
	}
	v := float64(n) * float64(n-1) * float64(2*n+5)
	for _, t := range counts {
		v -= float64(t) * float64(t-1) * float64(2*t+5)
	}
	v /= 18
	switch {
	case v <= 0:
		return 0
	case s > 0:
		return (s - 1) / math.Sqrt(v)
	case s < 0:
		return (s + 1) / math.Sqrt(v)
	}
	return 0
}

func normalCDF(z float64) float64 { return 0.5 * math.Erfc(-z/math.Sqrt2) }

// ValidateReportPath fails early on a report path that cannot be written, so a run does not
// finish before it learns so. An empty path is fine.
func ValidateReportPath(path string) error {
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE, 0o644)
	if err != nil {
		return fmt.Errorf("--saturation-report %q is not writable: %w", path, err)
	}
	return f.Close()
}
