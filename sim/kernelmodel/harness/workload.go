package harness

// workload.go states the workload of one scored point exactly as InferenceX's benchmark client
// drove it: the same per-request target lengths, in the same order, at the same concurrency.
//
// # What the real client does
//
// Every scored point was measured by InferenceX's own client, benchmark_serving.py
// (utils/bench_serving/ at the run dates; inferencex-e2e/infx/bench_serving/ since #3494),
// launched by benchmarks/benchmark_lib.sh run_benchmark_serving as
//
//	--dataset-name random --random-input-len ISL --random-output-len OSL
//	--random-range-ratio 0.8 --num-prompts 10*CONC --max-concurrency CONC
//	--request-rate inf --ignore-eos --num-warmups 2*CONC
//
// with no --seed, so --seed is its default 0. The client seeds NumPy's legacy global
// generator with it and draws every request's TARGET lengths before sending anything:
//
//	input_lens  = np.random.randint(int(ISL*0.8), ISL+1, size=10*CONC)
//	output_lens = np.random.randint(int(OSL*0.8), OSL+1, size=10*CONC)
//
// Those vectors are recoverable exactly. scripts/gen_inferencex_workloads.py reproduces the
// calls and writes them, per (ISL, OSL, CONC), to testdata/inferencex-workloads/seed0.json;
// fixture_test.go cross-checks the file against an independent Go implementation of NumPy's
// legacy generator. docs/kernel-exclusive/APPLES-TO-APPLES.md carries the file:line citations.
//
// # What the client sent
//
// The client turns each target into a text prompt, then decodes and re-encodes it (up to ten
// times) to land on the target, and sends whatever length it reached. That sent length is
// recovered offline: scripts/inferencex_prompt_lengths.py reproduces the construction with each
// point's tokenizer at pinned revisions, and promptlengths.go reads the committed evidence. For a
// VERIFIED point -- every pinned revision gives zero mismatch and no server-added token -- the
// sent length is exactly the TARGET the fixture replays, so the simulated prompt is that target.
// A point the evidence file does not hold (e.g. a gated tokenizer whose run left no engine log) is
// not verified and carries that gap; docs/kernel-exclusive/APPLES-TO-APPLES.md "The sent prompt
// lengths" has the file:line citations. With --ignore-eos the output length is the target exactly
// (max_tokens = output_len).

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

// The benchmark protocol's constants, from the source cited above. Named so the fixture loader
// and the tests check the file against the same values the protocol states.
const (
	// inferencexRangeRatio is RANDOM_RANGE_RATIO, fixed at 0.8 in every benchmark workflow.
	inferencexRangeRatio = 0.8
	// inferencexRequestsPerConcurrency is --num-prompts / --max-concurrency.
	inferencexRequestsPerConcurrency = 10
	// InferenceXWarmupsPerConcurrency is --num-warmups / --max-concurrency: the benchmark sends
	// this many copies of request 0 per unit of concurrency before the main phase. Run
	// reproduces the state they leave rather than replaying them; see Run.
	InferenceXWarmupsPerConcurrency = 2
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
