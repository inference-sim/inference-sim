package routertable

import (
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// run is one row of a sweep's CSV.
type run struct {
	shape, workload, policy string
	concurrency             int
	seed                    string
	requests, completed     string
	fig                     map[string]float64
}

// cell is one (shape, workload, concurrency): the runs of each policy.
type cell struct {
	shape, workload string
	concurrency     int
	by              map[string][]run
}

const repoRoot = "../../.."

// readRuns reads a sweep's CSV and checks what the ratios assume: every policy has the same three
// seeds in every cell and, when complete is set, every run completed all its requests. The
// bounds' file must be complete; the low-concurrency check keeps the runs that ended with requests
// timed out, which it does not use for a ratio.
func readRuns(t *testing.T, path string, complete bool) []*cell {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(repoRoot, path))
	if err != nil {
		t.Fatal(err)
	}
	rows, err := csv.NewReader(strings.NewReader(string(b))).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	head := rows[0]
	col := map[string]int{}
	for i, h := range head {
		col[h] = i
	}
	cells := map[string]*cell{}
	var order []*cell
	for _, r := range rows[1:] {
		conc, err := strconv.Atoi(r[col["concurrency"]])
		if err != nil {
			t.Fatal(err)
		}
		x := run{shape: r[col["shape"]], workload: r[col["workload"]], policy: r[col["policy"]], concurrency: conc,
			seed: r[col["seed"]], requests: r[col["requests"]], completed: r[col["completed"]], fig: map[string]float64{}}
		for _, h := range head {
			switch h {
			case "shape", "workload", "concurrency", "policy", "seed", "requests", "completed":
			default:
				v, err := strconv.ParseFloat(r[col[h]], 64)
				if err != nil || math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
					t.Fatalf("%s %v: %s is %q, want a positive finite number", path, r, h, r[col[h]])
				}
				x.fig[h] = v
			}
		}
		if complete && x.requests != x.completed {
			t.Fatalf("%s: %s of %s requests completed in %v; a run that did not finish has no row", path, x.completed, x.requests, r)
		}
		k := fmt.Sprintf("%s/%s/%d", x.shape, x.workload, x.concurrency)
		c := cells[k]
		if c == nil {
			c = &cell{shape: x.shape, workload: x.workload, concurrency: conc, by: map[string][]run{}}
			cells[k] = c
			order = append(order, c)
		}
		c.by[x.policy] = append(c.by[x.policy], x)
	}
	var seeds map[string]bool
	for _, c := range order {
		for p, rs := range c.by {
			got := map[string]bool{}
			for _, r := range rs {
				got[r.seed] = true
			}
			if len(rs) != 3 || len(got) != 3 {
				t.Fatalf("%s: cell %s/%s/%d policy %s has %d runs over %d seeds, want 3 distinct seeds", path, c.shape, c.workload, c.concurrency, p, len(rs), len(got))
			}
			if seeds == nil {
				seeds = got
			}
			for s := range seeds {
				if !got[s] {
					t.Fatalf("%s: cell %s/%s/%d policy %s lacks seed %s", path, c.shape, c.workload, c.concurrency, p, s)
				}
			}
		}
	}
	return order
}

func mean(rs []run, metric string) float64 {
	s := 0.0
	for _, r := range rs {
		s += r.fig[metric]
	}
	return s / float64(len(rs))
}

// recompute is the bound's definition: the largest, over the regime's cells and the
// alternatives, of the larger of the two seed means over the smaller, rounded up to two decimals.
// It returns the number of cells and the shapes the regime's cells cover.
func recompute(cells []*cell, policy string, r Ratio) (ratio float64, n int, shapes []string, err error) {
	have := map[string]bool{}
	worst := 1.0
	for _, c := range cells {
		g := r.Regime
		if string(g.Workload) != c.workload || c.concurrency < g.ConcurrencyMin || c.concurrency > g.ConcurrencyMax {
			continue
		}
		in := false
		for _, s := range g.Shapes {
			in = in || s == c.shape
		}
		if !in {
			continue
		}
		n++
		have[c.shape] = true
		rs := c.by[policy]
		if len(rs) == 0 {
			return 0, 0, nil, fmt.Errorf("the evidence has no runs of policy %q in %s/%s/%d", policy, c.shape, c.workload, c.concurrency)
		}
		a := mean(rs, r.Metric)
		for _, o := range r.Against {
			alt := c.by[o]
			if len(alt) == 0 {
				return 0, 0, nil, fmt.Errorf("the evidence has no runs of alternative %q in %s/%s/%d", o, c.shape, c.workload, c.concurrency)
			}
			b := mean(alt, r.Metric)
			worst = math.Max(worst, math.Max(a/b, b/a))
		}
	}
	for s := range have {
		shapes = append(shapes, s)
	}
	return math.Ceil(worst*100-1e-9) / 100, n, shapes, nil
}

func loadCommitted(t *testing.T) map[string]*Entry {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(repoRoot, "catalog", "routers", "*.yaml"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no entries under catalog/routers: %v", err)
	}
	out := map[string]*Entry{}
	for _, path := range paths {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var e Entry
		if err := yaml.Unmarshal(b, &e); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		e.Name = strings.TrimSuffix(filepath.Base(path), ".yaml")
		out[e.Name] = &e
	}
	return out
}

// Every bound of a committed entry is what its evidence file gives: the file has the bytes the
// bound names, the cells of the regime exist with every seed, the policy and the alternatives were
// run, and the largest ratio recomputed here equals the one the entry states. A bound edited by hand
// or an evidence file regenerated without updating the entry fails here.
func TestTheCommittedBoundsAreTheirEvidences(t *testing.T) {
	entries := loadCommitted(t)
	bounded := 0
	for name, e := range entries {
		if e.Bound == nil {
			continue
		}
		bounded++
		t.Run(name, func(t *testing.T) {
			b, err := os.ReadFile(filepath.Join(repoRoot, e.Bound.Evidence.File))
			if err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(b)
			if got := hex.EncodeToString(sum[:]); got != e.Bound.Evidence.SHA256 {
				t.Fatalf("%s has sha256 %s, but the bound names %s", e.Bound.Evidence.File, got, e.Bound.Evidence.SHA256)
			}
			cells := readRuns(t, e.Bound.Evidence.File, true)
			for i, r := range e.Bound.Ratios {
				got, n, shapes, err := recompute(cells, string(e.Policy), r)
				if err != nil {
					t.Fatalf("ratios[%d]: %v", i, err)
				}
				if n == 0 {
					t.Errorf("ratios[%d]: no cell of the evidence is in the regime", i)
				}
				if math.Abs(got-r.MaxRatio) > 1e-9 {
					t.Errorf("ratios[%d] (%s, workload %s, concurrency %d-%d): the entry states %.2f, the evidence gives %.2f", i, r.Metric,
						r.Regime.Workload, r.Regime.ConcurrencyMin, r.Regime.ConcurrencyMax, r.MaxRatio, got)
				}
				if len(shapes) != len(r.Regime.Shapes) {
					t.Errorf("ratios[%d]: the regime names shapes %v, the evidence has cells for %v", i, r.Regime.Shapes, shapes)
				}
			}
		})
	}
	if bounded == 0 {
		t.Fatal("no committed entry carries a bound")
	}
}

// The recomputation is the check, so it must fail when the file or the entry moves: a changed figure,
// a changed ratio and a missing alternative each change what it returns.
func TestTheRecomputationNoticesAChangedFigureOrAStatedRatio(t *testing.T) {
	cells := []*cell{{shape: "1P1D", workload: "balanced", concurrency: 16, by: map[string][]run{
		"round-robin": {{fig: map[string]float64{"m": 100}}, {fig: map[string]float64{"m": 110}}, {fig: map[string]float64{"m": 90}}},
		"alt":         {{fig: map[string]float64{"m": 150}}, {fig: map[string]float64{"m": 150}}, {fig: map[string]float64{"m": 150}}},
	}}}
	r := Ratio{Metric: "m", Against: []string{"alt"}, Regime: Regime{Workload: WorkloadBalanced, ConcurrencyMin: 16, ConcurrencyMax: 16, Shapes: []string{"1P1D"}}}
	got, n, _, err := recompute(cells, "round-robin", r)
	if err != nil || n != 1 || got != 1.50 {
		t.Fatalf("got %v over %d cells, %v; want 1.50 over 1", got, n, err)
	}
	cells[0].by["alt"][0].fig["m"] = 300 // mean 200: the ratio is now 2.00
	if got, _, _, _ := recompute(cells, "round-robin", r); got != 2.00 {
		t.Errorf("a changed figure gave %v, want 2.00", got)
	}
	r.Against = []string{"missing"}
	if _, _, _, err := recompute(cells, "round-robin", r); err == nil {
		t.Error("an alternative with no runs must be an error, not a ratio of 1")
	}
	// The ratio is symmetric: the policy below or above the alternative gives the same number.
	cells[0].by["alt"][0].fig["m"] = 150
	r.Against = []string{"alt"}
	cells[0].by["round-robin"] = []run{{fig: map[string]float64{"m": 300}}}
	if got, _, _, _ := recompute(cells, "round-robin", r); got != 2.00 {
		t.Errorf("the policy above the alternative gave %v, want 2.00", got)
	}
}

// The regime of each ratio names the model and hardware the evidence ran: the scenario the sweep
// writes states the same two, so a bound cannot claim a part it did not run.
func TestTheBoundsNameTheModelAndHardwareTheSweepRan(t *testing.T) {
	b, err := os.ReadFile(filepath.Join(repoRoot, "registry", "docs", "router-sensitivity", "scenario.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(b)
	for name, e := range loadCommitted(t) {
		if e.Bound == nil {
			continue
		}
		for i, r := range e.Bound.Ratios {
			if !strings.Contains(text, "\nmodel: "+r.Regime.Model+"\n") || !strings.Contains(text, "  hardware: "+r.Regime.Hardware+"\n") {
				t.Errorf("%s ratios[%d]: the regime names %s on %s, which scenario.yaml does not run", name, i, r.Regime.Model, r.Regime.Hardware)
			}
		}
	}
}

// meanOf returns the seed mean of one policy in one cell.
func meanOf(t *testing.T, cells []*cell, shape, workload string, conc int, policy, metric string) float64 {
	t.Helper()
	for _, c := range cells {
		if c.shape == shape && c.workload == workload && c.concurrency == conc {
			if rs := c.by[policy]; len(rs) > 0 {
				return mean(rs, metric)
			}
		}
	}
	t.Fatalf("no runs of %s in %s/%s/%d", policy, shape, workload, conc)
	return 0
}

// The round-robin entry says the layer 10 prediction, that llm-d's shared counter gives a prefill
// pool of even size every other endpoint, was not reproduced. The evidence it cites shows it: with a
// workload that saturates the prefill pool, round-robin with two prefill endpoints serves more
// than the one-endpoint rate by nearly the factor of two (so both endpoints are used), and
// with four serves more than three times it. A pool that used one endpoint would serve about
// the one-endpoint rate.
func TestTheEvidenceDoesNotShowAStarvedPrefillPool(t *testing.T) {
	cells := readRuns(t, "registry/docs/router-sensitivity/runs.csv", true)
	for _, conc := range []int{16, 64, 256, 1024} {
		one := meanOf(t, cells, "1P3D", "prefill-bound", conc, "round-robin", "request_throughput")
		two := meanOf(t, cells, "2P3D", "prefill-bound", conc, "round-robin", "request_throughput")
		four := meanOf(t, cells, "4P1D", "prefill-bound", conc, "round-robin", "request_throughput")
		if two < 1.8*one {
			t.Errorf("concurrency %d: two prefill endpoints served %.2f req/s, one served %.2f: an endpoint is unused", conc, two, one)
		}
		if four < 3.0*one {
			t.Errorf("concurrency %d: four prefill endpoints served %.2f req/s, one served %.2f: an endpoint is unused", conc, four, one)
		}
	}
	// At low concurrency with two prefill endpoints (alternation.csv, where the counter would
	// alternate most strictly): from four clients the pool serves more than the one-endpoint rate.
	alt := readRuns(t, "registry/docs/router-sensitivity/alternation.csv", false)
	one := meanOf(t, cells, "1P3D", "prefill-bound", 1024, "round-robin", "request_throughput")
	for _, conc := range []int{4, 8} {
		if got := meanOf(t, alt, "2P3D", "prefill-bound", conc, "round-robin", "request_throughput"); got < 1.2*one {
			t.Errorf("concurrency %d: two prefill endpoints served %.2f req/s against %.2f for one", conc, got, one)
		}
	}
}

// The kv entry says the policies that read load differ from each other by up to 1.70 (balanced) and
// 1.79 (decode-bound) in median time to first token at 64 to 256 clients. Recompute it.
func TestTheKVEntryQuotesTheSpreadTheEvidenceGives(t *testing.T) {
	cells := readRuns(t, "registry/docs/router-sensitivity/runs.csv", true)
	for workload, want := range map[WorkloadKind]float64{WorkloadBalanced: 1.70, WorkloadDecodeBound: 1.79} {
		r := Ratio{Metric: "ttft_p50_ms", Against: []string{"least-active-requests"},
			Regime: Regime{Workload: workload, ConcurrencyMin: 64, ConcurrencyMax: 256, Shapes: []string{"1P3D", "2P3D", "3P2D", "4P1D"}}}
		got, _, _, err := recompute(cells, "default", r)
		if err != nil || math.Abs(got-want) > 1e-9 {
			t.Errorf("%s: default against least-active-requests is %.2f (%v), the kv entry says %.2f", workload, got, err, want)
		}
	}
	e := loadCommitted(t)["dynamo-kv"]
	if e == nil || e.Policy != PolicyNone || !strings.Contains(e.Reason, "1.70 (balanced) and 1.79 (decode-bound)") {
		t.Errorf("dynamo-kv must be a none entry quoting the spread: %+v", e)
	}
}
