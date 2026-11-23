package cmd

import (
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/spf13/pflag"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
)

// The Student t interval of 1..5 at 95%: mean 3, s = sqrt(2.5), t(0.975, 4) = 2.7764451.
func TestTInterval_MatchesTheTextbookValue(t *testing.T) {
	mean, low, high := tInterval([]float64{1, 2, 3, 4, 5}, 0.95)
	half := 2.7764451051977987 * math.Sqrt(2.5) / math.Sqrt(5)
	if math.Abs(mean-3) > 1e-12 || math.Abs(low-(3-half)) > 1e-9 || math.Abs(high-(3+half)) > 1e-9 {
		t.Errorf("tInterval = (%v, %v, %v), want (3, %v, %v)", mean, low, high, 3-half, 3+half)
	}
}

// A law, not a value: the interval is centred on the mean, widens with the level, and is
// degenerate when every replication agrees.
func TestTInterval_Laws(t *testing.T) {
	vals := []float64{10, 12, 9, 11, 14, 10}
	mean, l90, h90 := tInterval(vals, 0.90)
	_, l99, h99 := tInterval(vals, 0.99)
	if math.Abs((l90+h90)/2-mean) > 1e-9 {
		t.Errorf("the interval is not centred on the mean")
	}
	if !(l99 < l90 && h99 > h90) {
		t.Errorf("a 99%% interval [%v, %v] must contain the 90%% interval [%v, %v]", l99, h99, l90, h90)
	}
	_, l, h := tInterval([]float64{4, 4, 4, 4, 4}, 0.95)
	if l != 4 || h != 4 {
		t.Errorf("replications that agree give a zero-width interval, got [%v, %v]", l, h)
	}
}

// DEV-6: replication seeds come from a named stream of the run's seed -- the same seed gives the
// same list, a different base seed a different one, and no two replications share a seed.
func TestReplicationSeed_IsAStreamOfTheRunSeed(t *testing.T) {
	seen := map[int64]bool{}
	for k := range 40 {
		s := replicationSeed(7, k)
		if s != replicationSeed(7, k) {
			t.Fatalf("replication %d: the seed is not deterministic", k)
		}
		if s < 0 {
			t.Fatalf("replication %d: seed %d is negative", k, s)
		}
		if seen[s] {
			t.Fatalf("replication %d repeats a seed", k)
		}
		seen[s] = true
		if s == replicationSeed(8, k) {
			t.Errorf("replication %d: base seeds 7 and 8 give the same replication seed", k)
		}
	}
}

func TestChildArgs_CarryTheRunAndNotTheDriver(t *testing.T) {
	fs := pflag.NewFlagSet("run", pflag.ContinueOnError)
	fs.Int("replications", 1, "")
	fs.Int("max-replications", 30, "")
	fs.Float64("interval-width", 0.1, "")
	fs.String("format", "json", "")
	fs.String("output", "", "")
	fs.String("view", "", "")
	fs.Int64("seed", 42, "")
	fs.Int("num-requests", 100, "")
	fs.StringSlice("catalog-overlay", nil, "")
	if err := fs.Parse([]string{"--replications", "5", "--max-replications", "9", "--interval-width", "0.2",
		"--format", "yaml", "--output", "x.json", "--view", "summary", "--seed", "7", "--num-requests", "30",
		"--catalog-overlay", "a", "--catalog-overlay", "b"}); err != nil {
		t.Fatal(err)
	}
	got := strings.Join(childArgs(fs, 12345), " ")
	want := "run --format json --seed=12345 --catalog-overlay=a --catalog-overlay=b --num-requests=30"
	if got != want {
		t.Errorf("childArgs = %q, want %q", got, want)
	}
}

func replicationDocs(values ...float64) ([]*result.Result, []int64) {
	var docs []*result.Result
	var seeds []int64
	for i, v := range values {
		docs = append(docs, &result.Result{Metrics: map[string]result.Figure{
			"ttft_p99_ms": {Value: v, Samples: 10},
			"throughput":  {Value: 5, Samples: 3},
		}})
		seeds = append(seeds, int64(i))
	}
	return docs, seeds
}

func TestAggregateReplications_StopsOnWidthOrLimit(t *testing.T) {
	plan := intervalPlan{Min: 5, Max: 8, Width: 0.1}

	// Replications that agree: the widest figure has width zero and the rule stops on width.
	docs, seeds := replicationDocs(100, 100, 100, 100, 100)
	agg, stopped, err := aggregateReplications(docs, seeds, plan)
	if err != nil || !stopped {
		t.Fatalf("stopped = %v, err = %v; want a stop on width", stopped, err)
	}
	iv := agg.Estimate.Interval
	if agg.Estimate.Replications != 5 || iv.StoppedBy != result.StoppedByWidth || iv.RelativeWidthReached != 0 {
		t.Errorf("estimate = %+v interval = %+v", agg.Estimate, iv)
	}
	if f := agg.Metrics["ttft_p99_ms"]; f.Value != 100 || f.Samples != 50 || f.Interval == nil {
		t.Errorf("figure = %+v, want the mean, the summed samples and an interval", f)
	}

	// Scattered replications below the limit: not stopped, and the first document is untouched.
	docs, seeds = replicationDocs(10, 100, 20, 90, 30)
	if _, stopped, err := aggregateReplications(docs, seeds, plan); err != nil || stopped {
		t.Errorf("stopped = %v, err = %v; want to continue below the limit", stopped, err)
	}
	if docs[0].Metrics["ttft_p99_ms"].Interval != nil || docs[0].Estimate.Replications != 0 {
		t.Errorf("aggregating changed the first replication's document")
	}

	// The same scatter at the limit: stopped by the limit, with the width reached stated.
	docs, seeds = replicationDocs(10, 100, 20, 90, 30, 80, 40, 70)
	agg, stopped, err = aggregateReplications(docs, seeds, plan)
	if err != nil || !stopped {
		t.Fatalf("stopped = %v, err = %v; want a stop at the limit", stopped, err)
	}
	if iv := agg.Estimate.Interval; iv.StoppedBy != result.StoppedByLimit || iv.RelativeWidthReached <= plan.Width {
		t.Errorf("interval = %+v, want a stop by limit above the target width", iv)
	}
}

// Replications that report different figures are not replications of one measurement.
func TestAggregateReplications_RefusesDifferentFigures(t *testing.T) {
	docs, seeds := replicationDocs(1, 2, 3, 4, 5)
	delete(docs[3].Metrics, "throughput")
	_, _, err := aggregateReplications(docs, seeds, intervalPlan{Min: 5, Max: 8, Width: 0.1})
	if err == nil || !strings.Contains(err.Error(), "replication 4") {
		t.Errorf("err = %v, want a refusal naming replication 4", err)
	}
}

func intervalArgs(extra ...string) []string {
	return resultRunArgs(append([]string{"--replications", "5", "--max-replications", "5"}, extra...)...)
}

// An interval on request (DEV-11): five replications under their named-stream seeds, every
// figure the mean with an interval that contains it, and the document stating how it was made.
// The sections other than metrics are the first replication's, which a single run at that
// replication's seed reproduces; and the whole document repeats byte for byte (INV-6).
func TestRun_ReplicationsAskForAnInterval(t *testing.T) {
	out, stderr, err := runKernelCLI(t, intervalArgs()...)
	if err != nil {
		t.Fatalf("run --replications 5: %v\n%s", err, lastLines(stderr, 8))
	}
	r := parseResult(t, out)
	e := r.Estimate
	if e.Replications != 5 || e.NoInterval != "" || e.Interval == nil {
		t.Fatalf("estimate = %+v, want 5 replications with an interval and no no-interval statement", e)
	}
	iv := e.Interval
	if iv.StoppedBy != result.StoppedByWidth && iv.StoppedBy != result.StoppedByLimit {
		t.Errorf("stopped_by = %q", iv.StoppedBy)
	}
	for k, s := range iv.Seeds {
		if want := replicationSeed(7, k); s != want {
			t.Errorf("seed %d = %d, want %d from the named stream", k, s, want)
		}
	}
	if r.Frame.Generator.Seed != 7 {
		t.Errorf("frame seed = %d, want the run's own seed 7", r.Frame.Generator.Seed)
	}
	// The options are the interval request's own command line, not a replication's.
	var seedOpt string
	for _, o := range r.Input.Options {
		if o.Name == "seed" {
			seedOpt = o.Value
		}
	}
	if seedOpt != "7" {
		t.Errorf("input.options seed = %q, want the run's --seed 7 and not a replication's", seedOpt)
	}
	if len(r.Metrics) == 0 {
		t.Fatal("no figures")
	}
	for id, f := range r.Metrics {
		if f.Interval == nil || f.Interval.Low > f.Value || f.Value > f.Interval.High {
			t.Errorf("%s: value %v interval %+v", id, f.Value, f.Interval)
		}
	}

	// The first replication is a single run at its seed.
	first, stderr, err := runKernelCLI(t, resultRunArgs("--seed", strconv.FormatInt(iv.Seeds[0], 10))...)
	if err != nil {
		t.Fatalf("first replication alone: %v\n%s", err, lastLines(stderr, 8))
	}
	fr := parseResult(t, first)
	if got, want := r.Requests, fr.Requests; got.Generated != want.Generated || got.Completed != want.Completed {
		t.Errorf("requests = %+v, want the first replication's %+v", got, want)
	}

	again, stderr, err := runKernelCLI(t, intervalArgs()...)
	if err != nil {
		t.Fatalf("second run: %v\n%s", err, lastLines(stderr, 8))
	}
	if withoutRuntime(t, again) != withoutRuntime(t, out) {
		t.Error("two runs of the same interval request differ outside runtime (INV-6)")
	}
}

func TestRun_AnIntervalRequestThatCannotBeHonouredIsRefused(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		msg  string
	}{
		{"two replications", resultRunArgs("--replications", "2"), "at least 5"},
		{"limit below the request", resultRunArgs("--replications", "6", "--max-replications", "5"), "below --replications"},
		{"width without replications", resultRunArgs("--interval-width", "0.2"), "needs --replications"},
		{"a trace export", resultRunArgs("--replications", "5", "--trace-output", "t"), "cannot be combined"},
		{"zero width", resultRunArgs("--replications", "5", "--interval-width", "0"), "positive fraction"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stdout, stderr, err := runKernelCLI(t, tc.args...)
			if err == nil || !strings.Contains(stderr, tc.msg) {
				t.Errorf("err = %v, want a refusal containing %q\n%s", err, tc.msg, lastLines(stderr, 5))
			}
			if stdout != "" {
				t.Errorf("a refused request wrote to stdout:\n%s", stdout)
			}
		})
	}
}
