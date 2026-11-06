package metrics

import (
	"reflect"
	"regexp"
	"strings"
	"testing"
)

var slug = regexp.MustCompile(`^[a-z0-9][a-z0-9_]*$`)

// TestEveryDefinitionIsComplete: each definition states every field a comparison checks,
// and a percentile method exactly when its statistic is a percentile.
func TestEveryDefinitionIsComplete(t *testing.T) {
	for _, d := range All() {
		if !slug.MatchString(d.ID) {
			t.Errorf("%q is not a lowercase slug", d.ID)
		}
		if d.Formula == "" || d.Statistic == "" || d.Population == "" || d.Window == "" ||
			d.Unit == "" || d.Unfinished == "" || d.Description == "" {
			t.Errorf("%s leaves a field empty: %+v", d.ID, d)
		}
		if d.Statistic.Percentile() != (d.PercentileMethod != "") {
			t.Errorf("%s: statistic %s with percentile method %q", d.ID, d.Statistic, d.PercentileMethod)
		}
		if d.Unit == UnitMs && d.Unfinished != UnfinishedExcluded {
			t.Errorf("%s is a latency and must state how unfinished requests are treated", d.ID)
		}
	}
}

// TestBenchmarkClientDefinitions pins the three definitions design §8.7 names: TPOT
// excludes the first token and is per request; ITL is pooled over tokens of requests with
// at least two output tokens (inference-sim #1894).
func TestBenchmarkClientDefinitions(t *testing.T) {
	for _, c := range []struct {
		id  string
		f   Formula
		pop Population
	}{
		{"ttft_mean_ms", FormulaTTFT, PopulationRequests},
		{"tpot_mean_ms", FormulaTPOT, PopulationRequestsTwoTokens},
		{"tpot_p99_ms", FormulaTPOT, PopulationRequestsTwoTokens},
		{"itl_mean_ms", FormulaITL, PopulationTokens},
		{"itl_p90_ms", FormulaITL, PopulationTokens},
	} {
		d, ok := Lookup(c.id)
		if !ok {
			t.Fatalf("%s is not in the registry", c.id)
		}
		if d.Formula != c.f || d.Population != c.pop {
			t.Errorf("%s: formula %s over %s, want %s over %s", c.id, d.Formula, d.Population, c.f, c.pop)
		}
		if !strings.Contains(d.Follows, "vllm/benchmarks/serve.py") {
			t.Errorf("%s does not cite the benchmark client it follows", c.id)
		}
	}
}

// aiperf's inter-token latency is a mean taken per request; the registry pools gaps as
// `vllm bench serve` does, so the ITL family must not claim to follow aiperf.
func TestITLDoesNotFollowAiperf(t *testing.T) {
	for _, d := range All() {
		if d.Formula == FormulaITL && strings.Contains(d.Follows, "aiperf") {
			t.Errorf("%s follows %q, but aiperf's ITL is a different population", d.ID, d.Follows)
		}
	}
}

// TestCompareRefusesDifferentDefinitions is DEV-8's test: a comparison of two figures is
// refused unless their definitions are the same, and the refusal names the field.
func TestCompareRefusesDifferentDefinitions(t *testing.T) {
	tpot, _ := Lookup("tpot_mean_ms")
	itl, _ := Lookup("itl_mean_ms")
	if _, err := Compare(Figure{tpot, 10}, Figure{itl, 12}); err == nil ||
		!strings.Contains(err.Error(), "formula") {
		t.Errorf("tpot against itl: got %v, want a refusal naming formula", err)
	}
	p90, _ := Lookup("ttft_p90_ms")
	p99, _ := Lookup("ttft_p99_ms")
	if _, err := Compare(Figure{p90, 1}, Figure{p99, 2}); err == nil || !strings.Contains(err.Error(), "statistic") {
		t.Errorf("p90 against p99: got %v, want a refusal naming statistic", err)
	}
	// The same definition under another identifier is comparable: the name decides nothing.
	renamed := tpot
	renamed.ID = "my_tpot"
	if d, err := Compare(Figure{tpot, 10}, Figure{renamed, 12.5}); err != nil || d != 2.5 {
		t.Errorf("same definition: got %v, %v; want 2.5, nil", d, err)
	}
	// A definition that differs only in how unfinished requests count is a different metric.
	lower := tpot
	lower.Unfinished = UnfinishedNotApplicable
	if _, err := Compare(Figure{tpot, 1}, Figure{lower, 1}); err == nil || !strings.Contains(err.Error(), "unfinished") {
		t.Errorf("unfinished treatment: got %v, want a refusal", err)
	}
}

// TestEveryPreviouslyReportedMetricIsDefined lists the figures inference-sim's Result
// reported as of be78af83 (cmd/result_build.go:168-237 standardMetrics, and
// cmd/result_inputs.go:262-310 engineSection's step-interval, scheduling-delay,
// gateway-delay and cache figures); each has an identifier here.
func TestEveryPreviouslyReportedMetricIsDefined(t *testing.T) {
	for _, id := range []string{
		"output_tok_s", "output_tok_s_per_gpu", "request_throughput",
		"ttft_mean_ms", "ttft_p50_ms", "ttft_p90_ms", "ttft_p95_ms", "ttft_p99_ms",
		"e2e_mean_ms", "e2e_p50_ms", "e2e_p90_ms", "e2e_p95_ms", "e2e_p99_ms",
		"tpot_mean_ms", "tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms",
		"interactivity_mean", "interactivity_p50", "interactivity_p90",
		"step_interval_mean_ms", "step_interval_p50_ms", "step_interval_p90_ms", "step_interval_p95_ms", "step_interval_p99_ms",
		"scheduling_delay_p99_ms", "gateway_queue_delay_mean_ms", "gateway_queue_delay_p90_ms",
		"cache_hit_rate", "preemptions", "goodput_rps", "slo_attainment",
	} {
		if _, ok := Lookup(id); !ok {
			t.Errorf("%s is not in the registry", id)
		}
	}
}

// Output tokens per second and output tokens per second per GPU share a formula, statistic,
// population, window and unit; they differ in what the rate divides by. They are not
// comparable, and the refusal names the denominator (DEV-8).
func TestPerGPUThroughputIsNotComparableToTheAbsoluteRate(t *testing.T) {
	abs, _ := Lookup("output_tok_s")
	per, _ := Lookup("output_tok_s_per_gpu")
	if _, err := Compare(Figure{abs, 1000}, Figure{per, 125}); err == nil || !strings.Contains(err.Error(), "denominator") {
		t.Fatalf("got %v, want a refusal naming denominator", err)
	}
}

// Every rate states what it divides by, and only a rate does.
func TestEveryRateStatesItsDenominator(t *testing.T) {
	for _, d := range All() {
		perSecond := d.Unit == UnitTokensPerS || d.Unit == UnitRequestsPerS
		if perSecond != (d.Denominator != "") {
			t.Errorf("%s: unit %s with denominator %q", d.ID, d.Unit, d.Denominator)
		}
	}
}

// Comparable names the field where two definitions differ for every field sameMeaning
// reads: the two lists are parallel, and a field in one and not the other would let a
// difference through unnamed or be named when it decides nothing.
func TestComparableAgreesWithSameMeaning(t *testing.T) {
	base, _ := Lookup("output_tok_s")
	variants := map[string]func(*Definition){
		"formula":           func(d *Definition) { d.Formula = FormulaTotalThroughput },
		"statistic":         func(d *Definition) { d.Statistic = StatMean },
		"population":        func(d *Definition) { d.Population = PopulationTokens },
		"window":            func(d *Definition) { d.Window = WindowRun },
		"unit":              func(d *Definition) { d.Unit = UnitCount },
		"percentile_method": func(d *Definition) { d.PercentileMethod = PercentileNumpyLinear },
		"unfinished":        func(d *Definition) { d.Unfinished = UnfinishedExcluded },
		"denominator":       func(d *Definition) { d.Denominator = DenominatorWindowAndGPUs },
	}
	for field, change := range variants {
		o := base
		change(&o)
		if base.sameMeaning(o) {
			t.Errorf("%s: sameMeaning ignores it", field)
		}
		if err := Comparable(base, o); err == nil || !strings.Contains(err.Error(), field) {
			t.Errorf("%s: Comparable gave %v, want a refusal naming it", field, err)
		}
	}
	// Names and prose decide nothing.
	o := base
	o.ID, o.Description, o.Follows = "other", "other", "other"
	if !base.sameMeaning(o) || Comparable(base, o) != nil {
		t.Error("ID, Description and Follows must not affect comparability")
	}
	// Every field of Definition is either in the variants or one of the three names.
	skip := map[string]bool{"ID": true, "Description": true, "Follows": true}
	rt := reflect.TypeOf(Definition{})
	if got, want := rt.NumField()-len(skip), len(variants); got != want {
		t.Errorf("Definition has %d meaning fields and the test varies %d: add the new field here, to sameMeaning and to Comparable", got, want)
	}
}
