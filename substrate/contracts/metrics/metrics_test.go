package metrics

import (
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
