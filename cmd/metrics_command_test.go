package cmd

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/metrics"
	"github.com/inference-sim/inference-sim/sim"
)

// `metrics --json` prints the registry exactly: the same identifiers in the same order, and for
// each the same definition field for field, scopes and aggregates included. The command holds
// no definition of its own (DEV-8, X7), so the comparison is with metrics.All and with a
// decoding of the output that shares no code with the printer.
func TestMetricsCommand_PrintsTheRegistryExactly(t *testing.T) {
	// The cobra command itself, with its flag, in a child process: a miswired RunE, a lost
	// flag or a printer given part of the registry fails here, not only a bad printer.
	stdout, stderr, err := runRawCLI(t, "metrics", "--json")
	if err != nil {
		t.Fatalf("blis metrics --json: %v\n%s", err, stderr)
	}
	var got struct {
		Metrics []metrics.Definition `json:"metrics"`
	}
	dec := json.NewDecoder(strings.NewReader(stdout))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&got); err != nil {
		t.Fatalf("the output is not {\"metrics\": [...]} of definitions: %v", err)
	}
	want := metrics.All()
	if len(want) < 30 {
		t.Fatalf("non-vacuity: the registry holds %d metrics", len(want))
	}
	if len(got.Metrics) != len(want) {
		t.Fatalf("printed %d metrics, the registry holds %d", len(got.Metrics), len(want))
	}
	for i, w := range want {
		g := got.Metrics[i]
		if g.ID != w.ID {
			t.Fatalf("metric %d is %q, want %q: not in the registry's order", i, g.ID, w.ID)
		}
		if err := metrics.Comparable(g, w); err != nil {
			t.Errorf("%s: printed definition differs: %v", w.ID, err)
		}
		if g.Better != w.Better {
			t.Errorf("%s: printed direction %q, registry %q", w.ID, g.Better, w.Better)
		}
		if !sameNames(g.Scopes, w.Scopes) || !sameNames(g.Aggregates, w.Aggregates) {
			t.Errorf("%s: printed scopes %v aggregates %v, registry scopes %v aggregates %v",
				w.ID, g.Scopes, g.Aggregates, w.Scopes, w.Aggregates)
		}
		if len(g.Scopes) == 0 {
			t.Errorf("%s: the output states no scope", w.ID)
		}
	}
}

func sameNames[T ~string](a, b []T) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// A command that drops a metric, or one of its scopes or aggregates, is found: the comparison
// above is not satisfied by a printer that edits the registry (X18).
func TestMetricsCommand_AnOmittedOrAlteredEntryIsFound(t *testing.T) {
	all := metrics.All()
	differs := func(printed []metrics.Definition) bool {
		if len(printed) != len(all) {
			return true
		}
		for i, w := range all {
			if printed[i].ID != w.ID || !sameNames(printed[i].Scopes, w.Scopes) || !sameNames(printed[i].Aggregates, w.Aggregates) {
				return true
			}
		}
		return false
	}
	if differs(append([]metrics.Definition(nil), all...)) {
		t.Fatal("the registry differs from itself")
	}
	if !differs(all[1:]) {
		t.Error("an omitted metric was not found")
	}
	narrowed := append([]metrics.Definition(nil), all...)
	narrowed[0].Scopes = narrowed[0].Scopes[:1]
	if !differs(narrowed) {
		t.Error("a dropped scope was not found")
	}
	widened := append([]metrics.Definition(nil), all...)
	widened[0].Aggregates = append(append([]metrics.Aggregate(nil), widened[0].Aggregates...), metrics.AggregateMin)
	if !differs(widened) {
		t.Error("an added aggregate was not found")
	}
}

// The text form has one line per metric and states every metric's scopes and aggregates.
func TestMetricsCommand_TextNamesEachMetricWithItsScopeAndAggregate(t *testing.T) {
	stdout, stderr, err := runRawCLI(t, "metrics")
	if err != nil {
		t.Fatalf("blis metrics: %v\n%s", err, stderr)
	}
	lines := strings.Split(strings.TrimSuffix(stdout, "\n"), "\n")
	if len(lines) != len(metrics.All()) {
		t.Fatalf("%d lines for %d metrics", len(lines), len(metrics.All()))
	}
	for i, d := range metrics.All() {
		if !strings.HasPrefix(lines[i], d.ID+" ") || !strings.Contains(lines[i], "better="+string(d.Better)) || !strings.Contains(lines[i], "scopes="+joinNames(d.Scopes)) ||
			!strings.Contains(lines[i], "aggregates="+joinNames(d.Aggregates)) {
			t.Errorf("line %d is %q, want %s with its scopes and aggregates", i, lines[i], d.ID)
		}
	}
}

// The Result names only metrics of the registry, and each figure it states is at a scope and
// aggregate the registry admits for it (DEV-8). A Result figure carries no scope field today:
// the document's `metrics` map is the cluster's measured-window figures, so every figure is
// validated as a cluster-scope figure computed directly (no aggregate). A metric the registry
// defines at no cluster scope would fail here.
func TestResultFiguresNameOnlyRegistryMetricsAtAdmittedScopes(t *testing.T) {
	spec := capSpecBody(t, "version: \"2\"\nseed: 7\naggregate_rate: 4\nnum_requests: 12\nclients:\n"+
		"  - id: c\n    rate_fraction: 1\n    arrival: {process: poisson}\n"+
		"    input_distribution: {type: constant, params: {value: 300}}\n"+
		"    output_distribution: {type: constant, params: {value: 40}}\n")
	o := capRun(t, runArgs(colocated(2), "--workload-spec", spec)...)
	doc := resultDoc(t, o.stdout)
	figs, ok := doc["metrics"].(map[string]any)
	if !ok || len(figs) < 10 {
		t.Fatalf("non-vacuity: the Result states %d metrics", len(figs))
	}
	for _, problem := range resultFigureProblems(figs) {
		t.Error(problem)
	}
}

// resultFigureProblems names every figure of a Result's metrics map that is not a registry
// metric valid at the cluster scope, computed directly.
func resultFigureProblems(figs map[string]any) []string {
	var ids []string
	for id := range figs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	var problems []string
	for _, id := range ids {
		def, known := metrics.Lookup(id)
		if !known {
			problems = append(problems, fmt.Sprintf("the Result reports %q, which the registry does not define", id))
			continue
		}
		v, _ := figs[id].(map[string]any)["value"].(float64)
		f := metrics.Figure{Definition: def, Value: v, Scope: metrics.ScopeCluster}
		if err := f.Validate(); err != nil {
			problems = append(problems, fmt.Sprintf("%s: %v", id, err))
		}
	}
	return problems
}

// Known-bad input for that check (X18): a metric the registry lacks is found, so the check above
// cannot pass vacuously (every registry metric is defined at the cluster scope). The substrate tests its own Figure.Validate; this tests the use made of it here.
func TestResultFigureCheck_FindsAnUnknownMetric(t *testing.T) {
	good := map[string]any{"e2e_mean_ms": map[string]any{"value": 1.0}}
	if p := resultFigureProblems(good); len(p) != 0 {
		t.Fatalf("a registry metric at the cluster scope was refused: %v", p)
	}
	if p := resultFigureProblems(map[string]any{"no_such_metric": map[string]any{"value": 1.0}}); len(p) != 1 ||
		!strings.Contains(p[0], "does not define") {
		t.Errorf("an unknown metric: problems %v", p)
	}
}

// The throughput forms InferenceX plots are stated in the Result when, and only when, the pinned
// registry defines them (blis-substrate#52). Their values are the window's token rates over the
// run's GPUs: total tokens over all GPUs, and for a disaggregated run output tokens over the
// decode pool's GPUs.
func TestThroughputForms_ValuesFollowTheDefinitions(t *testing.T) {
	sum := sim.MeasuredSummary{InputThroughputTokS: 900, OutputThroughputTokS: 100}
	all := func(string) bool { return true }
	none := func(string) bool { return false }
	if got := throughputForms(sum, 8, 0, all); len(got) != 1 || got[totalTokensPerGPU] != 125 {
		t.Errorf("a colocated run: %v, want only %s = (900 + 100) / 8 = 125", got, totalTokensPerGPU)
	}
	got := throughputForms(sum, 8, 2, all)
	if got[totalTokensPerGPU] != 125 || got[outputTokensPerDecodeGPU] != 50 {
		t.Errorf("a disaggregated run: %v, want %s = 125 and %s = 100 / 2 = 50", got, totalTokensPerGPU, outputTokensPerDecodeGPU)
	}
	if got := throughputForms(sum, 8, 2, none); len(got) != 0 {
		t.Errorf("a registry that defines neither form: the Result states %v", got)
	}
	if got := throughputForms(sum, 0, 0, all); len(got) != 0 {
		t.Errorf("a run with no GPUs states %v", got)
	}
}

// A run's Result states each throughput form the registry defines, and the figures are the
// run's own: total tokens per second over all GPUs from the Result's own window, and the decode
// form only for a disaggregated run. At a pin whose registry defines neither, the test checks
// that the Result states neither.
func TestResult_StatesTheRegisteredThroughputForms(t *testing.T) {
	_, haveTotal := metrics.Lookup(totalTokensPerGPU)
	_, haveDecode := metrics.Lookup(outputTokensPerDecodeGPU)
	pd := runArgs(append([]string{"--scenarios", pdScenarios, "--scenario", "glm-5-h200-3p1d-ib.yaml",
		"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1", "--seed", "11",
		"--horizon", "900000000"}, burst(12, 8, 1500, 64)...))
	stdout, stderr, err := runKernelCLI(t, pd...)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
	}
	figs, _ := resultDoc(t, stdout)["metrics"].(map[string]any)
	value := func(id string) (float64, bool) {
		f, ok := figs[id].(map[string]any)
		v, _ := f["value"].(float64)
		return v, ok
	}
	outPerGPU, ok := value("output_tok_s_per_gpu")
	if !ok || outPerGPU <= 0 {
		t.Fatalf("non-vacuity: the run states no output_tok_s_per_gpu: %v", figs["output_tok_s_per_gpu"])
	}
	// glm-5 is TP8 per pool: 3 prefill + 1 decode instances occupy 32 GPUs, the decode pool 8.
	total, gotTotal := value(totalTokensPerGPU)
	if gotTotal != haveTotal {
		t.Errorf("registry defines %s: %v; the Result states it: %v", totalTokensPerGPU, haveTotal, gotTotal)
	}
	if haveTotal && total <= outPerGPU {
		t.Errorf("%s = %v is not above the output-only rate %v although the prompts are 1,500 tokens", totalTokensPerGPU, total, outPerGPU)
	}
	dec, gotDec := value(outputTokensPerDecodeGPU)
	if gotDec != haveDecode {
		t.Errorf("registry defines %s: %v; the Result states it: %v", outputTokensPerDecodeGPU, haveDecode, gotDec)
	}
	if haveDecode && (dec < outPerGPU*3.99 || dec > outPerGPU*4.01) {
		t.Errorf("%s = %v, want about 4 x output_tok_s_per_gpu = %v (32 GPUs over the decode pool's 8)", outputTokensPerDecodeGPU, dec, outPerGPU*4)
	}
	// A colocated run has no decode pool and states no decode form.
	co := runArgs(append([]string{"--scenarios", pdScenarios, "--scenario", "llama-3.1-70b-instruct-h200-tp4-4node.yaml", "--num-instances", "1", "--seed", "3"}, burst(10, 8, 512, 32)...))
	out, stderr, err := runKernelCLI(t, co...)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
	}
	if _, ok := resultDoc(t, out)["metrics"].(map[string]any)[outputTokensPerDecodeGPU]; ok {
		t.Errorf("a colocated run states %s", outputTokensPerDecodeGPU)
	}
}
