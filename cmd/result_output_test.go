package cmd

// Behavioral contracts of the Result document (#1871, blis-schemas#41 sections A and D):
//
//	BC-R1 stdout of run and replay is exactly one Result that blis-schemas validates: JSON by
//	      default, YAML with --format yaml, the same document in both.
//	BC-R2 --output writes the same bytes stdout carries.
//	BC-R3 INV-6: the same input at the same revisions yields a byte-identical Result outside
//	      runtime.
//	BC-R4 INV-13: a run that exports a trace and its replay with the same scenario and run
//	      controls yield equal metrics, requests and breakdowns.
//	BC-R5 tpot_*_ms is the per-request time per output token (schema 1's itl_*), and
//	      step_interval_*_ms the engine's per-step interval; no itl_* is reported (the registry's
//	      itl pools every token gap, which BLIS does not compute).
//	BC-R6 requests counts each generated request's fate (DEV-9): the fates sum to generated, and
//	      generated is the cluster's own count, not a sum of fates.
//	BC-R7 --view summary renders the text sections to stderr and leaves stdout unchanged.
//	BC-R8 input records the resolved scenario, deployment and the --policy-config bundle
//	      verbatim; frame records the run controls; engine, revisions and data what the run read.
//	BC-R9 --metrics-path is refused; the Result carries what the file held.
//	BC-R10 a document blis-schemas refuses is fatal, naming the problem.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/metrics"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// resultRunArgs is a small colocated run on the pinned fixture scenario.
func resultRunArgs(extra ...string) []string {
	return append([]string{"run", "--scenario", kernelTestScenario, "--seed", "7", "--num-requests", "30",
		"--rate", "10", "--horizon", "600000000"}, extra...)
}

// BC-R1, BC-R2: stdout is one valid Result in either encoding, and --output holds its bytes.
func TestResult_StdoutIsOneValidResultAndOutputIsItsBytes(t *testing.T) {
	for _, tc := range []struct {
		format string
		first  byte
	}{
		{"", '{'},
		{"json", '{'},
		{"yaml", 'k'}, // "kind: Result"
	} {
		t.Run("format="+tc.format, func(t *testing.T) {
			file := filepath.Join(t.TempDir(), "result.out")
			args := resultRunArgs("--output", file)
			if tc.format != "" {
				args = append(args, "--format", tc.format)
			}
			stdout, stderr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
			}
			if stdout == "" || stdout[0] != tc.first {
				t.Fatalf("stdout does not begin as a %q Result document:\n%s", tc.format, lastLines(stdout, 5))
			}
			var r result.Result
			if err := result.Unmarshal([]byte(stdout), &r); err != nil {
				t.Fatalf("stdout is not one Result document (strict decode): %v", err)
			}
			if err := validateResult(&r); err != nil {
				t.Fatalf("stdout does not validate: %v", err)
			}
			if r.Kind != result.Kind || r.Requests.Completed != 30 {
				t.Fatalf("non-vacuity: kind %q, completed %d of 30", r.Kind, r.Requests.Completed)
			}
			got, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, []byte(stdout)) {
				t.Errorf("--output is not byte-identical to stdout")
			}
		})
	}
}

// BC-R1: the JSON and YAML encodings carry the same document.
func TestResult_JSONAndYAMLAreOneDocument(t *testing.T) {
	j, stderr, err := runKernelCLI(t, resultRunArgs()...)
	if err != nil {
		t.Fatalf("json: %v\n%s", err, lastLines(stderr, 5))
	}
	y, stderr, err := runKernelCLI(t, resultRunArgs("--format", "yaml")...)
	if err != nil {
		t.Fatalf("yaml: %v\n%s", err, lastLines(stderr, 5))
	}
	if withoutRuntime(t, j) != withoutRuntime(t, y) {
		t.Errorf("the YAML document differs from the JSON one outside runtime\n%s", firstDiff(withoutRuntime(t, j), withoutRuntime(t, y)))
	}
}

// BC-R3 (INV-6): two runs of one input are byte-identical outside runtime, and the runtime
// block is the only difference allowed.
func TestResult_DeterministicOutsideRuntime(t *testing.T) {
	a, _, err := runKernelCLI(t, resultRunArgs()...)
	if err != nil {
		t.Fatal(err)
	}
	b, _, err := runKernelCLI(t, resultRunArgs()...)
	if err != nil {
		t.Fatal(err)
	}
	if withoutRuntime(t, a) != withoutRuntime(t, b) {
		t.Errorf("INV-6: two runs of one input differ outside runtime\n%s", firstDiff(withoutRuntime(t, a), withoutRuntime(t, b)))
	}
	if parseResult(t, a).Runtime == nil {
		t.Error("non-vacuity: the document has no runtime block to exempt")
	}
}

// BC-R4 (INV-13): a run exported as a trace and replayed with the same scenario and controls
// yields equal metrics, requests and breakdowns.
func TestResult_RunAndReplayAgreeOnMetricsRequestsAndBreakdowns(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "trace")
	runOut, stderr, err := runKernelCLI(t, resultRunArgs("--trace-output", prefix)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	repOut, stderr, err := runKernelCLI(t, "replay", "--scenario", kernelTestScenario, "--seed", "7",
		"--horizon", "600000000", "--trace-header", prefix+".yaml", "--trace-data", prefix+".csv")
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 5))
	}
	run, rep := parseResult(t, runOut), parseResult(t, repOut)
	if run.Requests.Completed != 30 {
		t.Fatalf("non-vacuity: the run completed %d of 30", run.Requests.Completed)
	}
	// The three sections INV-13 names, and every other section the simulation computed.
	if !jsonEqual(t, run.Metrics, rep.Metrics) || !jsonEqual(t, run.Requests, rep.Requests) ||
		!jsonEqual(t, run.Breakdowns, rep.Breakdowns) {
		t.Errorf("INV-13: run and replay differ\nmetrics %v\n        %v\nrequests %+v\n         %+v",
			run.Metrics, rep.Metrics, run.Requests, rep.Requests)
	}
	if a, b := inv13View(t, runOut), inv13View(t, repOut); a != b {
		t.Errorf("INV-13: run and replay differ in a simulated section\n%s", firstDiff(a, b))
	}
	// The run references the trace it exported, by digest.
	if a, ok := run.Artifacts["trace_data"]; !ok || len(a.SHA256) != 64 || a.Rows == nil || *a.Rows != 30 {
		t.Errorf("the run's artifacts do not reference its exported trace: %+v", run.Artifacts)
	}
	// The replay records the trace it read as its workload.
	if w := rep.Frame.Generator.Workload; w == nil || w.Trace == nil || w.Trace.Rows != 30 {
		t.Errorf("the replay's frame.generator.workload does not reference its trace: %+v", w)
	}
}

func jsonEqual(t *testing.T, a, b any) bool {
	t.Helper()
	x, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	y, err := json.Marshal(b)
	if err != nil {
		t.Fatal(err)
	}
	return bytes.Equal(x, y)
}

// BC-R5: tpot_*_ms is the per-request time per output token; the per-step interval is
// step_interval_*_ms. Under speculative decoding they differ by the accepted length, which is
// what makes the distinction observable. The Result's tpot_mean_ms is also the mean of its rows'
// tpot_us, a figure computed a second way.
func TestResult_TPOTIsPerTokenAndTheStepIntervalIsPerStep(t *testing.T) {
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", pdScenarios, "--scenario", dsparkScenario,
		"--num-requests", "20", "--rate", "0.5", "--seed", "42", "--speculative-acceptance-rate", "0.502")
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	r := parseResult(t, stdout)
	for _, stat := range []string{"mean", "p50", "p90", "p99"} {
		name := "tpot_" + stat + "_ms"
		if d, ok := metrics.Lookup(name); !ok || d.Formula != metrics.FormulaTPOT {
			t.Errorf("%s is defined %+v, want the registry's per-request tpot", name, d)
		}
		if _, ok := r.Metrics["itl_"+stat+"_ms"]; ok {
			t.Errorf("itl_%s_ms is reported: the registry's itl pools token gaps, which BLIS does not compute", stat)
		}
	}
	var sum float64
	var n int
	for _, row := range r.RequestRows {
		if row.TPOTUs != nil {
			sum += *row.TPOTUs / 1e3
			n++
		}
	}
	if want := r.Metrics["tpot_mean_ms"].Value; n == 0 || math.Abs(sum/float64(n)-want) > 1e-6*want {
		t.Errorf("tpot_mean_ms %v, want the mean %v of the %d rows' tpot_us", want, sum/float64(max(n, 1)), n)
	}
	step := metric(t, stdout, "step_interval_p90_ms")
	// Non-vacuity: with drafts accepted, a step emits more than one token, so the two differ.
	if tpot := r.Metrics["tpot_p90_ms"].Value; step <= tpot {
		t.Errorf("under speculation the step interval %v must exceed the per-token TPOT %v", step, tpot)
	}
}

// BC-R6: requests counts each generated request's fate, the drops by reason, and generated is
// the cluster's own count: under a bounded flow-control queue some arrivals are rejected, and the
// fates still sum to a count they do not define.
func TestResult_RequestsCountEveryFateAndGenerated(t *testing.T) {
	// TestINV13_RunReplayParity_LLMDRouterInputs's router: flow control bounded at 8 requests.
	dir := colocatedRouterVariant(t, "router:\n  epp_flags: [--pool-name=vllm, --refresh-metrics-interval=20ms]\n"+
		"  endpoint_picker_config: epp-config.yaml\n  manifests: [pool.yaml]\n", "      app: vllm\n",
		map[string]string{"epp-config.yaml": writeRouterFile(t, "epp-config.yaml", inv13Config),
			"pool.yaml": writeRouterFile(t, "pool.yaml", inv13Manifests)})
	const n = 1500
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "2",
		"--rate", "2000", "--num-requests", "1500", "--seed", "7", "--horizon", "600000000",
		"--gateway-latency", "300", "--epp-scrape-jitter", "5000", noRenderQueue[0], noRenderQueue[1])
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	q := parseResult(t, stdout).Requests
	if q.Generated != n {
		t.Errorf("generated = %d, want the %d the workload produced", q.Generated, n)
	}
	if q.Dropped[result.DropGatewayQueueRejected] == 0 {
		t.Fatalf("non-vacuity: the bounded queue rejected nothing; the gateway drop is not exercised: %+v", q)
	}
	if q.Sum() != q.Generated {
		t.Errorf("DEV-9 does not close: %+v", q)
	}
	doc := resultDoc(t, stdout)
	for _, k := range []string{"generated", "completed", "timed_out", "in_flight_at_horizon", "queued_at_horizon",
		"not_started", "length_capped", "dropped.gateway_queue_rejected"} {
		if _, ok := resultValue(doc, "requests."+k); !ok {
			t.Errorf("requests has no %s", k)
		}
	}
}

// BC-R7: --view summary renders the text sections to stderr and leaves stdout as it was.
func TestResult_ViewSummaryRendersToStderrOnly(t *testing.T) {
	plain, _, err := runKernelCLI(t, resultRunArgs("--slo-ttft", "default=1ms")...)
	if err != nil {
		t.Fatal(err)
	}
	viewed, stderr, err := runKernelCLI(t, resultRunArgs("--slo-ttft", "default=1ms", "--view", "summary")...)
	if err != nil {
		t.Fatal(err)
	}
	if withoutRuntime(t, plain) != withoutRuntime(t, viewed) {
		t.Error("--view summary changed stdout")
	}
	for _, section := range []string{"=== Result ", "=== Per-SLO Metrics ==="} {
		if !strings.Contains(stderr, section) {
			t.Errorf("--view summary did not render %q to stderr:\n%s", section, lastLines(stderr, 20))
		}
	}
	if strings.Contains(plain, "=== ") {
		t.Error("a text section reached stdout")
	}
}

// BC-R8: input records the resolved scenario and deployment, the run controls, and a
// --policy-config bundle verbatim.
func TestResult_InputRecordsWhatTheRunWasGiven(t *testing.T) {
	bundle := filepath.Join(t.TempDir(), "policy.yaml")
	if err := os.WriteFile(bundle, []byte("scheduler: fcfs\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := runKernelCLI(t, resultRunArgs("--policy-config", bundle)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	r := parseResult(t, stdout)
	in := r.Input
	if in.Scenario.Name == "" || len(in.Deployment.Pools) == 0 || in.Scenario.Model == "" {
		t.Errorf("input does not record the scenario and deployment: %+v", in)
	}
	g, stop := r.Frame.Generator, r.Frame.Stop
	if g.Load.RequestRate != 10 || stop.Requests != 30 || g.Seed != 7 || stop.HorizonS == nil || *stop.HorizonS != 600 {
		t.Errorf("frame generator %+v, stop %+v; want rate 10, 30 requests, seed 7, horizon 600 s", g, stop)
	}
	if in.Policy["scheduler"] != "fcfs" {
		t.Errorf("input.policy = %v, want the bundle verbatim", in.Policy)
	}
	if r.Revisions.Simulator == "" || r.Revisions.Substrate == "" || r.Engine.Kind != "vllm" ||
		r.Engine.Version != in.Scenario.EngineRelease() {
		t.Errorf("revisions %+v, engine %+v", r.Revisions, r.Engine)
	}
	if r.Data.Catalog.Hash == "" || r.Data.Registry.Hash == "" || r.Data.Build == "" {
		t.Errorf("data record %+v, want both sources with their hashes and the build", r.Data)
	}
	if len(r.SelectedEntries.Coefficients) == 0 {
		t.Errorf("selected_entries names no coefficient, though the kernel priced the run")
	}
}

// BC-R9: the legacy output files are gone. blis run and blis replay refuse
// --metrics-path, and the Result carries the rows and aggregates those files held.
func TestResult_MetricsPathIsRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "metrics.json")
	_, stderr, err := runKernelCLI(t, resultRunArgs("--metrics-path", path)...)
	if err == nil || !strings.Contains(stderr, "unknown flag: --metrics-path") {
		t.Errorf("--metrics-path was not refused: %v\n%s", err, lastLines(stderr, 3))
	}
	if _, statErr := os.Stat(path); statErr == nil {
		t.Error("a --metrics-path file was written")
	}
}

// BC-R10: a document blis-schemas refuses is not printed; the problem is named.
func TestResult_AnInvalidDocumentIsRefused(t *testing.T) {
	r := &result.Result{Kind: result.Kind, SchemaVersion: result.SchemaVersion, Name: "x"}
	_, err := encodeResult(r, result.FormatJSON)
	if err == nil || !strings.Contains(err.Error(), "does not validate") || !strings.Contains(err.Error(), "metrics") {
		t.Errorf("an empty document was accepted or the refusal names nothing: %v", err)
	}
}

// --format and --view refuse a value they do not know, before the run.
func TestResult_UnknownFormatOrViewIsRefused(t *testing.T) {
	for _, extra := range [][]string{{"--format", "xml"}, {"--view", "table"}} {
		_, stderr, err := runKernelCLI(t, resultRunArgs(extra...)...)
		if err == nil || !strings.Contains(stderr, extra[0]) {
			t.Errorf("%v was not refused naming the flag: %v\n%s", extra, err, lastLines(stderr, 3))
		}
	}
}

// A run whose horizon ends before any request arrives is a valid run that measured nothing:
// it prints a Result with zero rates over the horizon and the requests it was asked for, not a
// validation failure.
func TestResult_ARunWithNoArrivalsBeforeTheHorizonStillEmits(t *testing.T) {
	stdout, stderr, err := runKernelCLI(t, "run", "--scenario", kernelTestScenario, "--seed", "7",
		"--num-requests", "50", "--rate", "1", "--horizon", "10")
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	r := parseResult(t, stdout)
	if r.Requests.Generated != 0 || r.Frame.Stop.Requests != 50 {
		t.Errorf("generated %d, frame.stop.requests %d; want 0 and the 50 asked for", r.Requests.Generated, r.Frame.Stop.Requests)
	}
	if r.Window.DurationS != 10e-6 || r.Metrics["request_throughput"].Value != 0 {
		t.Errorf("window %+v, request_throughput %v; want the 10 us horizon and 0", r.Window, r.Metrics["request_throughput"])
	}
}

// --view summary renders the sections in the order run and replay printed them: on a P/D run
// with SLO targets, the per-SLO section precedes the P/D section, and the anomaly and KV
// sections, when present, precede both.
func TestResult_ViewSummaryKeepsTheSectionOrder(t *testing.T) {
	_, stderr, err := runKernelCLI(t, append([]string{"run", "--view", "summary", "--slo-ttft", "default=1ms"}, pdBurst...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	last := -1
	for _, section := range []string{"=== Result ", "=== Per-SLO Metrics ===", "=== PD Metrics ==="} {
		at := strings.Index(stderr, section)
		if at < 0 {
			t.Fatalf("--view summary rendered no %q:\n%s", section, lastLines(stderr, 30))
		}
		if at < last {
			t.Errorf("%q is out of order", section)
		}
		last = at
	}
	if !strings.Contains(stderr, "Disaggregated Requests: 40") {
		t.Errorf("the P/D section does not report the 40 disaggregated requests:\n%s", lastLines(stderr, 20))
	}
}

// A run in which nothing arrives and no --horizon bounds simulated time has no span to state
// a rate over; it is refused with a message naming the cause, not a validator dump.
func TestResult_ARunThatSimulatedNoTimeIsRefused(t *testing.T) {
	spec := filepath.Join(t.TempDir(), "spec.yaml")
	if err := os.WriteFile(spec, []byte(`version: "2"
seed: 1
aggregate_rate: 1
num_requests: 5
clients:
  - id: c
    rate_fraction: 1
    slo_class: standard
    arrival: {process: poisson}
    input_distribution: {type: constant, params: {value: 64}}
    output_distribution: {type: constant, params: {value: 8}}
    lifecycle:
      windows: [{start_us: 0, end_us: 10}]
`), 0o644); err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := runKernelCLI(t, "run", "--scenario", kernelTestScenario, "--workload-spec", spec)
	if err == nil {
		t.Fatalf("a run that simulated no time printed:\n%s", stdout)
	}
	if !strings.Contains(stderr, "simulated no time") || strings.Contains(stderr, "does not validate") {
		t.Errorf("not refused with the cause named:\n%s", lastLines(stderr, 3))
	}
	if stdout != "" {
		t.Errorf("a refused run printed to stdout:\n%s", stdout)
	}
}

// A component's integer report survives into the document exactly: toGeneric keeps an integer
// above 2^53 an integer rather than rounding it through float64.
func TestResult_ComponentReportsKeepIntegersExact(t *testing.T) {
	const big = int64(9007199254740993) // 2^53 + 1
	g := toGeneric(struct {
		N int64   `json:"n"`
		F float64 `json:"f"`
	}{big, 0.5}).(map[string]any)
	if g["n"] != big || g["f"] != 0.5 {
		t.Errorf("toGeneric = %#v, want n=%d (int64) and f=0.5", g, big)
	}
}

// With no --output an absolute artifact path is recorded by its base name, so the document does
// not carry the host's directory layout (INV-6); a relative one as given, cleaned; with
// --output, relative to its directory.
func TestResult_ArtifactRefsCarryNoHostLayout(t *testing.T) {
	defer func(o string) { resultOutput = o }(resultOutput)
	for _, tc := range []struct{ output, path, want string }{
		{"", "/some/host/dir/trace.csv", "trace.csv"},
		{"", "./sub/../sub/trace.csv", "sub/trace.csv"},
		{"/out/r.json", "/out/run/trace.csv", "run/trace.csv"},
	} {
		resultOutput = tc.output
		if got := artifactRef(tc.path); got != tc.want {
			t.Errorf("output %q path %q: ref %q, want %q", tc.output, tc.path, got, tc.want)
		}
	}
}

// --output is written before stdout: a run that cannot write it fails with stdout empty, so a
// consumer of stdout never sees a complete document from a failed run.
func TestResult_AnUnwritableOutputLeavesStdoutEmpty(t *testing.T) {
	stdout, stderr, err := runKernelCLI(t, resultRunArgs("--output", filepath.Join(t.TempDir(), "missing", "dir", "r.json"))...)
	if err == nil || !strings.Contains(stderr, "--output") {
		t.Fatalf("an unwritable --output did not fail naming it: %v\n%s", err, lastLines(stderr, 3))
	}
	if stdout != "" {
		t.Errorf("stdout carried a document from a failed run:\n%s", lastLines(stdout, 5))
	}
}

// #1615: a replay without --horizon is unbounded, as the run it replays was, so its document
// states no horizon and equals the run's (INV-13); an explicit horizon is recorded as given.
func TestResult_ADefaultReplayStatesNoHorizonAndAnExplicitOneIsRecorded(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "trace")
	runOut, stderr, err := runKernelCLI(t, "run", "--scenario", kernelTestScenario, "--seed", "7",
		"--num-requests", "30", "--rate", "10", "--trace-output", prefix)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	replay := func(extra ...string) *result.Result {
		out, stderr, err := runKernelCLI(t, append([]string{"replay", "--scenario", kernelTestScenario, "--seed", "7",
			"--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv"}, extra...)...)
		if err != nil {
			t.Fatalf("replay %v: %v\n%s", extra, err, lastLines(stderr, 5))
		}
		return parseResult(t, out)
	}
	def := replay()
	if h := def.Frame.Stop.HorizonS; h != nil {
		t.Fatalf("a default replay records the horizon %v, want none", *h)
	}
	if got := replay("--horizon", "600000000").Frame.Stop.HorizonS; got == nil || *got != 600 {
		t.Errorf("an explicit horizon is recorded as %v, want 600 s", got)
	}
	// The run had no horizon; the replay given the same (unbounded) horizon reproduces it.
	run := parseResult(t, runOut)
	if run.Frame.Stop.HorizonS != nil {
		t.Fatalf("non-vacuity: the run states a horizon %v", *run.Frame.Stop.HorizonS)
	}
	same := replay("--horizon", "9223372036854775807")
	if !jsonEqual(t, run.Metrics, same.Metrics) || !jsonEqual(t, run.Requests, same.Requests) {
		t.Errorf("INV-13: the replay at the run's horizon differs from the run")
	}
}

// #1871 review S1: generated is the cluster's own arrival count, checked against a count the
// ledger does not define: the exported trace's rows (one per arrival, written by the arrival
// hook). The run is cut by its horizon while requests are still in the host's render pipeline,
// so a ledger that restated its bucket sum, or lost those requests, would disagree.
func TestResult_GeneratedIsTheArrivalCountOnAHorizonCutRun(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "trace")
	stdout, stderr, err := runKernelCLI(t, "run", "--scenario", kernelTestScenario, "--seed", "7",
		"--num-requests", "200", "--rate", "400", "--horizon", "300000", "--trace-output", prefix)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	q := parseResult(t, stdout).Requests
	trace, err := workload.LoadTraceV2(prefix+".yaml", prefix+".csv")
	if err != nil {
		t.Fatal(err)
	}
	if q.Generated != len(trace.Records) {
		t.Errorf("generated = %d, want the %d arrivals the trace recorded", q.Generated, len(trace.Records))
	}
	if q.Unfinished() == 0 || q.Generated >= 200 {
		t.Fatalf("non-vacuity: the horizon did not cut the run with requests waiting: %+v", q)
	}
	if q.Sum() != q.Generated {
		t.Errorf("DEV-9: generated %d, fates %d: %+v", q.Generated, q.Sum(), q)
	}
}

// resultLedger states generated from the fate ledger's count of requests created, never from
// the fates: on a run whose ledger lost a fate, generated still states every request, so the
// contracts' validator sees the gap instead of a restated sum.
func TestResult_TheLedgerDoesNotRestateItsFateSum(t *testing.T) {
	reqs := makeSharedPrefixRequests(4)
	cs := newRoutedCluster(newKernelDeployment(t, 1, nil).Config, reqs)
	if err := cs.Run(); err != nil {
		t.Fatal(err)
	}
	q := resultLedger(cs)
	if q.Generated != len(reqs) || q.Sum() != len(reqs) {
		t.Errorf("generated %d, fates %d, want the %d requests created", q.Generated, q.Sum(), len(reqs))
	}
	q.Completed-- // a fate lost
	r := &result.Result{Requests: q}
	if p := r.Validate(); !strings.Contains(fmt.Sprint(p.All()), "fates sum to") {
		t.Errorf("a ledger whose fates do not sum to generated validates: %v", p.All())
	}
}

// #1871 review S3: the emit path validates before it writes: an invalid document reaches
// neither stdout nor --output, and the error names the problem.
func TestResult_TheEmitPathRefusesAnInvalidDocument(t *testing.T) {
	defer func(o string) { resultOutput = o }(resultOutput)
	resultOutput = filepath.Join(t.TempDir(), "r.json")
	var stdout bytes.Buffer
	err := writeResult(&stdout, &result.Result{Kind: result.Kind, SchemaVersion: result.SchemaVersion, Name: "x"})
	if err == nil || !strings.Contains(err.Error(), "does not validate") {
		t.Fatalf("an invalid document was written: %v", err)
	}
	if stdout.Len() != 0 {
		t.Errorf("stdout carried %d bytes of an invalid document", stdout.Len())
	}
	if _, err := os.Stat(resultOutput); !os.IsNotExist(err) {
		t.Errorf("--output was written for an invalid document (%v)", err)
	}
}

// #1871 review S5: --fitness-weights and --summarize-trace are reported in the document, not
// only by --view summary.
func TestResult_FitnessAndTraceSummaryAreInTheDocument(t *testing.T) {
	out, stderr, err := runKernelCLI(t, resultRunArgs("--fitness-weights", "p99_ttft:3,mean_e2e:1,throughput:2",
		"--trace-level", "decisions", "--summarize-trace")...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	if s := resultNum(t, out, "router.fitness.score"); s <= 0 || s > 6 {
		t.Errorf("router.fitness.score = %v, want a weighted score in (0, 6]", s)
	}
	if n := resultNum(t, out, "router.trace_summary.TotalDecisions"); n <= 0 {
		t.Errorf("router.trace_summary.TotalDecisions = %v, want the run's routing decisions", n)
	}
	if !strings.Contains(stderr, "--view summary") {
		t.Errorf("no hint that --view summary prints them as text:\n%s", lastLines(stderr, 5))
	}
}

// #1871 review S6, S8, S9: a replay's input names its trace without the host's directory; a
// workload-spec run records --seed and the spec's own seed; a closed-loop cap the follow-up
// rounds exceeded is kept as an override.
func TestResult_InputRecordsTheTraceTheSeedsAndTheCap(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "sub", "trace")
	if err := os.MkdirAll(filepath.Dir(prefix), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, stderr, err := runKernelCLI(t, resultRunArgs("--trace-output", prefix)...); err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
	}
	rep, stderr, err := runKernelCLI(t, "replay", "--scenario", kernelTestScenario, "--seed", "7",
		"--trace-header", prefix+".yaml", "--trace-data", prefix+".csv")
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 3))
	}
	if d := parseResult(t, rep).Frame.Generator.Workload.Trace.Data; d != "trace.csv" {
		t.Errorf("frame.generator.workload.trace.data = %q, want the file name only", d)
	}

	spec := filepath.Join(t.TempDir(), "spec.yaml")
	if err := os.WriteFile(spec, []byte(`version: "2"
seed: 99
category: reasoning
aggregate_rate: 2
num_requests: 6
clients:
  - id: chat
    rate_fraction: 1
    slo_class: standard
    closed_loop: true
    arrival: {process: poisson}
    input_distribution: {type: constant, params: {value: 64}}
    output_distribution: {type: constant, params: {value: 8}}
    reasoning:
      reason_ratio_distribution: {type: constant, params: {value: 0}}
      multi_turn: {max_rounds: 3, think_time_us: 500000, context_growth: accumulate}
`), 0o644); err != nil {
		t.Fatal(err)
	}
	out, stderr, err := runKernelCLI(t, "run", "--scenario", kernelTestScenario, "--workload-spec", spec)
	if err != nil {
		t.Fatalf("spec run: %v\n%s", err, lastLines(stderr, 3))
	}
	r := parseResult(t, out)
	rows := map[string]string{}
	for _, o := range r.Overrides {
		rows[o.Setting] = o.Run
	}
	if r.Frame.Generator.Seed != 99 || rows["workload_seed"] != "99" {
		t.Errorf("frame seed %d, workload_seed %q; want the spec's 99, which its draws used", r.Frame.Generator.Seed, rows["workload_seed"])
	}
	if !strings.HasPrefix(rows["workload-spec"], "sha256:") {
		t.Errorf("the spec is not pinned by sha256: %v", rows)
	}
	// #1905: follow-up rounds count against the cap, so the run generates exactly the 6 it was
	// given and needs no override row for exceeding it.
	if r.Requests.Generated != 6 || rows["num_requests_cap"] != "" {
		t.Errorf("generated %d, cap row %q; want the cap of 6 held, with no override", r.Requests.Generated, rows["num_requests_cap"])
	}
}

// #1615 (INV-13): a replay given no --horizon completes what the run it replays completed.
// The old default, twice the last arrival, cut the tail of the run off at a high rate.
func TestResult_ReplayWithoutHorizonReproducesTheRun(t *testing.T) {
	for _, rate := range []string{"20", "60"} {
		prefix := filepath.Join(t.TempDir(), "trace")
		runOut, stderr, err := runKernelCLI(t, "run", "--scenario", kernelTestScenario, "--seed", "7",
			"--num-requests", "60", "--rate", rate, "--trace-output", prefix)
		if err != nil {
			t.Fatalf("rate %s run: %v\n%s", rate, err, lastLines(stderr, 5))
		}
		repOut, stderr, err := runKernelCLI(t, "replay", "--scenario", kernelTestScenario, "--seed", "7",
			"--trace-header", prefix+".yaml", "--trace-data", prefix+".csv")
		if err != nil {
			t.Fatalf("rate %s replay: %v\n%s", rate, err, lastLines(stderr, 5))
		}
		run, rep := parseResult(t, runOut), parseResult(t, repOut)
		if run.Requests.Completed != 60 {
			t.Fatalf("non-vacuity: rate %s run completed %d of 60", rate, run.Requests.Completed)
		}
		if !jsonEqual(t, run.Requests, rep.Requests) || !jsonEqual(t, run.Metrics, rep.Metrics) {
			t.Errorf("rate %s: replay differs from run\nrequests %+v\n         %+v", rate, run.Requests, rep.Requests)
		}
	}
}

// #1914 (INV-13): a closed-loop accumulate session workload exported by run replays to the
// same figures, cross-round prefix reuse included; the header says the rounds are deltas.
func TestResult_AccumulateSessionsRoundTripThroughTheTrace(t *testing.T) {
	dir := t.TempDir()
	spec := filepath.Join(dir, "ss.yaml")
	if err := os.WriteFile(spec, []byte(`version: "2"
seed: 7
category: reasoning
aggregate_rate: 5.0
num_requests: 60
cohorts:
  - id: agents
    population: 10
    rate_fraction: 1.0
    arrival: {process: poisson}
    input_distribution: {type: gaussian, params: {mean: 256, std_dev: 64, min: 64, max: 1024}}
    output_distribution: {type: exponential, params: {mean: 64}}
    reasoning:
      reason_ratio_distribution: {type: constant, params: {value: 0}}
      multi_turn: {max_rounds: 4, think_time_us: 500000, context_growth: accumulate, single_session: true}
    closed_loop: true
`), 0o644); err != nil {
		t.Fatal(err)
	}
	prefix := filepath.Join(dir, "tr")
	runOut, stderr, err := runKernelCLI(t, "run", "--scenario", kernelTestScenario, "--seed", "7",
		"--workload-spec", spec, "--trace-output", prefix)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	hdr, err := os.ReadFile(prefix + ".yaml")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(hdr), "session_context_growth: accumulate") {
		t.Fatalf("the exported header does not say the rounds are accumulate deltas:\n%s", hdr)
	}
	repOut, stderr, err := runKernelCLI(t, "replay", "--scenario", kernelTestScenario, "--seed", "7",
		"--session-mode", "closed-loop", "--trace-header", prefix+".yaml", "--trace-data", prefix+".csv")
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 5))
	}
	run, rep := parseResult(t, runOut), parseResult(t, repOut)
	if run.Requests.Completed < 40 {
		t.Fatalf("non-vacuity: the run completed %d requests", run.Requests.Completed)
	}
	if !jsonEqual(t, run.Requests, rep.Requests) || !jsonEqual(t, run.Metrics, rep.Metrics) {
		t.Errorf("INV-13: replay differs from run\nrequests %+v\n         %+v\nmetrics %v\n        %v", run.Requests, rep.Requests, run.Metrics, rep.Metrics)
	}
}

// #2071 (DEV-9): the Result carries one row per generated request, each with its fate, and
// the rows agree with the figures: the completed rows' time to first token has the mean
// ttft_mean_ms reports, and a request cut off or dropped states no latency.
func TestResult_RequestRowsAreOnePerGeneratedRequest(t *testing.T) {
	out, stderr, err := runKernelCLI(t, "run", "--scenario", kernelTestScenario, "--seed", "7",
		"--num-requests", "60", "--rate", "400", "--horizon", "700000")
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	r := parseResult(t, out)
	if len(r.RequestRows) != r.Requests.Generated || r.Requests.Generated == 0 {
		t.Fatalf("%d rows for %d generated requests", len(r.RequestRows), r.Requests.Generated)
	}
	if r.Requests.Unfinished() == 0 || r.Requests.Completed == 0 {
		t.Fatalf("non-vacuity: the horizon should cut the run mid-way: %+v", r.Requests)
	}
	var sum float64
	var n int
	for _, row := range r.RequestRows {
		if row.Fate != result.FateCompleted {
			if row.E2EUs != nil || row.CompletionIndex != 0 {
				t.Errorf("%s (%s) states a request latency", row.ID, row.Fate)
			}
			continue
		}
		sum += float64(*row.TTFTUs) / 1e3
		n++
	}
	if want := r.Metrics["ttft_mean_ms"].Value; n != r.Requests.Completed || math.Abs(sum/float64(n)-want) > 1e-6*want {
		t.Errorf("rows' mean ttft %v over %d completed, the metric %v over %d", sum/float64(n), n, want, r.Requests.Completed)
	}
}

// SIM-10, DEV-7 (#2054): a Result reproduced from its own embedded input is byte-identical to
// the original outside runtime, except the two documented members, rerun_of and
// revision_changes. Checked on a run and on the replay of its trace, on a real simulation.
func TestResult_FromResultReproducesTheDocument(t *testing.T) {
	dir := t.TempDir()
	prefix := filepath.Join(dir, "trace")
	origRun, stderr, err := runKernelCLI(t, resultRunArgs("--trace-output", prefix)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	origRunFile := filepath.Join(dir, "run.json")
	if err := os.WriteFile(origRunFile, []byte(origRun), 0o644); err != nil {
		t.Fatal(err)
	}
	origRep, stderr, err := runKernelCLI(t, "replay", "--scenario", kernelTestScenario, "--seed", "7",
		"--trace-header", prefix+".yaml", "--trace-data", prefix+".csv")
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 5))
	}
	origRepFile := filepath.Join(dir, "replay.json")
	if err := os.WriteFile(origRepFile, []byte(origRep), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		command, orig, file string
		extra               []string
	}{
		{"run", origRun, origRunFile, nil},
		{"replay", origRep, origRepFile, []string{"--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv"}},
	} {
		t.Run(tc.command, func(t *testing.T) {
			again, stderr, err := runKernelCLI(t, append([]string{tc.command, "--from-result", tc.file}, tc.extra...)...)
			if err != nil {
				t.Fatalf("%s --from-result: %v\n%s", tc.command, err, lastLines(stderr, 8))
			}
			want, got := parseResult(t, tc.orig), parseResult(t, again)
			if got.RerunOf == nil || got.RerunOf.Name != want.Name || len(got.RerunOf.SHA256) != 64 {
				t.Fatalf("rerun_of = %+v, want the original %q by digest", got.RerunOf, want.Name)
			}
			if len(got.RevisionChanges) != 0 {
				t.Errorf("revision_changes = %+v, want none (same build and data)", got.RevisionChanges)
			}
			got.RerunOf, got.RevisionChanges, got.Runtime, want.Runtime = nil, nil, nil, nil
			a, _ := result.Marshal(want, result.FormatJSON)
			b, _ := result.Marshal(got, result.FormatJSON)
			if string(a) != string(b) {
				t.Errorf("SIM-10: the reproduced Result differs from the original\n%s", firstDiff(string(a), string(b)))
			}
			if len(want.RequestRows) == 0 || len(want.Input.Options) == 0 {
				t.Errorf("non-vacuity: the original records %d rows and %d options", len(want.RequestRows), len(want.Input.Options))
			}
		})
	}
}

// A reproduction refuses a file option that is not the file the Result was produced with, and
// a Result of the other command.
func TestResult_FromResultRefusesAChangedFileAndTheOtherCommand(t *testing.T) {
	dir := t.TempDir()
	spec := filepath.Join(dir, "spec.yaml")
	body := "version: \"2\"\nseed: 3\ncategory: language\naggregate_rate: 5\nnum_requests: 10\nclients:\n  - id: c\n    rate_fraction: 1\n    arrival: {process: poisson}\n    input_distribution: {type: constant, params: {value: 64}}\n    output_distribution: {type: constant, params: {value: 8}}\n"
	if err := os.WriteFile(spec, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	out, stderr, err := runKernelCLI(t, "run", "--scenario", kernelTestScenario, "--workload-spec", spec)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	file := filepath.Join(dir, "r.json")
	if err := os.WriteFile(file, []byte(out), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, stderr, err := runKernelCLI(t, "run", "--from-result", file); err == nil || !strings.Contains(stderr, "pass the file again") {
		t.Errorf("a missing file option was not refused: %v\n%s", err, lastLines(stderr, 3))
	}
	other := filepath.Join(dir, "other.yaml")
	if err := os.WriteFile(other, []byte(strings.Replace(body, "seed: 3", "seed: 4", 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, stderr, err := runKernelCLI(t, "run", "--from-result", file, "--workload-spec", other); err == nil || !strings.Contains(stderr, "is not the file the Result was produced with") {
		t.Errorf("a changed file was not refused: %v\n%s", err, lastLines(stderr, 3))
	}
	if _, stderr, err := runKernelCLI(t, "run", "--from-result", file, "--workload-spec", spec); err != nil {
		t.Errorf("the same file was refused: %v\n%s", err, lastLines(stderr, 3))
	}
	if _, stderr, err := runKernelCLI(t, "replay", "--from-result", file, "--workload-spec", spec); err == nil || !strings.Contains(stderr, "came from `blis run`") {
		t.Errorf("a run Result was accepted by replay: %v\n%s", err, lastLines(stderr, 3))
	}
}

// A reproduction on a different build lists each component whose revision changed (#2054): the
// original is edited to claim another simulator and registry, and the reproduction names both.
func TestResult_FromResultListsTheRevisionsThatChanged(t *testing.T) {
	dir := t.TempDir()
	out, stderr, err := runKernelCLI(t, resultRunArgs()...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	r := parseResult(t, out)
	current := r.Revisions.Simulator
	r.Revisions.Simulator = "v0.0.1-older"
	r.Data.Registry.Hash, r.Data.Registry.CombinedHash = strings.Repeat("a", 64), strings.Repeat("a", 64)
	data, err := result.Marshal(r, result.FormatJSON)
	if err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "old.json")
	if err := os.WriteFile(file, data, 0o644); err != nil {
		t.Fatal(err)
	}
	again, stderr, err := runKernelCLI(t, "run", "--from-result", file)
	if err != nil {
		t.Fatalf("--from-result: %v\n%s", err, lastLines(stderr, 5))
	}
	got := map[string]result.RevisionChange{}
	for _, c := range parseResult(t, again).RevisionChanges {
		got[c.Component] = c
	}
	if c := got["simulator"]; c.From != "v0.0.1-older" || c.To != current {
		t.Errorf("simulator change = %+v, want v0.0.1-older -> %s", c, current)
	}
	if c := got["registry"]; c.From != strings.Repeat("a", 64) || c.To == "" || c.To == c.From {
		t.Errorf("registry change = %+v", c)
	}
	if len(got) != 2 {
		t.Errorf("revision_changes = %+v, want exactly simulator and registry", got)
	}
}

// DEV-7 (#2054): the reproduction also holds for a disaggregated deployment, and for a
// data-parallel one at width 2.
func TestResult_FromResultReproducesDisaggregatedAndDataParallelRuns(t *testing.T) {
	for _, tc := range []struct {
		name, dir, scenario string
		extra               []string
		pools               int
	}{
		{"p/d", pdScenarios, "glm-5-h200-3p1d-ib.yaml",
			[]string{"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1"}, 2},
		{"dp2", "", dpScenario, []string{"--num-instances", "2"}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			args := append([]string{"run", "--scenario", tc.scenario, "--seed", "5", "--num-requests", "20", "--rate", "2"}, tc.extra...)
			if tc.dir != "" {
				args = append(args, "--scenarios", tc.dir)
			}
			orig, stderr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
			}
			file := filepath.Join(t.TempDir(), "r.json")
			if err := os.WriteFile(file, []byte(orig), 0o644); err != nil {
				t.Fatal(err)
			}
			again := []string{"run", "--from-result", file}
			if tc.dir != "" {
				again = append(again, "--scenarios", tc.dir)
			}
			out, stderr, err := runKernelCLI(t, again...)
			if err != nil {
				t.Fatalf("--from-result: %v\n%s", err, lastLines(stderr, 8))
			}
			want, got := parseResult(t, orig), parseResult(t, out)
			if want.Requests.Completed == 0 || len(want.Input.Deployment.Pools) != tc.pools || got.RerunOf == nil {
				t.Fatalf("non-vacuity: completed %d, pools %d, rerun_of %v", want.Requests.Completed, len(want.Input.Deployment.Pools), got.RerunOf)
			}
			got.RerunOf, got.RevisionChanges, got.Runtime, want.Runtime = nil, nil, nil, nil
			a, _ := result.Marshal(want, result.FormatJSON)
			b, _ := result.Marshal(got, result.FormatJSON)
			if string(a) != string(b) {
				t.Errorf("the reproduced Result differs\n%s", firstDiff(string(a), string(b)))
			}
		})
	}
}

// The CONTRIBUTING walkthrough's test: a row's slo_class is its client's class (it is the
// walkthrough's own example, so the tutorial's code stays run by the suite).
func TestResult_RowSLOClassIsItsClientsClass(t *testing.T) {
	spec := filepath.Join(t.TempDir(), "spec.yaml")
	body := `version: "2"
seed: 5
category: language
aggregate_rate: 10
num_requests: 20
clients:
  - {id: a, slo_class: critical, rate_fraction: 0.5, arrival: {process: poisson},
     input_distribution: {type: constant, params: {value: 64}},
     output_distribution: {type: constant, params: {value: 8}}}
  - {id: b, slo_class: batch, rate_fraction: 0.5, arrival: {process: poisson},
     input_distribution: {type: constant, params: {value: 64}},
     output_distribution: {type: constant, params: {value: 8}}}
`
	if err := os.WriteFile(spec, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	out, stderr, err := runKernelCLI(t, "run", "--scenario", kernelTestScenario, "--workload-spec", spec)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	seen := map[string]int{}
	for _, row := range parseResult(t, out).RequestRows {
		seen[row.SLOClass]++
	}
	if seen["critical"] == 0 || seen["batch"] == 0 || len(seen) != 2 {
		t.Errorf("rows by slo_class: %v, want both classes and no other", seen)
	}
}

// The Result's per-request rows state the figures the measured-window aggregate averages
// (DEV-9, #2071): a harness that scores from the rows computes the aggregate the Result reports.
// Recomputed here from the rows for a colocated run and for a disaggregated one, where a request's
// time per output token is its own latency over its tokens and not the decode leg's step interval.
func TestResult_RowsRecomputeTheAggregatesTheyComeFrom(t *testing.T) {
	cases := map[string][]string{
		"colocated": {"run", "--scenario", kernelTestScenario, "--seed", "7", "--num-requests", "80", "--rate", "50"},
		"disaggregated": append([]string{"run"}, append(append([]string{}, pdTopology...),
			"--num-requests", "40", "--rate", "5")...),
	}
	for name, args := range cases {
		t.Run(name, func(t *testing.T) {
			out, stderr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
			}
			r := parseResult(t, out)
			var e2e, tpot float64
			var nE2E, nTPOT int
			for _, row := range r.RequestRows {
				if row.Fate != result.FateCompleted {
					continue
				}
				if row.E2EUs != nil {
					e2e += float64(*row.E2EUs) / 1e3
					nE2E++
				}
				if row.TPOTUs != nil {
					tpot += *row.TPOTUs / 1e3
					nTPOT++
				}
			}
			if nE2E < 20 || nTPOT < 20 {
				t.Fatalf("non-vacuity: %d rows state E2E and %d state TPOT", nE2E, nTPOT)
			}
			for _, c := range []struct {
				id        string
				got       float64
				n         int
				tolerance float64
			}{
				// Rows state integer microseconds, so the means agree to well under a microsecond.
				{"e2e_mean_ms", e2e / float64(nE2E), nE2E, 1e-3},
				{"tpot_mean_ms", tpot / float64(nTPOT), nTPOT, 1e-3},
			} {
				want := r.Metrics[c.id].Value
				if math.Abs(c.got-want) > c.tolerance {
					t.Errorf("%s: the mean of the %d rows is %.6f, the aggregate %.6f", c.id, c.n, c.got, want)
				}
			}
		})
	}
}

// An unfinished request states what it produced (the substrate's rule: only a request that emitted
// no token has no TTFT). A request cut off by its client deadline or by the horizon after emitting
// tokens states its output_tokens and its TTFT, and still no E2E or TPOT, which are samples only
// of a completed request. A request that emitted nothing states none of the three.
func TestResult_UnfinishedRowsStateTheTokensTheyProduced(t *testing.T) {
	common := []string{"run", "--scenario", kernelTestScenario, "--seed", "7", "--num-requests", "30",
		"--rate", "300", "--output-tokens", "400", "--output-tokens-stdev", "0", "--output-tokens-min", "400",
		"--output-tokens-max", "400"}
	cases := map[string][]string{
		"timed out":      append(append([]string{}, common...), "--timeout", "1"),
		"cut by horizon": append(append([]string{}, common...), "--horizon", "600000"),
	}
	for name, args := range cases {
		t.Run(name, func(t *testing.T) {
			out, stderr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
			}
			r := parseResult(t, out)
			var withTokens, without int
			for _, row := range r.RequestRows {
				if row.Fate == result.FateCompleted {
					continue
				}
				if row.E2EUs != nil || row.TPOTUs != nil {
					t.Errorf("%s (%s) states a completed-only sample", row.ID, row.Fate)
				}
				if row.OutputTokens > 0 {
					withTokens++
					if row.TTFTUs == nil {
						t.Errorf("%s (%s) emitted %d tokens and states no time to first token", row.ID, row.Fate, row.OutputTokens)
					}
				} else {
					without++
					if row.TTFTUs != nil {
						t.Errorf("%s (%s) emitted no token and states a time to first token", row.ID, row.Fate)
					}
				}
			}
			if withTokens == 0 {
				t.Fatalf("non-vacuity: no unfinished request had emitted a token (%d without)", without)
			}
		})
	}
}
