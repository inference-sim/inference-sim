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
//	BC-R5 itl_*_ms is per token (today's request_itl_*), never the per-decode-step interval,
//	      which is engine step_interval_*.
//	BC-R6 requests carries every INV-1 term plus rejected and generated, observed independently
//	      (generated is the cluster's arrival count, not a sum of buckets).
//	BC-R7 --view summary renders the text sections to stderr and leaves stdout unchanged.
//	BC-R8 input records the resolved scenario, deployment and run controls, and the
//	      --policy-config bundle verbatim.
//	BC-R9 --metrics-path keeps its content and logs a deprecation pointing at --output.
//	BC-R10 a document blis-schemas refuses is fatal, naming the problem.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/measurement"
	"github.com/inference-sim/blis-schemas/spec/result"

	sim "github.com/inference-sim/inference-sim/sim"
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
	if !jsonEqual(t, run.Metrics, rep.Metrics) || !jsonEqual(t, run.Requests, rep.Requests) ||
		!jsonEqual(t, run.Breakdowns, rep.Breakdowns) {
		t.Errorf("INV-13: run and replay differ\nmetrics %v\n        %v\nrequests %+v\n         %+v",
			run.Metrics, rep.Metrics, run.Requests, rep.Requests)
	}
	// The run references the trace it exported, by digest.
	if a, ok := run.Artifacts["trace_data"]; !ok || len(a.SHA256) != 64 || a.Rows == nil || *a.Rows != 30 {
		t.Errorf("the run's artifacts do not reference its exported trace: %+v", run.Artifacts)
	}
	// The replay records the trace it read as its workload.
	if w := rep.Input.Run.Workload; w == nil || w.Trace == nil || w.Trace.Rows != 30 {
		t.Errorf("the replay's input.run.workload does not reference its trace: %+v", w)
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

// BC-R5: itl_*_ms is the per-token request ITL; the per-step interval the legacy itl_* keys
// held is the engine's step_interval_*. Under speculative decoding they differ by the accepted
// length, which is what makes the distinction observable.
func TestResult_ITLIsPerTokenAndTheStepIntervalIsTheEngines(t *testing.T) {
	path := filepath.Join(t.TempDir(), "metrics.json")
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", pdScenarios, "--scenario", dsparkScenario,
		"--num-requests", "20", "--rate", "0.5", "--seed", "42", "--speculative-acceptance-rate", "0.502",
		"--metrics-path", path)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	var file map[string]any
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatal(err)
	}
	r := parseResult(t, stdout)
	for _, stat := range []string{"mean", "p50", "p90", "p99"} {
		name := "itl_" + stat + "_ms"
		if got, want := r.Metrics[name], file["request_itl_"+stat+"_ms"].(float64); got != want {
			t.Errorf("%s = %v, want the per-token request_itl_%s_ms %v", name, got, stat, want)
		}
		if d := r.MetricDefs[name]; d.Formula != measurement.FormulaTPOTExclFirst || d.Population != measurement.PopulationPerRequest {
			t.Errorf("%s is defined %+v, want tpot_excl_first per request", name, d)
		}
	}
	step := resultNum(t, stdout, "engine.cluster.step_interval_p90_ms")
	if step != file["step_interval_p90_ms"].(float64) {
		t.Errorf("engine step_interval_p90_ms %v, want the file's %v", step, file["step_interval_p90_ms"])
	}
	// Non-vacuity: with drafts accepted, a step emits more than one token, so the two differ.
	if step <= r.Metrics["itl_p90_ms"] {
		t.Errorf("under speculation the step interval %v must exceed the per-token ITL %v", step, r.Metrics["itl_p90_ms"])
	}
	for name := range r.Metrics {
		if strings.HasPrefix(name, "step_interval") {
			t.Errorf("the per-step interval %s is a metric; it is the engine's", name)
		}
	}
}

// BC-R6: the ledger carries every INV-1 term plus rejected and generated, and generated is the
// cluster's own arrival count: under a bounded flow-control queue some arrivals are rejected,
// and the clauses still hold against a count the buckets do not define.
func TestResult_RequestsLedgerIsTheElevenTermFormPlusRejectedAndGenerated(t *testing.T) {
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
	if q.GatewayQueueRejected == 0 {
		t.Fatalf("non-vacuity: the bounded queue rejected nothing; the cluster-only buckets are not exercised: %+v", q)
	}
	if q.Terminal() != q.Injected || q.Generated != q.Injected+q.Rejected {
		t.Errorf("INV-1 does not close: %+v", q)
	}
	// Every key of the eleven-term form and the pipeline clause is in the document.
	doc := resultDoc(t, stdout)
	for _, k := range []string{"generated", "rejected", "injected", "completed", "still_queued", "still_running",
		"dropped_unservable", "timed_out", "routing_rejections", "gateway_queue_depth", "gateway_queue_rejected",
		"gateway_evicted", "gateway_expired", "encode_routing_rejections", "length_capped"} {
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
	run := in.Run
	if run.Load.RequestRate != 10 || run.NumRequests != 30 || run.Seed != 7 || run.HorizonS == nil || *run.HorizonS != 600 {
		t.Errorf("input.run = %+v, want rate 10, 30 requests, seed 7, horizon 600 s", run)
	}
	if in.Policy["scheduler"] != "fcfs" {
		t.Errorf("input.policy = %v, want the bundle verbatim", in.Policy)
	}
	if r.Revisions.BLIS == "" || r.Revisions.BLISSchemas == "" || r.Revisions.Engine == nil ||
		r.Revisions.Engine.UpstreamVersion != in.Scenario.EngineVersion {
		t.Errorf("revisions = %+v", r.Revisions)
	}
}

// BC-R9: --metrics-path keeps its content (the legacy MetricsOutput, with per-request rows)
// and logs a deprecation pointing at --output; the document references the file by digest.
func TestResult_MetricsPathIsDeprecatedButKept(t *testing.T) {
	path := filepath.Join(t.TempDir(), "metrics.json")
	stdout, stderr, err := runKernelCLI(t, resultRunArgs("--metrics-path", path)...)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stderr, "--metrics-path is deprecated") || !strings.Contains(stderr, "--output") {
		t.Errorf("no deprecation naming --output on stderr:\n%s", lastLines(stderr, 10))
	}
	var file sim.MetricsOutput
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatal(err)
	}
	if file.CompletedRequests != 30 || len(file.Requests) != 30 || file.InstanceID != "cluster" {
		t.Errorf("the file lost its content: completed %d, %d rows, instance %q", file.CompletedRequests, len(file.Requests), file.InstanceID)
	}
	a, ok := parseResult(t, stdout).Artifacts["metrics_file"]
	if !ok || a.SHA256 != bytesDigest(data) {
		t.Errorf("the Result does not reference the file by its digest: %+v", a)
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
	if r.Requests.Generated != 0 || r.Input.Run.NumRequests != 50 {
		t.Errorf("generated %d, num_requests %d; want 0 and the 50 asked for", r.Requests.Generated, r.Input.Run.NumRequests)
	}
	if r.Window.DurationS != 10e-6 || r.Metrics["request_throughput"] != 0 {
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
