package cmd

// The capability examples: one scenario per M3.0 capability under examples/capabilities/, each run
// through the real CLI (the same path a reader's `blis run` takes, on the vendored catalog and
// registry) and each asserting the capability's observable effect against a control arm that has
// the capability off, so an example that is accepted but inert fails. the design
// shows each example's YAML; the examples are the files these tests run.
//
// Every run is small (tens of requests) and CPU-light. The runs are subprocesses that share no
// state, so the tests run in parallel.

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// capExamplesDir is the examples, relative to cmd/.
const capExamplesDir = "../examples/capabilities"

// capDoc is one run's Result document, as JSON.
type capDoc struct {
	t      *testing.T
	stdout string
	doc    map[string]any
}

// capExampleRun runs `blis <verb> --scenarios <dir> <args...>` and returns the Result. dir is the
// examples themselves unless a test passes a variant of them.
func capExampleRun(t *testing.T, dir string, args ...string) capDoc {
	t.Helper()
	stdout, stderr, err := runKernelCLI(t, append(args, "--scenarios", dir)...)
	if err != nil {
		t.Fatalf("blis %s: %v\n%s", strings.Join(args, " "), err, lastLines(stderr, 6))
	}
	return capDoc{t: t, stdout: stdout, doc: resultDoc(t, stdout)}
}

// capExampleFails runs a command that must be refused and returns its stderr.
func capExampleFails(t *testing.T, dir string, args ...string) string {
	t.Helper()
	_, stderr, err := runKernelCLI(t, append(args, "--scenarios", dir)...)
	if err == nil {
		t.Fatalf("blis %s: want a refusal, the run succeeded", strings.Join(args, " "))
	}
	return stderr
}

// num reads a number from the Result by its dotted path.
// promptHitRate is the share of prompt tokens the engine served from a cache: its own prefix-cache hit
// rate plus the share an offload tier served (vLLM's prompt_tokens_cached is local plus external,
// PrefillStats, vllm/v1/core/sched/scheduler.py:1251-1257@4ba787ec). Since blis-vllm#248 the engine
// counts the prompts a connector loads asynchronously in the prefix-cache queries, so the engine's
// own hit rate alone no longer includes what a tier keeps.
func (d capDoc) promptHitRate() float64 {
	d.t.Helper()
	rate := d.num("engine.cluster.cache_hit_rate")
	if v, ok := resultValue(d.doc, "engine.cluster.cpu_tier_hit_rate"); ok {
		if f, ok := v.(float64); ok {
			rate += f
		}
	}
	return rate
}

func (d capDoc) num(path string) float64 {
	d.t.Helper()
	v, ok := resultValue(d.doc, path)
	if !ok {
		d.t.Fatalf("the Result has no %s", path)
	}
	f, ok := v.(float64)
	if !ok {
		d.t.Fatalf("%s = %v, not a number", path, v)
	}
	return f
}

// rows returns the Result's request rows.
func (d capDoc) rows() []map[string]any {
	d.t.Helper()
	raw, _ := d.doc["request_rows"].([]any)
	out := make([]map[string]any, 0, len(raw))
	for _, r := range raw {
		if m, ok := r.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

// instances returns every engine instance's own figures, by instance id.
func (d capDoc) instances() map[string]map[string]any {
	d.t.Helper()
	return instanceFigures(d.t, d.stdout)
}

// completed returns the number of completed requests in the ledger.
func (d capDoc) completed() int { return int(d.num("requests.completed")) }

// field reads a number from a request row, 0 when absent.
func rowNum(r map[string]any, k string) float64 {
	f, _ := r[k].(float64)
	return f
}

func rowStr(r map[string]any, k string) string {
	s, _ := r[k].(string)
	return s
}

// exampleVariant copies the examples to a fresh directory and replaces old with new in the named
// file, once. A replacement that finds nothing fails the test: a variant must be a real edit of the
// example it controls.
func exampleVariant(t *testing.T, file, old, replacement string) string {
	t.Helper()
	out := t.TempDir()
	err := filepath.Walk(capExamplesDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(capExamplesDir, path)
		if info.IsDir() {
			return os.MkdirAll(filepath.Join(out, rel), 0o755)
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if rel == file {
			if !strings.Contains(string(raw), old) {
				t.Fatalf("%s has no %q to replace", file, old)
			}
			raw = []byte(strings.Replace(string(raw), old, replacement, 1))
		}
		return os.WriteFile(filepath.Join(out, rel), raw, 0o644)
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// withoutOffload is the cpu-tier example with its offload block removed: the control arm.
func withoutOffload(t *testing.T, file string) string {
	t.Helper()
	raw := readFile(t, filepath.Join(capExamplesDir, file))
	i := strings.Index(raw, "\noffload:")
	if i < 0 {
		t.Fatalf("%s has no offload block", file)
	}
	return exampleVariant(t, file, raw[i:], "\n")
}

// Engine selection (SIM-1): the scenario names `engine: {kind, version}` and the build runs the
// release it names. An older vLLM release, an unregistered kind and the retired engine_version
// field are each refused by name, with the release the build does run.
func TestCapabilityExample_EngineIsSelectedByKindAndVersion(t *testing.T) {
	t.Parallel()
	args := []string{"run", "--scenario", "session-load-only.yaml", "--num-instances", "4", "--rate", "20", "--num-requests", "8"}
	ok := capExampleRun(t, capExamplesDir, args...)
	if e, _ := ok.doc["engine"].(map[string]any); e["version"] != "0.33.0" || e["kind"] != "vllm" {
		t.Errorf("the Result names engine %v, want vllm 0.33.0", e)
	}
	const engine = `engine: {kind: vllm, version: "0.33.0"}`
	for _, tc := range []struct{ name, edit, want string }{
		{"older vLLM", `engine: {kind: vllm, version: "0.32.0"}`, "this build runs vllm 0.33.0"},
		{"unregistered kind", `engine: {kind: tensorrt-llm, version: "0.5.23"}`, "this build runs the kinds sglang, vllm"},
		{"retired field", engine + "\nengine_version: \"0.33.0\"", "engine_version"},
	} {
		stderr := capExampleFails(t, exampleVariant(t, "session-load-only.yaml", engine, tc.edit), args...)
		if !strings.Contains(stderr, tc.want) {
			t.Errorf("%s: want a refusal containing %q, got\n%s", tc.name, tc.want, lastLines(stderr, 3))
		}
	}
}

// KV tier, CPU: a workload of 160 distinct 2,000-token prompts does not fit the example's HBM, and
// blocks evicted to the CPU tier are found there on the prompt's next request. With the offload
// block removed the same run hits less and answers later.
func TestCapabilityExample_CPUTierKeepsEvictedPrefixes(t *testing.T) {
	t.Parallel()
	args := []string{"run", "--scenario", "cpu-tier.yaml", "--num-instances", "1", "--horizon", "25000000",
		"--workload-spec", filepath.Join(capExamplesDir, "spill-workload.yaml")}
	with := capExampleRun(t, capExamplesDir, args...)
	without := capExampleRun(t, withoutOffload(t, "cpu-tier.yaml"), args...)
	hit, hitOff := with.promptHitRate(), without.promptHitRate()
	if hit <= hitOff+0.03 {
		t.Errorf("cache hit rate %.3f with the CPU tier, %.3f without: the tier keeps nothing", hit, hitOff)
	}
	if a, b := with.num("metrics.ttft_mean_ms"), without.num("metrics.ttft_mean_ms"); a >= b {
		t.Errorf("mean TTFT %.1f ms with the CPU tier, %.1f ms without: the tier does not shorten prefill", a, b)
	}
	if with.completed() <= without.completed() {
		t.Errorf("%d requests completed by the horizon with the tier, %d without", with.completed(), without.completed())
	}
}

// Prefill/decode with pools of unequal width: four TP4 prefill engines and one TP8 decode engine.
// Every request is disaggregated and handed off once, the decode engine is priced for its own
// layout (a TP8 decode steps faster than a TP4 one on the same model), and the handoff is
// conserved.
func TestCapabilityExample_PDPoolsOfUnequalTP(t *testing.T) {
	t.Parallel()
	shape := []string{"run", "--rate", "5", "--num-requests", "20", "--seed", "3"}
	wide := capExampleRun(t, capExamplesDir, append(shape, "--scenario", "pd-unequal-tp.yaml",
		"--num-instances", "5", "--prefill-instances", "4", "--decode-instances", "1")...)
	if wide.completed() != 20 || wide.num("breakdowns.pd.disaggregated_requests") != 20 || wide.num("breakdowns.pd.kv_transfers") != 20 {
		t.Fatalf("completed %d, disaggregated %v, handoffs %v: want 20 of each", wide.completed(),
			wide.num("breakdowns.pd.disaggregated_requests"), wide.num("breakdowns.pd.kv_transfers"))
	}
	// The control arm is the same deployment with the decode pool at TP4 (two decode engines).
	raw := readFile(t, filepath.Join(capExamplesDir, "pd-unequal-tp.yaml"))
	i := strings.Index(raw, "role: decode")
	narrow := capExampleRun(t, exampleVariant(t, "pd-unequal-tp.yaml", raw[i:], strings.Replace(raw[i:], "tp: 8", "tp: 4", 1)),
		append(shape, "--scenario", "pd-unequal-tp.yaml", "--num-instances", "6", "--prefill-instances", "4", "--decode-instances", "2")...)
	itl := func(d capDoc, id string) float64 { return d.instances()[id]["itl_mean_ms"].(float64) }
	if w, n := itl(wide, "instance_4"), itl(narrow, "instance_4"); !(w > 0 && w < 0.9*n) {
		t.Errorf("the TP8 decode engine steps at ITL %.2f ms, the TP4 one at %.2f ms: the decode pool is not priced for its own layout", w, n)
	}
}

// Per-pool hardware (SIM-17): a cluster of an H200 pool and an H100 pool. The first two instances
// are placed in the H200 pool and step faster than the last two, which are in the H100 pool, and the
// smallest KV budget in the cluster is the H100's.
func TestCapabilityExample_NodePoolsOfTwoChips(t *testing.T) {
	t.Parallel()
	d := capExampleRun(t, capExamplesDir, "run", "--scenario", "node-pools.yaml", "--num-instances", "4",
		"--policy-config", filepath.Join(capExamplesDir, "node-pools-bundle.yaml"), "--rate", "4", "--num-requests", "40")
	if d.completed() != 40 {
		t.Fatalf("%d of 40 requests completed", d.completed())
	}
	itl := func(id string) float64 { return d.instances()[id]["itl_mean_ms"].(float64) }
	fast, older := (itl("instance_0")+itl("instance_1"))/2, (itl("instance_2")+itl("instance_3"))/2
	if fast >= 0.95*older {
		t.Errorf("the H200 pool steps at ITL %.2f ms, the H100 pool at %.2f ms: instances are not priced by their pool's chip", fast, older)
	}
	// The same deployment on H200 alone holds more KV per replica than the mixed cluster's smallest.
	h200 := exampleVariant(t, "node-pools.yaml", "hardware: h100", "hardware: h200")
	alone := capExampleRun(t, h200, "run", "--scenario", "node-pools.yaml", "--num-instances", "4",
		"--policy-config", writeH200Pools(t), "--rate", "4", "--num-requests", "40")
	if m, a := d.num("engine.cluster.kv_capacity_tokens_per_replica"), alone.num("engine.cluster.kv_capacity_tokens_per_replica"); m >= a {
		t.Errorf("the mixed cluster's smallest KV budget is %.0f tokens, an all-H200 cluster's %.0f: the H100 pool is not sized for its chip", m, a)
	}
}

// writeH200Pools is the node-pools bundle with both pools H200.
func writeH200Pools(t *testing.T) string {
	t.Helper()
	raw := readFile(t, filepath.Join(capExamplesDir, "node-pools-bundle.yaml"))
	raw = strings.Replace(raw, "gpu_type: h100", "gpu_type: h200", 1)
	raw = strings.Replace(raw, "gpu_memory_gib: 79.1895752", "gpu_memory_gib: 139.80072021", 1)
	path := filepath.Join(t.TempDir(), "bundle.yaml")
	if err := os.WriteFile(path, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// sessionSpread returns, for the sessions with more than one completed round, the mean number of
// distinct instances that served a session.
func sessionSpread(d capDoc) (mean float64, sessions int) {
	by := map[string][]string{}
	for _, r := range d.rows() {
		if rowStr(r, "fate") == "completed" {
			by[rowStr(r, "session_id")] = append(by[rowStr(r, "session_id")], rowStr(r, "instance"))
		}
	}
	total := 0
	for _, inst := range by {
		if len(inst) < 2 {
			continue
		}
		sessions++
		seen := map[string]bool{}
		for _, i := range inst {
			seen[i] = true
		}
		total += len(seen)
	}
	if sessions == 0 {
		return 0, 0
	}
	return float64(total) / float64(sessions), sessions
}

// Session affinity through the hosted router: with the session-affinity scorer every round of a
// session goes to the endpoint that served its first round, so the engine's prefix cache holds the
// earlier rounds; routed by queue depth alone the rounds of a session spread over the endpoints.
func TestCapabilityExample_SessionAffinityPinsASessionToItsEndpoint(t *testing.T) {
	t.Parallel()
	args := func(scenario string) []string {
		return []string{"run", "--scenario", scenario, "--num-instances", "4",
			"--workload-spec", filepath.Join(capExamplesDir, "multiturn-sessions.yaml")}
	}
	pinned := capExampleRun(t, capExamplesDir, args("session-affinity.yaml")...)
	spread := capExampleRun(t, capExamplesDir, args("session-load-only.yaml")...)
	pm, ps := sessionSpread(pinned)
	sm, ss := sessionSpread(spread)
	if ps < 50 || ss < 50 {
		t.Fatalf("non-vacuity: %d and %d multi-round sessions, want at least 50 each", ps, ss)
	}
	if pm != 1 {
		t.Errorf("with session affinity a session was served by %.2f instances on average, want exactly 1", pm)
	}
	if sm < 1.3 {
		t.Errorf("without it a session was served by %.2f instances on average: the control does not spread sessions", sm)
	}
	if a, b := pinned.num("engine.cluster.cache_hit_rate"), spread.num("engine.cluster.cache_hit_rate"); a < b+0.1 {
		t.Errorf("cache hit rate %.3f with session affinity, %.3f without: pinning keeps no context", a, b)
	}
}

// Drain-gated phases: the measured phase starts when every warm-up request has its fate, not at a
// clock time, and the measured window holds the measured requests alone.
func TestCapabilityExample_PhasesStartWhenThePreviousPhaseHasDrained(t *testing.T) {
	t.Parallel()
	d := capExampleRun(t, capExamplesDir, "run", "--scenario", "cpu-tier.yaml", "--num-instances", "1",
		"--workload-spec", filepath.Join(capExamplesDir, "phases-workload.yaml"))
	var warmDone, measuredStart float64
	measuredStart = math.Inf(1)
	counts := map[string]int{}
	for _, r := range d.rows() {
		ph := rowStr(r, "phase")
		counts[ph]++
		arrival, e2e := rowNum(r, "arrival_us"), rowNum(r, "e2e_us")
		switch ph {
		case "warmup":
			warmDone = math.Max(warmDone, arrival+e2e)
		case "measured":
			measuredStart = math.Min(measuredStart, arrival)
		}
	}
	if counts["warmup"] != 12 || counts["measured"] != 24 {
		t.Fatalf("rows per phase %v, want 12 warmup and 24 measured", counts)
	}
	if measuredStart < warmDone {
		t.Errorf("the first measured request arrived at %.0f us, before the last warm-up request ended at %.0f us", measuredStart, warmDone)
	}
	if got := d.num("window.requests"); got != 24 {
		t.Errorf("the measured window holds %v requests, want the 24 measured ones", got)
	}
	// The cache the warm-up left serves the measured phase: its requests find the 512-token prefix.
	hit := 0
	for _, r := range d.rows() {
		if rowStr(r, "phase") == "measured" && rowNum(r, "computed_tokens") < rowNum(r, "prompt_tokens") {
			hit++
		}
	}
	if hit < 20 {
		t.Errorf("%d of 24 measured requests found cached prefix tokens, want nearly all: the phases did not share the cache", hit)
	}
}

// Constant-load replay (aiperf without --scenario): the session slots stay full, a warm-up of the
// same load runs first and is outside the measured window, and sub-agent requests take no slot.
func TestCapabilityExample_ConstantLoadReplay(t *testing.T) {
	d := capExampleRun(t, "../examples", "replay", "--scenario", "agentx-dsv41-flash-h100-2x-tp8-dspark5.yaml",
		"--agentic-scenario", "aiperf-constant-load", "--speculative-acceptance-rate", "0.5",
		"--corpus", "../sim/workload/testdata/agentx/weka_v7_slice.jsonl",
		"--workload-spec", filepath.Join(capExamplesDir, "constant-load-workload.yaml"),
		"--concurrency", "3", "--benchmark-duration", "60s")
	if got := d.num("engine.cluster.agentic.lanes"); got != 3 {
		t.Errorf("session slots = %v, want 3", got)
	}
	if got := d.num("engine.cluster.agentic.profile_start_us"); got < 10_000_000 {
		t.Errorf("the measured phase starts at %.0f us, want after the 10 s warm-up", got)
	}
	counts := map[string]int{}
	for _, r := range d.rows() {
		counts[rowStr(r, "phase")]++
	}
	if counts["warmup"] == 0 || counts["profile"] == 0 {
		t.Errorf("rows per phase %v: want warm-up and measured rows", counts)
	}
	if sub := d.num("engine.cluster.agentic.subagent_requests"); sub <= 0 {
		t.Error("the replay sent no sub-agent request")
	}
	if got, want := d.num("window.requests"), float64(counts["profile"]); got != want {
		t.Errorf("the measured window holds %v requests, the measured rows number %v", got, want)
	}
}

// Pooled session replay: `blis replay --concurrent-sessions K` holds K sessions live at a time
// over the recorded sessions. The replay runs exactly --total-sessions sessions, never more than K
// live at once, and fills the pool.
func TestCapabilityExample_PooledSessionReplayHoldsTheStatedConcurrency(t *testing.T) {
	t.Parallel()
	prefix := filepath.Join(t.TempDir(), "sessions")
	capExampleRun(t, capExamplesDir, "run", "--scenario", "session-load-only.yaml", "--num-instances", "4",
		"--workload-spec", filepath.Join(capExamplesDir, "multiturn-sessions.yaml"), "--trace-output", prefix)
	const pool, total = 8, 40
	d := capExampleRun(t, capExamplesDir, "replay", "--scenario", "session-load-only.yaml", "--num-instances", "4",
		"--trace-header", prefix+".yaml", "--trace-data", prefix+".csv",
		"--concurrent-sessions", "8", "--total-sessions", "40")
	type span struct{ start, end float64 }
	spans := map[string]*span{}
	for _, r := range d.rows() {
		s := rowStr(r, "session_id")
		a, e := rowNum(r, "arrival_us"), rowNum(r, "arrival_us")+rowNum(r, "e2e_us")
		if spans[s] == nil {
			spans[s] = &span{a, e}
		}
		spans[s].start, spans[s].end = math.Min(spans[s].start, a), math.Max(spans[s].end, e)
	}
	if len(spans) != total {
		t.Fatalf("the replay ran %d sessions, want %d", len(spans), total)
	}
	peak := 0
	for _, s := range spans {
		live := 0
		for _, o := range spans {
			if o.start <= s.start && s.start < o.end {
				live++
			}
		}
		peak = max(peak, live)
	}
	if peak > pool {
		t.Errorf("%d sessions were live at once, the pool holds %d", peak, pool)
	}
	if peak < pool {
		t.Errorf("at most %d sessions were live at once, the pool should fill to %d", peak, pool)
	}
}

// Open-loop accumulate replay: an open-loop accumulate session exports as per-round deltas and
// `--session-mode fixed-accumulate` replays it row for row: the same arrivals, prompts, cached
// tokens and outputs.
func TestCapabilityExample_OpenLoopAccumulateSessionsReplayRowForRow(t *testing.T) {
	t.Parallel()
	prefix := filepath.Join(t.TempDir(), "open-loop")
	run := capExampleRun(t, capExamplesDir, "run", "--scenario", "session-load-only.yaml", "--num-instances", "4",
		"--workload-spec", filepath.Join(capExamplesDir, "open-loop-sessions.yaml"), "--trace-output", prefix)
	if !strings.Contains(readFile(t, prefix+".yaml"), "session_context_growth: accumulate") {
		t.Fatal("the exported header does not state session_context_growth: accumulate")
	}
	rep := capExampleRun(t, capExamplesDir, "replay", "--scenario", "session-load-only.yaml", "--num-instances", "4",
		"--trace-header", prefix+".yaml", "--trace-data", prefix+".csv", "--session-mode", "fixed-accumulate")
	key := func(d capDoc) []map[string]any {
		rows := d.rows()
		slices.SortFunc(rows, func(a, b map[string]any) int {
			if c := rowNum(a, "arrival_us") - rowNum(b, "arrival_us"); c != 0 {
				if c < 0 {
					return -1
				}
				return 1
			}
			return strings.Compare(rowStr(a, "id"), rowStr(b, "id"))
		})
		return rows
	}
	a, b := key(run), key(rep)
	if len(a) != 160 || len(b) != 160 {
		t.Fatalf("%d rows in the run and %d in the replay, want 160 each", len(a), len(b))
	}
	grew := 0
	for i := range a {
		for _, f := range []string{"arrival_us", "prompt_tokens", "computed_tokens", "output_tokens"} {
			if rowNum(a[i], f) != rowNum(b[i], f) {
				t.Fatalf("row %d: %s is %v in the run and %v in the replay", i, f, a[i][f], b[i][f])
			}
		}
		if rowNum(a[i], "computed_tokens") < rowNum(a[i], "prompt_tokens") {
			grew++
		}
	}
	if grew == 0 {
		t.Error("no round reused the context of the round before it: the sessions did not accumulate")
	}
}

// Autoscaling: the model autoscaler grows the model beyond its first replica under the first burst
// and the cluster autoscaler adds nodes for the replicas that do not fit; after the silence the model
// scales to zero, and the second burst waits for a replica to load from the NVMe tier, which a run
// that never scales to zero does not. A slower weight source loads longer.
func TestCapabilityExample_AutoscalerScalesUpAndToZeroAndLoadsFromAStorageTier(t *testing.T) {
	t.Parallel()
	bundle := filepath.Join(capExamplesDir, "autoscale-bundle.yaml")
	run := func(b string) (capDoc, float64) {
		d := capExampleRun(t, capExamplesDir, "run", "--scenario", "autoscale.yaml", "--num-instances", "1",
			"--policy-config", b, "--workload-spec", filepath.Join(capExamplesDir, "bursts-workload.yaml"),
			"--horizon", "120000000")
		var first float64 = -1
		firstAt := math.Inf(1)
		for _, r := range d.rows() {
			if a := rowNum(r, "arrival_us"); a >= 40_000_000 && a < firstAt && rowStr(r, "fate") == "completed" {
				firstAt, first = a, rowNum(r, "ttft_us")
			}
		}
		return d, first
	}
	variant := func(old, replacement string) string {
		raw := strings.Replace(readFile(t, bundle), old, replacement, 1)
		if raw == readFile(t, bundle) {
			t.Fatalf("the bundle has no %q", old)
		}
		path := filepath.Join(t.TempDir(), "bundle.yaml")
		if err := os.WriteFile(path, []byte(raw), 0o644); err != nil {
			t.Fatal(err)
		}
		return path
	}
	scaled, lateWait := run(bundle)
	if n := len(scaled.instances()); n < 4 {
		t.Errorf("the run ended with %d instances, want the autoscaler to have added several", n)
	}
	if d := scaled.completed(); d != int(scaled.num("requests.generated")) {
		t.Errorf("%d of %v requests completed", d, scaled.num("requests.generated"))
	}
	_, lateKept := run(variant("scale_to_zero_idle_us: 5000000", "scale_to_zero_idle_us: 0"))
	if lateKept > 1_000_000 {
		t.Fatalf("fixture: the second burst waited %.0f us for a replica although the model never scaled to zero", lateKept)
	}
	if lateWait < 2_000_000 {
		t.Errorf("the second burst's first request waited %.0f us after the model scaled to zero, want at least the 2 s loading delay", lateWait)
	}
	_, lateSlow := run(variant("weight_source: nvme_gen4", "weight_source: nvme_gen3"))
	if lateSlow <= lateWait+500_000 {
		t.Errorf("loading from a Gen3 NVMe drive took the burst's first request %.0f us, from NVMe %.0f us: the weight source does not set the load time", lateSlow, lateWait)
	}
}

// Every instance states how busy its engine was over the measured window: the shares are
// fractions, the prefill and decode parts add up to the busy share, and the cluster's is the
// instances' mean.
func TestCapabilityExample_InstancesStateTheirMeasuredWindow(t *testing.T) {
	t.Parallel()
	d := capExampleRun(t, capExamplesDir, "run", "--scenario", "node-pools.yaml", "--num-instances", "4",
		"--policy-config", filepath.Join(capExamplesDir, "node-pools-bundle.yaml"), "--rate", "4", "--num-requests", "40")
	var sum float64
	insts := d.instances()
	if len(insts) != 4 {
		t.Fatalf("%d instances, want 4", len(insts))
	}
	for id, in := range insts {
		mw, ok := in["measured_window"].(map[string]any)
		if !ok {
			t.Fatalf("instance %s states no measured_window", id)
		}
		busy := mw["busy_share"].(float64)
		if busy <= 0 || busy > 1 {
			t.Errorf("instance %s busy_share %v, want in (0, 1]", id, busy)
		}
		if math.Abs(mw["prefill_busy_share"].(float64)+mw["decode_busy_share"].(float64)-busy) > 1e-9 {
			t.Errorf("instance %s: prefill and decode shares do not add up to the busy share %v", id, busy)
		}
		if mw["window_end_us"].(float64) <= mw["window_start_us"].(float64) {
			t.Errorf("instance %s: an empty measured window", id)
		}
		sum += busy
	}
	if got := d.num("engine.cluster.busy_share"); math.Abs(got-sum/4) > 1e-9 {
		t.Errorf("cluster busy_share %v, want the instances' mean %v", got, sum/4)
	}
}

// KV tier, file system (fs): blocks the 8 GiB CPU tier evicts spill to the NVMe device and are
// read back on the prompt's next request. Against the same deployment without the device the run
// hits the cache more and completes more requests by the horizon, and the Result states the device
// link's concurrent copies (copies share link capacity with each other and with P/D pulls).
func TestCapabilityExample_FSTierKeepsWhatTheCPUTierEvicts(t *testing.T) {
	t.Parallel()
	args := []string{"run", "--scenario", "fs-tier.yaml", "--num-instances", "1", "--horizon", "25000000",
		"--workload-spec", filepath.Join(capExamplesDir, "spill-workload.yaml")}
	with := capExampleRun(t, capExamplesDir, args...)
	raw := readFile(t, filepath.Join(capExamplesDir, "fs-tier.yaml"))
	line := "\n    - {tier: nvme_gen4, bytes: 4398046511104}"
	without := capExampleRun(t, exampleVariant(t, "fs-tier.yaml", line, ""), args...)
	if !strings.Contains(raw, line) {
		t.Fatal("fixture: the example states no fs tier")
	}
	if a, b := with.promptHitRate(), without.promptHitRate(); a <= b+0.02 {
		t.Errorf("cache hit rate %.3f with the fs tier, %.3f without: the device keeps nothing", a, b)
	}
	if with.completed() <= without.completed() {
		t.Errorf("%d requests completed with the fs tier, %d without", with.completed(), without.completed())
	}
	pd, _ := with.doc["breakdowns"].(map[string]any)["pd"].(map[string]any)
	if _, ok := pd["link_peak_concurrent_transfers_tier-nvme_gen4-to-node-0"]; !ok {
		t.Errorf("the Result states no copy on the NVMe device's link: %v", pd)
	}
}

// figureOf finds a named number anywhere in the Result document.
func (d capDoc) figureOf(name string) (float64, bool) { return figure(d.doc, name) }

// KV tier, peer-to-peer (p2p): a P/D run over vLLM's p2p tier hands every request off through the
// prefill engines' CPU tier. The engines exchange control messages, every one is delivered, and the
// run completes with each request disaggregated; the NIXL arm of the same deployment sends none.
func TestCapabilityExample_PDOverTheP2PTier(t *testing.T) {
	t.Parallel()
	args := []string{"run", "--scenario", "pd-p2p.yaml", "--num-instances", "3", "--prefill-instances", "2",
		"--decode-instances", "1", "--rate", "2", "--num-requests", "6", "--seed", "1"}
	d := capExampleRun(t, capExamplesDir, args...)
	sent, ok := d.figureOf("peer_messages_sent")
	if !ok || sent == 0 {
		t.Fatalf("the Result states no peer messages (%v, %v): the engines did not talk", sent, ok)
	}
	if got, _ := d.figureOf("peer_messages_delivered"); got != sent {
		t.Errorf("%v peer messages sent, %v delivered", sent, got)
	}
	if got, _ := d.figureOf("peer_messages_dropped"); got != 0 {
		t.Errorf("%v peer messages dropped", got)
	}
	links := map[string]bool{}
	keysWithPrefix(d.doc, "link_peak_concurrent_transfers_nic-", links)
	if !links["link_peak_concurrent_transfers_nic-egress-node-0"] || !links["link_peak_concurrent_transfers_nic-ingress-node-2"] {
		t.Errorf("the prefill node's WRITEs did not cross its NIC to the decode node; NIC links used: %v", links)
	}
	if d.completed() != 6 || d.num("breakdowns.pd.disaggregated_requests") != 6 {
		t.Errorf("completed %d, disaggregated %v, want 6 of each", d.completed(), d.num("breakdowns.pd.disaggregated_requests"))
	}
	nixl := capExampleRun(t, capExamplesDir, "run", "--scenario", "pd-unequal-tp.yaml", "--num-instances", "5",
		"--prefill-instances", "4", "--decode-instances", "1", "--rate", "2", "--num-requests", "6", "--seed", "1")
	if n, _ := nixl.figureOf("peer_messages_sent"); n != 0 {
		t.Errorf("the NIXL control sent %v peer messages", n)
	}
}

// KV tier, object store (obj): a bucket of the catalog's s3 class below the CPU tier keeps what the
// CPU tier evicts. The Result states the bucket's link, and against the same deployment without it
// the run hits the cache more.
func TestCapabilityExample_ObjectStoreTierKeepsWhatTheCPUTierEvicts(t *testing.T) {
	t.Parallel()
	args := []string{"run", "--scenario", "obj-tier.yaml", "--num-instances", "1", "--horizon", "25000000",
		"--workload-spec", filepath.Join(capExamplesDir, "spill-workload.yaml")}
	with := capExampleRun(t, capExamplesDir, args...)
	without := capExampleRun(t, exampleVariant(t, "obj-tier.yaml", "\n    - {tier: s3, bytes: 1099511627776}", ""), args...)
	pd, _ := with.doc["breakdowns"].(map[string]any)["pd"].(map[string]any)
	if _, ok := pd["link_peak_concurrent_transfers_tier-s3-to-node-0"]; !ok {
		t.Errorf("the Result states no copy on the bucket's link: %v", pd)
	}
	if a, b := with.promptHitRate(), without.promptHitRate(); a <= b+0.02 {
		t.Errorf("cache hit rate %.3f with the bucket, %.3f without: the bucket keeps nothing", a, b)
	}
}

// P/D pools of different data-parallel width: four one-rank prefill pods and decode pods of two
// ranks each. Every request is disaggregated and handed off once, and both decode ranks serve.
func TestCapabilityExample_PDPoolsOfUnequalDataParallelWidth(t *testing.T) {
	t.Parallel()
	d := capExampleRun(t, capExamplesDir, "run", "--scenario", "pd-unequal-dp.yaml", "--num-instances", "6",
		"--prefill-instances", "4", "--decode-instances", "2", "--rate", "5", "--num-requests", "20", "--seed", "3")
	if d.completed() != 20 || d.num("breakdowns.pd.disaggregated_requests") != 20 || d.num("breakdowns.pd.kv_transfers") != 20 {
		t.Fatalf("completed %d, disaggregated %v, handoffs %v: want 20 of each", d.completed(),
			d.num("breakdowns.pd.disaggregated_requests"), d.num("breakdowns.pd.kv_transfers"))
	}
	for _, id := range []string{"instance_4", "instance_5"} {
		if n, _ := d.instances()[id]["injected_requests"].(float64); n == 0 {
			t.Errorf("decode rank %s served no request", id)
		}
	}
}

// MFU: every instance states its model-FLOPs utilization over the measured window, a fraction of the
// peak that cannot exceed its busy share, and a heavier load raises it.
func TestCapabilityExample_InstancesStateTheirMFU(t *testing.T) {
	t.Parallel()
	run := func(rate, n string) capDoc {
		return capExampleRun(t, capExamplesDir, "run", "--scenario", "node-pools.yaml", "--num-instances", "4",
			"--policy-config", filepath.Join(capExamplesDir, "node-pools-bundle.yaml"), "--rate", rate, "--num-requests", n)
	}
	light, heavy := run("2", "40"), run("40", "200")
	for id, in := range light.instances() {
		mw := in["measured_window"].(map[string]any)
		mfu, ok := mw["mfu"].(float64)
		if !ok || mfu <= 0 || mfu > mw["busy_share"].(float64) {
			t.Errorf("instance %s mfu %v, want in (0, busy_share %v]", id, mw["mfu"], mw["busy_share"])
		}
	}
	if a, b := light.num("engine.cluster.mfu"), heavy.num("engine.cluster.mfu"); b <= a {
		t.Errorf("cluster mfu %.5f at 40 req/s, %.5f at 2 req/s: more load per weight read must raise it", b, a)
	}
}
