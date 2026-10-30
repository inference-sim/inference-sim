package cmd

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/llmd"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// pdAlwaysEPPConfig is an EndpointPickerConfig that disaggregates every request (llm-d's P/D
// sample configuration with the always-disagg-pd-decider), relative to pdScenarios. The P/D
// scenarios there name it in their router block; the router's default configuration does not
// disaggregate.
const pdAlwaysEPPConfig = "router/pd-always.yaml"

// withPDRouter copies the P/D scenarios' router configuration (pdAlwaysEPPConfig) into dir, a
// directory holding a variant of one of them, so the variant's relative router block resolves.
func withPDRouter(t *testing.T, dir string) string {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(pdScenarios, pdAlwaysEPPConfig))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, filepath.Dir(pdAlwaysEPPConfig)), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, pdAlwaysEPPConfig), body, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// writeRouterFile writes body to a fresh file named name and returns its path.
func writeRouterFile(t *testing.T, name, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestInlineConfigFile(t *testing.T) {
	const body = "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\n"
	path := writeRouterFile(t, "epp.yaml", body)
	for _, tt := range []struct {
		name string
		in   []string
		want []string
	}{
		{"--config-file=path", []string{"--a=1", "--config-file=" + path, "--b"},
			[]string{"--a=1", "--config-text=" + body, "--b"}},
		{"--config-file path (two arguments)", []string{"--a=1", "--config-file", path, "--b"},
			[]string{"--a=1", "--config-text=" + body, "--b"}},
		{"no config file: unchanged", []string{"--a=1", "--config-text=x"}, []string{"--a=1", "--config-text=x"}},
		{"a trailing bare --config-file is left to the router to refuse", []string{"--config-file"}, []string{"--config-file"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got, err := inlineConfigFile(tt.in, func(p string) string { return p }); err != nil || !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got %q, %v; want %q", got, err, tt.want)
			}
		})
	}
}

// saveRouterVars snapshots the router variables and restores them at test end.
func saveRouterVars(t *testing.T) {
	t.Helper()
	latency, jitter, delay := gatewayLatencyUs, scrapeJitterUs, kvEventDelayUs
	stated, states, scen := scenarioRouter, scenarioStatesRouter, kernelScenario
	t.Cleanup(func() {
		gatewayLatencyUs, scrapeJitterUs, kvEventDelayUs = latency, jitter, delay
		scenarioRouter, scenarioStatesRouter, kernelScenario = stated, states, scen
	})
}

// routerCmd is a command carrying only the router flags, parsed from args.
func routerCmd(t *testing.T, args ...string) *cobra.Command {
	t.Helper()
	c := &cobra.Command{}
	registerRouterFlags(c)
	if err := c.ParseFlags(args); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	return c
}

// A run's router is the scenario's deployment half (the default standalone router when it states
// none) and the command line's environment half.
func TestResolveRouterInputs_ScenarioAndEnvironment(t *testing.T) {
	saveRouterVars(t)
	scenarioRouter, scenarioStatesRouter = routerInputs{}, false
	routerCmd(t)
	if got := resolveRouterInputs(); !reflect.DeepEqual(got.EPPFlags, llmd.DefaultEPPFlags()) || got.Manifests != "" || got.PodLabels != nil {
		t.Errorf("no router stated: got %+v, want the default EPP flags %q and nothing else", got, llmd.DefaultEPPFlags())
	}
	scenarioRouter = routerInputs{EPPFlags: []string{"--pool-name=vllm"}, Manifests: "kind: InferencePool", PodLabels: map[string]string{"app": "vllm"}}
	scenarioStatesRouter = true
	routerCmd(t, "--gateway-latency", "250", "--epp-scrape-jitter", "40")
	want := routerInputs{EPPFlags: []string{"--pool-name=vllm"}, Manifests: "kind: InferencePool",
		PodLabels: map[string]string{"app": "vllm"}, GatewayLatencyUs: 250, ScrapeJitterUs: 40}
	if got := resolveRouterInputs(); !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

func TestReconcileReplayRouter(t *testing.T) {
	saveRouterVars(t)
	recorded := &workload.TraceRouterConfig{
		EPPFlags:         []string{"--endpoint-selector=app=recorded", "--config-text=recorded"},
		Manifests:        "kind: InferencePool",
		PodLabels:        map[string]string{"app": "recorded"},
		GatewayLatencyUs: 100,
		ScrapeJitterUs:   7,
	}
	fromHeader := routerInputs{
		EPPFlags: recorded.EPPFlags, Manifests: recorded.Manifests, PodLabels: recorded.PodLabels,
		GatewayLatencyUs: recorded.GatewayLatencyUs, ScrapeJitterUs: recorded.ScrapeJitterUs,
	}

	t.Run("no header: the scenario and the command line", func(t *testing.T) {
		scenarioRouter, scenarioStatesRouter = routerInputs{}, false
		got := reconcileReplayRouter(routerCmd(t, "--gateway-latency", "9"), nil)
		if got.GatewayLatencyUs != 9 || !reflect.DeepEqual(got.EPPFlags, llmd.DefaultEPPFlags()) {
			t.Errorf("got %+v, want the command line's router (gateway latency 9, default EPP flags)", got)
		}
	})
	t.Run("header, a scenario stating no router and no flags: the header", func(t *testing.T) {
		scenarioRouter, scenarioStatesRouter = routerInputs{}, false
		if got := reconcileReplayRouter(routerCmd(t), recorded); !reflect.DeepEqual(got, fromHeader) {
			t.Errorf("got %+v, want the recorded router %+v", got, fromHeader)
		}
	})
	t.Run("header, a scenario and flags restating it: the header", func(t *testing.T) {
		scenarioRouter = routerInputs{EPPFlags: recorded.EPPFlags, Manifests: recorded.Manifests, PodLabels: recorded.PodLabels}
		scenarioStatesRouter = true
		c := routerCmd(t, "--gateway-latency", "100", "--epp-scrape-jitter", "7")
		if got := reconcileReplayRouter(c, recorded); !reflect.DeepEqual(got, fromHeader) {
			t.Errorf("got %+v, want the recorded router %+v", got, fromHeader)
		}
	})
}

// A replay whose router flags, or whose scenario's router, contradicts the router its trace was
// recorded with is refused (logrus.Fatalf, exit 1), never silently preferred over the header
// (INV-13).
func TestReconcileReplayRouter_ContradictionsFatal(t *testing.T) {
	recorded := &workload.TraceRouterConfig{EPPFlags: llmd.DefaultEPPFlags(), GatewayLatencyUs: 100}
	legs := map[string]struct {
		set  func()
		args []string
		want string
	}{
		"flag": {func() {}, []string{"--gateway-latency", "5"}, "router flags differ"},
		"scenario": {func() {
			scenarioRouter = routerInputs{EPPFlags: llmd.DefaultEPPFlags(), PodLabels: map[string]string{"app": "vllm"}}
			scenarioStatesRouter = true
		}, nil, "differs from the router the trace was recorded with"},
	}
	if leg, ok := legs[os.Getenv("BLIS_RECONCILE_LEG")]; ok {
		leg.set()
		reconcileReplayRouter(routerCmd(t, leg.args...), recorded)
		os.Exit(0) // must Fatalf before here
	}
	for name, leg := range legs {
		cmd := exec.Command(os.Args[0], "-test.run=^TestReconcileReplayRouter_ContradictionsFatal$")
		cmd.Env = append(os.Environ(), "BLIS_RECONCILE_LEG="+name)
		out, err := cmd.CombinedOutput()
		var exitErr *exec.ExitError
		if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
			t.Errorf("%s: want exit 1 (logrus.Fatalf) for a contradicting replay, got %v:\n%s", name, err, out)
			continue
		}
		if !strings.Contains(string(out), leg.want) {
			t.Errorf("%s: the refusal must name the contradiction, got:\n%s", name, out)
		}
	}
}

func TestPodLabelFunc_RoleLabels(t *testing.T) {
	plain := podLabelFunc(nil)
	for _, tt := range []struct {
		role cluster.PoolRole
		want map[string]string
	}{
		{0, map[string]string{}},
		{cluster.PoolRolePrefill, map[string]string{"llm-d.ai/role": "prefill"}},
		{cluster.PoolRoleDecode, map[string]string{"llm-d.ai/role": "decode"}},
		{cluster.PoolRolePrefillDecode, map[string]string{"llm-d.ai/role": "prefill-decode"}},
		{cluster.PoolRoleEncode, map[string]string{"llm-d.ai/role": "encode"}},
	} {
		if got := plain("instance_0", tt.role); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("role %v: got %v, want %v", tt.role, got, tt.want)
		}
	}

	extra := map[string]string{"app": "vllm", "decode:app": "vllm-decode", "prefill:tier": "fast"}
	labels := podLabelFunc(extra)
	if got := labels("instance_1", 0); !reflect.DeepEqual(got, map[string]string{"app": "vllm"}) {
		t.Errorf("colocated pod: got %v, want only the unscoped labels", got)
	}
	got := labels("instance_2", cluster.PoolRolePrefill)
	if want := (map[string]string{"app": "vllm", "tier": "fast", "llm-d.ai/role": "prefill"}); !reflect.DeepEqual(got, want) {
		t.Errorf("prefill pod: got %v, want %v", got, want)
	}
	if got := labels("instance_4", cluster.PoolRoleDecode); got["app"] != "vllm-decode" || got["tier"] != "" {
		t.Errorf("decode pod: got %v; a role-scoped label overrides the unscoped one and applies to its role only", got)
	}
	got["mutated"] = "x"
	if _, leaked := labels("instance_3", cluster.PoolRolePrefill)["mutated"]; leaked {
		t.Error("each call must return a fresh map")
	}
}

// The BLIS-native routing, admission, flow-control and P/D-decider flags are gone: the
// embedded llm-d router is configured only by its own inputs (cmd/router.go).
func TestRunAndReplay_BLISRoutingFlagsRemoved(t *testing.T) {
	removed := []string{
		"admission-policy", "admission-latency", "routing-latency", "token-bucket-capacity",
		"token-bucket-refill-rate", "routing-policy", "routing-scorers", "lora-scorer-weight",
		"snapshot-refresh-interval", "cache-signal-delay", "pd-decider", "pd-prefix-threshold",
		"prefill-routing-scorers", "decode-routing-scorers", "encode-decider",
		"flow-control", "saturation-detector", "dispatch-order", "slo-targets",
		"max-gateway-queue-depth", "queue-depth-threshold", "kv-cache-util-threshold",
		"max-concurrency", "per-band-capacity", "usage-limit-threshold", "fairness-policy",
		"request-ttl", "queue-shedding", "dispatch-tick-interval", "in-flight-eviction",
	}
	for _, c := range []*cobra.Command{runCmd, replayCmd} {
		for _, name := range removed {
			if c.Flags().Lookup(name) != nil {
				t.Errorf("%s still registers --%s", c.Name(), name)
			}
		}
		for _, name := range []string{"gateway-latency", "epp-scrape-jitter", "kv-event-delay"} {
			if c.Flags().Lookup(name) == nil {
				t.Errorf("%s must register the router environment flag --%s", c.Name(), name)
			}
		}
		// The router's deployment inputs are the scenario's router block and pool labels.
		for _, name := range []string{"epp-flags", "epp-config", "epp-manifests", "pod-labels"} {
			if c.Flags().Lookup(name) != nil {
				t.Errorf("%s still registers --%s; the scenario states the router (#1950)", c.Name(), name)
			}
		}
	}
}

// inv13Manifests is an InferencePool deployment with one objective; inv13Config bounds flow
// control so a burst both queues and rejects.
const inv13Manifests = `apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: vllm
spec:
  selector:
    matchLabels:
      app: vllm
  targetPorts:
  - number: 8000
  endpointPickerRef:
    name: epp
    port:
      number: 9002
---
apiVersion: llm-d.ai/v1
kind: InferenceObjective
metadata:
  name: standard
spec:
  priority: 1
  poolRefs:
  - name: vllm
`

const inv13Config = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: queue-scorer
- type: kv-cache-utilization-scorer
- type: prefix-cache-scorer
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
    weight: 2
  - pluginRef: kv-cache-utilization-scorer
  - pluginRef: prefix-cache-scorer
    weight: 3
  - pluginRef: max-score-picker
flowControl:
  maxRequests: 8
  defaultRequestTTL: 2s
`

// noRenderQueue makes each request's admission overhead an uncontended delay (#1946's render
// stage off, --renderer-num-workers 0). These fixtures drive the router's flow control at
// thousands of requests per second; with vLLM's default single render thread per API server
// the render queue, which vLLM's num_requests_waiting does not count, would hold the requests
// before any engine saturates, and flow control would never act.
var noRenderQueue = []string{"--renderer-num-workers", "0"}

// TestINV13_RunReplayParity_LLMDRouterInputs: a run whose scenario states every router input (a
// router block with EPP flags in pool mode, an EndpointPickerConfig with flow control and
// manifests, and pool labels) and that sets the environment (gateway latency, scrape jitter)
// exports them in its trace header, and a replay given the same scenario and no router flags
// reproduces the run byte for byte.
func TestINV13_RunReplayParity_LLMDRouterInputs(t *testing.T) {
	dir := colocatedRouterVariant(t, "router:\n  epp_flags: [--pool-name=vllm, --refresh-metrics-interval=20ms]\n"+
		"  endpoint_picker_config: epp-config.yaml\n  manifests: [pool.yaml]\n", "      app: vllm\n",
		map[string]string{"epp-config.yaml": writeRouterFile(t, "epp-config.yaml", inv13Config),
			"pool.yaml": writeRouterFile(t, "pool.yaml", inv13Manifests)})
	prefix := filepath.Join(t.TempDir(), "trace")
	horizon := "600000000"
	runOut, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "2",
		"--rate", "2000", "--num-requests", "1500", "--seed", "7", "--horizon", horizon,
		"--gateway-latency", "300", "--epp-scrape-jitter", "5000", noRenderQueue[0], noRenderQueue[1],
		"--trace-output", prefix)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 15))
	}
	if completed := clusterMetricInt(t, runOut, "completed_requests"); completed <= 0 {
		t.Fatalf("parity check would be vacuous: the run completed %d requests", completed)
	}
	if parseResult(t, runOut).Requests.GatewayQueueRejected == 0 {
		t.Errorf("the bounded flow-control queue never rejected; the router inputs may not have taken effect\n%s", runOut)
	}
	header, err := os.ReadFile(prefix + ".yaml")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"--pool-name=vllm", "maxRequests: 8", "kind: InferencePool", "app: vllm"} {
		if !strings.Contains(string(header), want) {
			t.Errorf("trace header does not record %q", want)
		}
	}
	// No --seed: the router seed comes from the header, and the router is shown canonical prompts,
	// so the replay's own placeholder tokens do not matter.
	replayOut, stderr, err := runKernelCLI(t, "replay", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "2",
		"--trace-header", prefix+".yaml", "--trace-data", prefix+".csv", "--horizon", horizon, noRenderQueue[0], noRenderQueue[1])
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 5))
	}
	if inv13View(t, replayOut) != inv13View(t, runOut) {
		t.Errorf("INV-13: the replay differs from the run\n--- run\n%s\n--- replay\n%s", runOut, replayOut)
	}
}

// TestINV13_RunReplayParity_PrefixSharingWorkload: with the default router (whose prefix-cache
// scorer reads prompt content) on a workload where most requests share a prefix, a replay given
// the same scenario and no other flags reproduces a four-instance run byte for byte.
func TestINV13_RunReplayParity_PrefixSharingWorkload(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "trace")
	horizon := "9223372036854775807"
	runOut, stderr, err := runKernelCLI(t, "run", "--scenarios", pdScenarios, "--scenario", capScenario, "--num-instances", "4",
		"--workload-spec", "../examples/prefix-affinity-demo.yaml", "--seed", "1", "--horizon", horizon, "--trace-output", prefix)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 15))
	}
	if ids := instanceIDs(t, runOut); len(ids) != 4 {
		t.Fatalf("want 4 instances, got %v", ids)
	}
	replayOut, stderr, err := runKernelCLI(t, "replay", "--scenarios", pdScenarios, "--scenario", capScenario, "--num-instances", "4",
		"--trace-header", prefix+".yaml", "--trace-data", prefix+".csv", "--horizon", horizon)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 5))
	}
	if inv13View(t, replayOut) != inv13View(t, runOut) {
		t.Errorf("INV-13: the replay differs from the run\n--- run\n%s\n--- replay\n%s", runOut, replayOut)
	}
	// Non-vacuity: on this workload the default router's placement depends on prompt content
	// (its prefix-cache scorer), so a content-blind configuration routes differently.
	loadOnlyDir := colocatedRouterVariant(t, "router:\n  endpoint_picker_config: load-only.yaml\n", "",
		map[string]string{"load-only.yaml": "../examples/llm-d/load-only.yaml"})
	loadOnly, stderr, err := runKernelCLI(t, "run", "--scenarios", loadOnlyDir, "--scenario", "variant.yaml", "--num-instances", "4",
		"--workload-spec", "../examples/prefix-affinity-demo.yaml", "--seed", "1", "--horizon", horizon)
	if err != nil {
		t.Fatalf("load-only run: %v\n%s", err, lastLines(stderr, 5))
	}
	if loadOnly == runOut {
		t.Error("the default and load-only configurations route this workload identically; the parity check does not exercise content-sensitive routing")
	}
}

// TestINV13_RunReplayParity_PrecisePrefix: a four-instance run routed by the precise prefix cache
// (KV events from the engines, delivered with a delay, speculative indexing on), replayed with no
// router flags, reproduces the run byte for byte; the delay is recorded in the header.
func TestINV13_RunReplayParity_PrecisePrefix(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "trace")
	horizon := "9223372036854775807"
	dir := colocatedRouterVariant(t, "router:\n  endpoint_picker_config: precise.yaml\n", "",
		map[string]string{"precise.yaml": "testdata/epp/precise.yaml"})
	runOut, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "4",
		"--workload-spec", "../examples/prefix-affinity-demo.yaml", "--seed", "1", "--horizon", horizon,
		"--kv-event-delay", "3000", "--trace-output", prefix)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 15))
	}
	header, err := os.ReadFile(prefix + ".yaml")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(header), "kv_event_delay_us: 3000") {
		t.Errorf("trace header does not record the KV event delay")
	}
	replayOut, stderr, err := runKernelCLI(t, "replay", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "4",
		"--trace-header", prefix+".yaml", "--trace-data", prefix+".csv", "--horizon", horizon)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 5))
	}
	if inv13View(t, replayOut) != inv13View(t, runOut) {
		t.Errorf("INV-13: the replay differs from the run\n--- run\n%s\n--- replay\n%s", runOut, replayOut)
	}
}

// TestINV13_RunReplayParity_RequestTTL: flow control ordered by edf-ordering-policy, with one
// client's requests carrying a 20 s lifetime (x-llm-d-inference-ttl) and the other's none; the
// trace records each request's deadline, so a replay with no router flags reproduces the run
// byte for byte.
func TestINV13_RunReplayParity_RequestTTL(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "trace")
	horizon := "9223372036854775807"
	dir := colocatedRouterVariant(t, "router:\n  endpoint_picker_config: edf.yaml\n", "",
		map[string]string{"edf.yaml": "testdata/epp/edf.yaml"})
	runOut, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "2",
		"--workload-spec", "testdata/request-ttl-workload.yaml", "--seed", "1", "--horizon", horizon,
		"--trace-output", prefix)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 15))
	}
	replayOut, stderr, err := runKernelCLI(t, "replay", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "2",
		"--trace-header", prefix+".yaml", "--trace-data", prefix+".csv", "--horizon", horizon)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 5))
	}
	if inv13View(t, replayOut) != inv13View(t, runOut) {
		t.Errorf("INV-13: the replay differs from the run\n--- run\n%s\n--- replay\n%s", runOut, replayOut)
	}
}

// TestINV13_RunReplayParity_InFlightEviction: a pool with critical and sheddable objectives under
// flow control with in-flight eviction; the replay reproduces the run byte for byte, evictions
// included.
func TestINV13_RunReplayParity_InFlightEviction(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "trace")
	horizon := "9223372036854775807"
	dir := colocatedRouterVariant(t, evictionRouter("eviction.yaml"), "      app: vllm\n", evictionFiles("eviction.yaml"))
	runOut, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "2",
		"--workload-spec", "testdata/eviction-workload.yaml", "--seed", "1", "--horizon", horizon, "--trace-output", prefix,
		noRenderQueue[0], noRenderQueue[1])
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 15))
	}
	if parseResult(t, runOut).Requests.GatewayEvicted == 0 {
		t.Fatalf("no request was evicted in flight; the test does not exercise the path\n%s", runOut)
	}
	replayOut, stderr, err := runKernelCLI(t, "replay", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "2",
		"--trace-header", prefix+".yaml", "--trace-data", prefix+".csv", "--horizon", horizon, noRenderQueue[0], noRenderQueue[1])
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 5))
	}
	if inv13View(t, replayOut) != inv13View(t, runOut) {
		t.Errorf("INV-13: the replay differs from the run\n--- run\n%s\n--- replay\n%s", runOut, replayOut)
	}
}

// evictionRouter is a router block in pool mode (--pool-name=vllm, testdata/epp/objectives.yaml's pool)
// naming the EndpointPickerConfig cfg and the critical/sheddable objectives.
func evictionRouter(cfg string) string {
	return "router:\n  epp_flags: [--pool-name=vllm]\n  endpoint_picker_config: " + cfg + "\n  manifests: [objectives.yaml]\n"
}

// evictionFiles are the files evictionRouter names, copied from testdata/epp.
func evictionFiles(cfg string) map[string]string {
	return map[string]string{cfg: "testdata/epp/" + cfg, "objectives.yaml": "testdata/epp/objectives.yaml"}
}

// pdRouterVariant is the committed 3P1D scenario with its router block replaced by router and each
// pool given labels (YAML lines indented under the pool; "" for none).
func pdRouterVariant(t *testing.T, router, prefillLabels, decodeLabels string, files map[string]string) string {
	t.Helper()
	edits := []string{"router:\n  endpoint_picker_config: " + pdAlwaysEPPConfig + "\n", ""}
	if prefillLabels != "" {
		edits = append(edits, "  - role: prefill\n    nodes: 3\n", "  - role: prefill\n    nodes: 3\n    labels:\n"+prefillLabels)
	}
	if decodeLabels != "" {
		edits = append(edits, "  - role: decode\n    nodes: 1\n", "  - role: decode\n    nodes: 1\n    labels:\n"+decodeLabels)
	}
	return routerVariant(t, filepath.Join(pdScenarios, "glm-5-h200-3p1d-ib.yaml"), router, files, edits...)
}

// TestINV13_RunReplayParity_InFlightEvictionDisaggregated: every request disaggregated, under flow
// control with in-flight eviction; disaggregated victims are revoked at their stage, and the
// replay reproduces the run byte for byte.
func TestINV13_RunReplayParity_InFlightEvictionDisaggregated(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "trace")
	horizon := "9223372036854775807"
	// Per-pool labels: both pools carry the InferencePool's selector label, and each its own tier.
	dir := pdRouterVariant(t, evictionRouter("pd-eviction.yaml"), "      app: vllm\n      tier: fast\n",
		"      app: vllm\n      tier: slow\n", evictionFiles("pd-eviction.yaml"))
	common := []string{"--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "4",
		"--prefill-instances", "3", "--decode-instances", "1", "--horizon", horizon, noRenderQueue[0], noRenderQueue[1]}
	runOut, stderr, err := runKernelCLI(t, append([]string{"run", "--workload-spec", "testdata/eviction-workload.yaml",
		"--seed", "1", "--trace-output", prefix}, common...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 15))
	}
	if parseResult(t, runOut).Requests.GatewayEvicted == 0 {
		t.Fatalf("no request was evicted in flight; the test does not exercise the path\n%s", runOut)
	}
	header := readFile(t, prefix+".yaml")
	for _, want := range []string{"app: vllm", "prefill:tier: fast", "decode:tier: slow", "--pool-name=vllm", "kind: InferenceObjective"} {
		if !strings.Contains(header, want) {
			t.Errorf("trace header does not record the scenario's router input %q", want)
		}
	}
	replayOut, stderr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv"}, common...)...)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 5))
	}
	if inv13View(t, replayOut) != inv13View(t, runOut) {
		t.Errorf("INV-13: the replay differs from the run\n--- run\n%s\n--- replay\n%s", runOut, replayOut)
	}
}
