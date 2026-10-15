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
// sample configuration with the always-disagg-pd-decider). Tests that need the P/D path pass
// it with --epp-config; the router's default configuration does not disaggregate.
const pdAlwaysEPPConfig = "testdata/epp/pd-always.yaml"

// writeRouterFile writes body to a fresh file named name and returns its path.
func writeRouterFile(t *testing.T, name, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadEPPFlags(t *testing.T) {
	for _, tt := range []struct {
		name string
		body string
		want []string
	}{
		{"list form is taken verbatim, in order",
			"- --pool-name=vllm\n- --v\n- \"4\"\n- --endpoint-selector=app=x\n",
			[]string{"--pool-name=vllm", "--v", "4", "--endpoint-selector=app=x"}},
		{"map form (Helm epp.flags) is sorted by name into --k=v",
			"v: 4\nzeta: \"true\"\npool-name: vllm\nalpha: a=b\n",
			[]string{"--alpha=a=b", "--pool-name=vllm", "--v=4", "--zeta=true"}},
		{"map form: a key already spelled --name is not doubled",
			"--pool-name: vllm\n", []string{"--pool-name=vllm"}},
		{"an empty file names no flags", "", nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := readEPPFlags(writeRouterFile(t, "flags.yaml", tt.body))
			if err != nil {
				t.Fatalf("readEPPFlags: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestReadEPPFlags_RejectsOtherShapes(t *testing.T) {
	for _, tt := range []struct{ name, body string }{
		{"a scalar", "--pool-name=vllm\n"},
		{"a list of maps", "- name: v\n  value: 4\n"},
		{"a map of lists", "v:\n  - 4\n"},
		{"malformed YAML", "[unclosed\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got, err := readEPPFlags(writeRouterFile(t, "flags.yaml", tt.body)); err == nil {
				t.Errorf("accepted %q as %q; want an error", tt.body, got)
			}
		})
	}
	if _, err := readEPPFlags(filepath.Join(t.TempDir(), "absent.yaml")); err == nil {
		t.Error("a missing flags file must be an error")
	}
}

// inlineConfigFile carries the configuration itself into the trace header, never a path the
// replaying host may not have (INV-13).
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
			if got := inlineConfigFile(tt.in); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

// saveRouterVars snapshots the router flag variables and restores them at test end.
func saveRouterVars(t *testing.T) {
	t.Helper()
	flags, cfg, manifests, labels := eppFlagsPath, eppConfigPath, eppManifestPaths, podLabelPairs
	latency, jitter := gatewayLatencyUs, scrapeJitterUs
	t.Cleanup(func() {
		eppFlagsPath, eppConfigPath, eppManifestPaths, podLabelPairs = flags, cfg, manifests, labels
		gatewayLatencyUs, scrapeJitterUs = latency, jitter
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

func TestResolveRouterInputs_DefaultsAndEPPConfig(t *testing.T) {
	saveRouterVars(t)
	routerCmd(t)
	if got := resolveRouterInputs(); !reflect.DeepEqual(got.EPPFlags, llmd.DefaultEPPFlags()) || got.Manifests != "" || got.PodLabels != nil {
		t.Errorf("no router flags: got %+v, want the default EPP flags %q and nothing else", got, llmd.DefaultEPPFlags())
	}

	const body = "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\n"
	routerCmd(t, "--epp-config", writeRouterFile(t, "epp.yaml", body), "--pod-labels", "app=vllm,tier=gpu",
		"--gateway-latency", "250", "--epp-scrape-jitter", "40")
	got := resolveRouterInputs()
	want := routerInputs{
		EPPFlags:         append(append([]string(nil), llmd.DefaultEPPFlags()...), "--config-text="+body),
		PodLabels:        map[string]string{"app": "vllm", "tier": "gpu"},
		GatewayLatencyUs: 250,
		ScrapeJitterUs:   40,
	}
	if !reflect.DeepEqual(got, want) {
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

	t.Run("no header: the command line", func(t *testing.T) {
		c := routerCmd(t, "--gateway-latency", "9")
		got := reconcileReplayRouter(c, nil)
		if got.GatewayLatencyUs != 9 || !reflect.DeepEqual(got.EPPFlags, llmd.DefaultEPPFlags()) {
			t.Errorf("got %+v, want the command line's router (gateway latency 9, default EPP flags)", got)
		}
	})
	t.Run("header and no router flags: the header", func(t *testing.T) {
		if got := reconcileReplayRouter(routerCmd(t), recorded); !reflect.DeepEqual(got, fromHeader) {
			t.Errorf("got %+v, want the recorded router %+v", got, fromHeader)
		}
	})
	t.Run("header and router flags restating it: the header", func(t *testing.T) {
		flags := writeRouterFile(t, "flags.yaml", "- --endpoint-selector=app=recorded\n- --config-text=recorded\n")
		manifests := writeRouterFile(t, "pool.yaml", "kind: InferencePool\n")
		c := routerCmd(t, "--epp-flags", flags, "--epp-manifests", manifests, "--pod-labels", "app=recorded",
			"--gateway-latency", "100", "--epp-scrape-jitter", "7")
		if got := reconcileReplayRouter(c, recorded); !reflect.DeepEqual(got, fromHeader) {
			t.Errorf("got %+v, want the recorded router %+v", got, fromHeader)
		}
	})
}

// A replay whose router flags contradict the router its trace was recorded with is refused
// (logrus.Fatalf, exit 1), never silently preferred over the header (INV-13).
func TestReconcileReplayRouter_ContradictingFlagsFatal(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		c := &cobra.Command{}
		registerRouterFlags(c)
		if err := c.ParseFlags([]string{"--gateway-latency", "5"}); err != nil {
			os.Exit(2)
		}
		reconcileReplayRouter(c, &workload.TraceRouterConfig{EPPFlags: llmd.DefaultEPPFlags(), GatewayLatencyUs: 100})
		os.Exit(0) // must Fatalf before here
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestReconcileReplayRouter_ContradictingFlagsFatal$")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
		t.Fatalf("want exit 1 (logrus.Fatalf) for contradicting router flags, got %v:\n%s", err, out)
	}
	if !strings.Contains(string(out), "router flags differ") {
		t.Errorf("the refusal must name the contradiction, got:\n%s", out)
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
		for _, name := range []string{"epp-flags", "epp-config", "epp-manifests", "pod-labels", "gateway-latency", "epp-scrape-jitter"} {
			if c.Flags().Lookup(name) == nil {
				t.Errorf("%s must register the router input --%s", c.Name(), name)
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

// TestINV13_RunReplayParity_LLMDRouterInputs: a run that sets every router input (EPP flags in
// pool mode, an EndpointPickerConfig with flow control, manifests, pod labels, gateway latency,
// scrape jitter) exports them in its trace header, and a replay given no router flags reproduces
// the run byte for byte.
func TestINV13_RunReplayParity_LLMDRouterInputs(t *testing.T) {
	flagsFile := writeRouterFile(t, "epp-flags.yaml", "- --pool-name=vllm\n- --refresh-metrics-interval=20ms\n")
	cfg := writeRouterFile(t, "epp-config.yaml", inv13Config)
	manifests := writeRouterFile(t, "pool.yaml", inv13Manifests)
	prefix := filepath.Join(t.TempDir(), "trace")
	horizon := "600000000"
	runOut, stderr, err := runKernelCLI(t, "run", "--scenarios", pdScenarios, "--scenario", capScenario, "--num-instances", "2",
		"--rate", "2000", "--num-requests", "1500", "--seed", "7", "--horizon", horizon,
		"--epp-flags", flagsFile, "--epp-config", cfg, "--epp-manifests", manifests,
		"--pod-labels", "app=vllm", "--gateway-latency", "300", "--epp-scrape-jitter", "5000",
		"--trace-output", prefix)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 15))
	}
	if completed := clusterMetricInt(t, runOut, "completed_requests"); completed <= 0 {
		t.Fatalf("parity check would be vacuous: the run completed %d requests", completed)
	}
	if !strings.Contains(runOut, "Gateway Queue Rejected:") {
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
	replayOut, stderr, err := runKernelCLI(t, "replay", "--scenarios", pdScenarios, "--scenario", capScenario, "--num-instances", "2",
		"--trace-header", prefix+".yaml", "--trace-data", prefix+".csv", "--horizon", horizon)
	if err != nil {
		t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 5))
	}
	if replayOut != runOut {
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
	if replayOut != runOut {
		t.Errorf("INV-13: the replay differs from the run\n--- run\n%s\n--- replay\n%s", runOut, replayOut)
	}
}

// TestResolveRouterInputs_PodLabelsAreScopedAndRoleIsReserved: --pod-labels accepts role-scoped
// labels; an unknown role, or an explicit llm-d.ai/role (which comes from the scenario's pools), is
// refused.
func TestResolveRouterInputs_PodLabelsAreScopedAndRoleIsReserved(t *testing.T) {
	saved := podLabelPairs
	t.Cleanup(func() { podLabelPairs = saved })
	podLabelPairs = []string{"app=vllm", "decode:app=vllm-decode"}
	if got := resolveRouterInputs().PodLabels; !reflect.DeepEqual(got, map[string]string{"app": "vllm", "decode:app": "vllm-decode"}) {
		t.Errorf("got %v", got)
	}
	for _, bad := range []string{"llm-d.ai/role=decode", "gpu:app=x", "decode:llm-d.ai/role=prefill"} {
		if os.Getenv("BLIS_POD_LABELS_LEG") == bad {
			podLabelPairs = []string{bad}
			resolveRouterInputs()
			return
		}
		cmd := exec.Command(os.Args[0], "-test.run=^TestResolveRouterInputs_PodLabelsAreScopedAndRoleIsReserved$")
		cmd.Env = append(os.Environ(), "BLIS_POD_LABELS_LEG="+bad)
		if out, err := cmd.CombinedOutput(); err == nil || !strings.Contains(string(out), "--pod-labels") {
			t.Errorf("--pod-labels %s: want a fatal refusal, got err=%v\n%s", bad, err, out)
		}
	}
}
