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
	if got := resolveRouterInputs(); !reflect.DeepEqual(got.EPPFlags, llmd.DefaultEPPFlags) || got.Manifests != "" || got.PodLabels != nil {
		t.Errorf("no router flags: got %+v, want the default EPP flags %q and nothing else", got, llmd.DefaultEPPFlags)
	}

	const body = "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\n"
	routerCmd(t, "--epp-config", writeRouterFile(t, "epp.yaml", body), "--pod-labels", "app=vllm,tier=gpu",
		"--gateway-latency", "250", "--epp-scrape-jitter", "40")
	got := resolveRouterInputs()
	want := routerInputs{
		EPPFlags:         append(append([]string(nil), llmd.DefaultEPPFlags...), "--config-text="+body),
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
		if got.GatewayLatencyUs != 9 || !reflect.DeepEqual(got.EPPFlags, llmd.DefaultEPPFlags) {
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
		reconcileReplayRouter(c, &workload.TraceRouterConfig{EPPFlags: llmd.DefaultEPPFlags, GatewayLatencyUs: 100})
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

	extra := map[string]string{"app": "vllm", "llm-d.ai/role": "decode"}
	labels := podLabelFunc(extra)
	if got := labels("instance_1", 0); !reflect.DeepEqual(got, extra) {
		t.Errorf("colocated pod: got %v, want only the extra labels %v", got, extra)
	}
	got := labels("instance_2", cluster.PoolRolePrefill)
	if got["app"] != "vllm" || got["llm-d.ai/role"] != "decode" {
		t.Errorf("prefill pod with extra labels: got %v; the extra labels apply on top of the role label", got)
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
