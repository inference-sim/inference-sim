package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/llmd"
	_ "github.com/inference-sim/inference-sim/sim/vendored" // the adapter's vendoring root
)

// routerVariant writes the scenario file src into a fresh scenario directory as variant.yaml with
// the deployment's router block set to router (YAML, appended verbatim; "" states none), each
// old->new edit applied, and each files entry (name in the directory -> source path) copied in, so
// the block's relative paths name them. It returns the directory.
func routerVariant(t *testing.T, src, router string, files map[string]string, edits ...string) string {
	t.Helper()
	body := readFile(t, src)
	for i := 0; i+1 < len(edits); i += 2 {
		if !strings.Contains(body, edits[i]) {
			t.Fatalf("%s has no %q to edit", src, edits[i])
		}
		body = strings.Replace(body, edits[i], edits[i+1], 1)
	}
	if router != "" {
		body = strings.TrimRight(body, "\n") + "\n\n" + router
	}
	dir := t.TempDir()
	for name, from := range files {
		dst := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dst, []byte(readFile(t, from)), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// colocatedRouterVariant is routerVariant over capScenario, whose one colocated pool may be given
// labels (YAML lines, indented under the pool) -- "" for none.
func colocatedRouterVariant(t *testing.T, router, labels string, files map[string]string) string {
	t.Helper()
	var edits []string
	if labels != "" {
		edits = []string{"  - role: colocated\n    nodes: 4\n", "  - role: colocated\n    nodes: 4\n    labels:\n" + labels}
	}
	return routerVariant(t, filepath.Join(pdScenarios, capScenario), router, files, edits...)
}

// absPath is p made absolute.
func absPath(t *testing.T, p string) string {
	t.Helper()
	abs, err := filepath.Abs(p)
	if err != nil {
		t.Fatal(err)
	}
	return abs
}

// The scenario's router block and pool labels map onto the router inputs exactly as the router
// reads them: flags verbatim (or the default standalone router), the EndpointPickerConfig inlined
// as --config-text, a --config-file flag inlined the same way, the manifests joined into one YAML
// stream, and the pools' labels onto role-scoped pod labels. Paths are the scenario file's.
func TestScenarioRouterInputs_MapsTheRouterBlock(t *testing.T) {
	const cfg = "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\n"
	dir := t.TempDir()
	for name, body := range map[string]string{"epp.yaml": cfg, "a.yaml": "kind: InferencePool\n\n", "sub/b.yaml": "kind: InferenceObjective\n"} {
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	colocated := []deployment.Role{deployment.RoleColocated}
	pd := []deployment.Role{deployment.RolePrefill, deployment.RoleDecode}
	for _, tt := range []struct {
		name string
		sh   kernelmodel.Shape
		want routerInputs
	}{
		{"no router block, no labels: the default standalone router",
			kernelmodel.Shape{Roles: colocated, Labels: []map[string]string{nil}, Dir: dir},
			routerInputs{EPPFlags: llmd.DefaultEPPFlags()}},
		{"endpoint_picker_config and manifests, relative to the scenario",
			kernelmodel.Shape{Roles: colocated, Labels: []map[string]string{nil}, Dir: dir,
				Router: &deployment.Router{EndpointPickerConfig: "epp.yaml", Manifests: []string{"a.yaml", "sub/b.yaml"}}},
			routerInputs{EPPFlags: append(append([]string(nil), llmd.DefaultEPPFlags()...), "--config-text="+cfg),
				Manifests: "kind: InferencePool\n---\nkind: InferenceObjective"}},
		{"epp_flags verbatim, a --config-file among them inlined",
			kernelmodel.Shape{Roles: colocated, Labels: []map[string]string{nil}, Dir: dir,
				Router: &deployment.Router{EPPFlags: []string{"--pool-name=vllm", "--config-file=epp.yaml", "--v=4"}}},
			routerInputs{EPPFlags: []string{"--pool-name=vllm", "--config-text=" + cfg, "--v=4"}}},
		{"a colocated pool's labels label every pod",
			kernelmodel.Shape{Roles: colocated, Labels: []map[string]string{{"app": "vllm", "tier": "gpu"}}, Dir: dir},
			routerInputs{EPPFlags: llmd.DefaultEPPFlags(), PodLabels: map[string]string{"app": "vllm", "tier": "gpu"}}},
		{"P/D pools: a shared label is unscoped, the rest scoped to the pool's role, llm-d.ai/role BLIS's",
			kernelmodel.Shape{Roles: pd, Dir: dir, Labels: []map[string]string{
				{"app": "vllm", "llm-d.ai/role": "prefill", "tier": "fast"},
				{"app": "vllm", "tier": "slow"}}},
			routerInputs{EPPFlags: llmd.DefaultEPPFlags(),
				PodLabels: map[string]string{"app": "vllm", "prefill:tier": "fast", "decode:tier": "slow"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := scenarioRouterInputs(tt.sh)
			if err != nil {
				t.Fatalf("scenarioRouterInputs: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got %+v\nwant %+v", got, tt.want)
			}
		})
	}
}

// The pod labels a scenario's pools state are what the router sees on each pod: a P/D scenario's
// prefill and decode pods carry their own pool's labels and the role label, never the other's.
func TestScenarioRouterInputs_PoolLabelsReachEachPod(t *testing.T) {
	in, err := scenarioRouterInputs(kernelmodel.Shape{
		Roles:  []deployment.Role{deployment.RolePrefill, deployment.RoleDecode},
		Labels: []map[string]string{{"app": "vllm", "tier": "fast"}, {"app": "vllm", "tier": "slow"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	labels := podLabelFunc(in.PodLabels)
	if got, want := labels("instance_0", 1<<0), map[string]string{"app": "vllm", "tier": "fast", "llm-d.ai/role": "prefill"}; !reflect.DeepEqual(got, want) {
		t.Errorf("prefill pod: got %v, want %v", got, want)
	}
	if got, want := labels("instance_1", 1<<1), map[string]string{"app": "vllm", "tier": "slow", "llm-d.ai/role": "decode"}; !reflect.DeepEqual(got, want) {
		t.Errorf("decode pod: got %v, want %v", got, want)
	}
}

// Kernel-backed scenarios have placement even without a NodePool. Their synthetic
// Pods must expose the distinct hosts the kernel uses to price a P/D handoff.
func TestScenarioRouterInputs_KernelPlacementReachesPodsWithoutNodePools(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	repos := kernelmodel.Repos{Scenarios: pdScenarios, Catalog: catalog, Registry: registry}
	const scenario = "glm-5-h200-3p1d-ib.yaml"
	prefill, err := kernelmodel.OpenRole(scenario, repos, deployment.RolePrefill)
	if err != nil {
		t.Fatal(err)
	}
	decode, err := kernelmodel.OpenRole(scenario, repos, deployment.RoleDecode)
	if err != nil {
		t.Fatal(err)
	}
	got, err := scenarioEndpointTopology(prefill, &kernelPools{prefill: prefill, decode: decode}, 4, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	for n := 0; n < 4; n++ {
		id := cluster.InstanceID(fmt.Sprintf("instance_%d", n))
		wantHost := fmt.Sprintf("blis-node-%d", n)
		if got[id].Hostname != wantHost || got[id].NodeCount != 1 || got[id].Rack != "" || got[id].Zone != "" || got[id].Region != "" {
			t.Errorf("%s topology = %+v, want only the kernel's %s host", id, got[id], wantHost)
		}
	}
}

// The declared GB200 rack boundary places decode rank 0 with both prefiller
// ranks and decode rank 1 across the boundary. The real EPP must choose the
// in-rack decoder; otherwise the scenario's transfer price and routing see
// different locations.
func TestScenarioRouter_KernelRackTopologyAffectsDecode(t *testing.T) {
	const epp = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: prefill-filter
- type: decode-filter
- type: max-score-picker
- type: topology-extractor
- type: topology-affinity-filter
  parameters:
    minAffinity: rack
- type: disagg-profile-handler
  parameters:
    stageOrder: prefill-first
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: max-score-picker
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: topology-affinity-filter
  - pluginRef: max-score-picker
`
	config := writeRouterFile(t, "topology.yaml", epp)
	zoneConfig := writeRouterFile(t, "zone-affinity.yaml", strings.Replace(epp, "minAffinity: rack", "minAffinity: zone", 1))
	withoutAffinity := strings.Replace(epp, "  - pluginRef: topology-affinity-filter\n", "", 1)
	plainConfig := writeRouterFile(t, "no-affinity.yaml", withoutAffinity)
	run := func(rackGPUs, trace, configFile string) (string, []string) {
		t.Helper()
		edits := []string{"router:\n  endpoint_picker_config: router/pd-always.yaml\n", ""}
		if rackGPUs == "none" {
			edits = append(edits, "  gpus_per_rack: 12\n", "")
		} else if rackGPUs != "12" {
			edits = append(edits, "  gpus_per_rack: 12\n", "  gpus_per_rack: "+rackGPUs+"\n")
		}
		dir := routerVariant(t, filepath.Join(pdScenarios, pdDPScenario),
			"router:\n  epp_flags:\n  - --endpoint-selector=llm-d.ai/inference-serving=true\n  - --endpoint-target-ports=8000\n  - --allow-experimental-plugins\n  endpoint_picker_config: topology.yaml\n",
			map[string]string{"topology.yaml": configFile}, edits...)
		common := []string{"--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "2",
			"--prefill-instances", "1", "--decode-instances", "1", "--seed", "7", "--horizon", "600000000"}
		args := append([]string{"run", "--num-requests", "8", "--rate", "4"}, common...)
		if trace != "" {
			args = append(args, "--trace-output", trace)
		}
		out, stderr, err := runKernelCLI(t, args...)
		if err != nil {
			t.Fatalf("topology-aware run: %v\n%s", err, lastLines(stderr, 12))
		}
		return out, common
	}
	trace := filepath.Join(t.TempDir(), "trace")
	boundary, common := run("12", trace, config)
	allSameRack, _ := run("16", "", config)
	undeclaredRack, _ := run("none", "", config)
	zoneFloor, _ := run("12", "", zoneConfig)
	withoutFilter, _ := run("12", "", plainConfig)
	for _, tc := range []struct {
		name, output     string
		decode2, decode3 int
	}{
		{"one in-rack decoder", boundary, 8, 0},
		{"both decoders in-rack", allSameRack, 4, 4},
		{"no rack declared: real filter fails open", undeclaredRack, 4, 4},
		{"zone floor uses tighter rack match", zoneFloor, 8, 0},
		{"declared rack but no affinity filter", withoutFilter, 4, 4},
	} {
		got2 := instanceMetricInt(t, tc.output, "instance_2", "completed_requests")
		got3 := instanceMetricInt(t, tc.output, "instance_3", "completed_requests")
		if got2 != tc.decode2 || got3 != tc.decode3 {
			t.Errorf("%s: decode counts 2=%d 3=%d, want 2=%d 3=%d", tc.name, got2, got3, tc.decode2, tc.decode3)
		}
	}
	if near, far := transferMeanUs(t, boundary), transferMeanUs(t, withoutFilter); near >= far {
		t.Errorf("in-rack routing mean KV transfer %.1f us, want less than mixed-rack %.1f us", near, far)
	}
	replayed, stderr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", trace + ".yaml", "--trace-data", trace + ".csv"}, common...)...)
	if err != nil {
		t.Fatalf("replay with kernel topology: %v\n%s", err, lastLines(stderr, 8))
	}
	if replayed != boundary {
		t.Errorf("kernel topology changed the run/replay output")
	}
}

func transferMeanUs(t *testing.T, stdout string) float64 {
	t.Helper()
	const label = "KV Transfer Duration (us): mean="
	at := strings.Index(stdout, label)
	if at < 0 {
		t.Fatal("no KV-transfer duration in output")
	}
	var mean float64
	if _, err := fmt.Sscanf(stdout[at+len(label):], "%f", &mean); err != nil {
		t.Fatal(err)
	}
	return mean
}

// Keep a copy of the pinned router fork's stock P/D topology config in
// cmd/testdata/epp. Its decode-first filter and scorer must actually start
// against scenario-derived Pod labels, even though this 3P1D fixture has no
// same-host choice and is not a performance test.
func TestScenarioRouter_UpstreamTopologySampleStarts(t *testing.T) {
	dir := routerVariant(t, filepath.Join(pdScenarios, "glm-5-h200-3p1d-ib.yaml"),
		"router:\n  epp_flags:\n  - --endpoint-selector=llm-d.ai/inference-serving=true\n  - --endpoint-target-ports=8000\n  - --allow-experimental-plugins\n  endpoint_picker_config: pd-topology.yaml\n",
		map[string]string{"pd-topology.yaml": "testdata/epp/pd-topology-upstream.yaml"},
		"router:\n  endpoint_picker_config: router/pd-always.yaml\n", "")
	out, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
		"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1",
		"--num-requests", "8", "--rate", "4", "--seed", "7")
	if err != nil {
		t.Fatalf("upstream topology config: %v\n%s", err, lastLines(stderr, 12))
	}
	if got := clusterMetricInt(t, out, "completed_requests"); got != 8 {
		t.Errorf("completed %d of 8 virtual requests", got)
	}
}

func instanceMetricInt(t *testing.T, stdout, id, field string) int {
	t.Helper()
	for _, raw := range extractJSONObjects(stdout) {
		var obj map[string]any
		if json.Unmarshal([]byte(raw), &obj) != nil || obj["instance_id"] != id {
			continue
		}
		value, ok := obj[field].(float64)
		if !ok {
			t.Fatalf("%s metrics have no numeric %s", id, field)
		}
		return int(value)
	}
	t.Fatalf("no %s metrics object in output", id)
	return 0
}

// A router block naming a file that does not exist, or naming the configuration twice, is an
// error naming the field, before any simulation state is built.
func TestScenarioRouterInputs_Refusals(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "present.yaml"), []byte("kind: InferencePool\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name, want string
		r          deployment.Router
	}{
		{"a missing endpoint_picker_config", "router.endpoint_picker_config", deployment.Router{EndpointPickerConfig: "absent.yaml"}},
		{"a missing manifest", "router.manifests[1]", deployment.Router{Manifests: []string{"present.yaml", "absent.yaml"}}},
		{"a missing --config-file", "--config-file", deployment.Router{EPPFlags: []string{"--config-file=absent.yaml"}}},
		{"a missing --config-file, two-argument form", "--config-file", deployment.Router{EPPFlags: []string{"--config-file", "absent.yaml"}}},
		{"the configuration named twice", "both name the router configuration",
			deployment.Router{EPPFlags: []string{"--config-text=x"}, EndpointPickerConfig: "absent.yaml"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := tt.r
			_, err := scenarioRouterInputs(kernelmodel.Shape{Roles: []deployment.Role{deployment.RoleColocated},
				Labels: []map[string]string{nil}, Dir: dir, Router: &r})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("got %v, want an error naming %q", err, tt.want)
			}
		})
	}
}

// The CLI refuses a scenario whose router block names a missing file, naming the scenario and the
// field (logrus.Fatalf, exit 1), on run and on replay alike.
func TestRunAndReplay_ScenarioRouterBadPathFatal(t *testing.T) {
	dir := colocatedRouterVariant(t, "router:\n  endpoint_picker_config: absent.yaml\n", "", nil)
	trace := filepath.Join(t.TempDir(), "trace")
	if _, stderr, err := runKernelCLI(t, "run", "--scenarios", pdScenarios, "--scenario", capScenario, "--num-instances", "2",
		"--num-requests", "4", "--rate", "2", "--trace-output", trace); err != nil {
		t.Fatalf("source run: %v\n%s", err, lastLines(stderr, 5))
	}
	for _, args := range [][]string{
		{"run", "--num-requests", "4", "--rate", "2"},
		{"replay", "--trace-header", trace + ".yaml", "--trace-data", trace + ".csv"},
	} {
		_, stderr, err := runKernelCLI(t, append(args, "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "2")...)
		if err == nil {
			t.Errorf("%s accepted a router block naming a missing file", args[0])
			continue
		}
		if !strings.Contains(stderr, "router.endpoint_picker_config") || !strings.Contains(stderr, "variant.yaml") {
			t.Errorf("%s: the refusal must name the scenario and the field; stderr:\n%s", args[0], lastLines(stderr, 3))
		}
	}
}

// A replay given a scenario whose router differs from the one the trace was recorded with is
// refused, like a contradicting flag (INV-13); one stating the recorded router, or none, replays
// the recorded router byte for byte.
func TestReplay_ScenarioRouterContradictingTheHeaderFatal(t *testing.T) {
	loadOnly := map[string]string{"load-only.yaml": "../examples/llm-d/load-only.yaml"}
	recorded := colocatedRouterVariant(t, "router:\n  endpoint_picker_config: load-only.yaml\n", "      app: vllm\n", loadOnly)
	prefix := filepath.Join(t.TempDir(), "trace")
	common := []string{"--scenario", "variant.yaml", "--num-instances", "2", "--horizon", "600000000"}
	runOut, stderr, err := runKernelCLI(t, append([]string{"run", "--scenarios", recorded, "--num-requests", "20", "--rate", "10",
		"--seed", "3", "--trace-output", prefix}, common...)...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	replay := func(dir string) (string, string, error) {
		return runKernelCLI(t, append([]string{"replay", "--scenarios", dir,
			"--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv"}, common...)...)
	}
	for _, tt := range []struct{ name, dir string }{
		{"the recorded router restated", recorded},
		{"no router stated", colocatedRouterVariant(t, "", "", nil)},
	} {
		out, stderr, err := replay(tt.dir)
		if err != nil {
			t.Errorf("%s: replay refused: %v\n%s", tt.name, err, lastLines(stderr, 3))
		} else if out != runOut {
			t.Errorf("%s: INV-13: the replay differs from the run", tt.name)
		}
	}
	for _, tt := range []struct{ name, dir string }{
		{"another configuration", colocatedRouterVariant(t, "router:\n  endpoint_picker_config: prefix-affinity.yaml\n", "      app: vllm\n",
			map[string]string{"prefix-affinity.yaml": "../examples/llm-d/prefix-affinity.yaml"})},
		{"other pod labels", colocatedRouterVariant(t, "router:\n  endpoint_picker_config: load-only.yaml\n", "      app: other\n", loadOnly)},
		{"other EPP flags", colocatedRouterVariant(t, "router:\n  endpoint_picker_config: load-only.yaml\n  epp_flags: [--endpoint-selector=app=vllm]\n",
			"      app: vllm\n", loadOnly)},
	} {
		_, stderr, err := replay(tt.dir)
		if err == nil {
			t.Errorf("%s: a replay whose scenario contradicts the recorded router was accepted", tt.name)
			continue
		}
		if !strings.Contains(stderr, "differs from the router the trace was recorded with") {
			t.Errorf("%s: the refusal must name the contradiction; stderr:\n%s", tt.name, lastLines(stderr, 3))
		}
	}
}

// Pool labels alone (no router block) reach the router: an InferencePool selecting app=vllm, named
// by a router block, serves a pool whose labels say app=vllm, and the same scenario with the label
// stating another value leaves the pool with no endpoints, so nothing completes. The labels-only
// case is checked from the trace header the run records.
func TestRun_PoolLabelsWithoutARouterBlockReachTheRouter(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "trace")
	labelsOnly := colocatedRouterVariant(t, "", "      app: vllm\n      tier: gpu\n", nil)
	if _, stderr, err := runKernelCLI(t, "run", "--scenarios", labelsOnly, "--scenario", "variant.yaml", "--num-instances", "2",
		"--num-requests", "6", "--rate", "3", "--trace-output", prefix); err != nil {
		t.Fatalf("labels-only run: %v\n%s", err, lastLines(stderr, 5))
	}
	header := readFile(t, prefix+".yaml")
	for _, want := range []string{"app: vllm", "tier: gpu", "--endpoint-selector=llm-d.ai/inference-serving=true"} {
		if !strings.Contains(header, want) {
			t.Errorf("a labels-only scenario's run does not record %q in its router header", want)
		}
	}
	pool := writeRouterFile(t, "pool.yaml", inv13Manifests)
	served := func(label string) int {
		dir := colocatedRouterVariant(t, "router:\n  epp_flags: [--pool-name=vllm]\n  manifests: [pool.yaml]\n",
			"      app: "+label+"\n", map[string]string{"pool.yaml": pool})
		out, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "2",
			"--num-requests", "6", "--rate", "3", "--horizon", "60000000")
		if err != nil {
			t.Fatalf("app=%s: %v\n%s", label, err, lastLines(stderr, 5))
		}
		return clusterMetricInt(t, out, "completed_requests")
	}
	if got := served("vllm"); got != 6 {
		t.Errorf("pods labelled app=vllm, the pool's selector, completed %d of 6 requests", got)
	}
	if got := served("other"); got != 0 {
		t.Errorf("pods the pool's selector does not match completed %d requests; the pool labels are not reaching the router", got)
	}
}

// The router inputs are the scenario's: the four flags that carried them are not registered on
// run or replay, as --tp is not, and are refused as unknown.
func TestRunAndReplay_RouterDeploymentFlagsRemoved(t *testing.T) {
	for _, name := range []string{"epp-flags", "epp-config", "epp-manifests", "pod-labels"} {
		for _, command := range []string{"run", "replay"} {
			_, stderr, err := runKernelCLI(t, command, "--scenario", capScenario, "--scenarios", pdScenarios, "--"+name, "x")
			if err == nil || !strings.Contains(stderr, "unknown flag: --"+name) {
				t.Errorf("%s --%s: want an unknown-flag refusal, got err=%v\n%s", command, name, err, lastLines(stderr, 3))
			}
		}
	}
}

// Every example scenario under examples/llm-d/scenarios loads, reads its router block and runs;
// the P/D one disaggregates every request.
func TestExampleRouterScenariosRun(t *testing.T) {
	const dir = "../examples/llm-d/scenarios"
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".yaml") {
			continue
		}
		n++
		t.Run(e.Name(), func(t *testing.T) {
			t.Parallel()
			args := []string{"run", "--scenarios", dir, "--scenario", e.Name(), "--num-requests", "8", "--rate", "4", "--num-instances", "2"}
			pd := strings.Contains(e.Name(), "3p1d")
			if pd {
				args = append(args, "--prefill-instances", "1", "--decode-instances", "1")
			}
			out, stderr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
			}
			if pd && !strings.Contains(out, "Disaggregated Requests: 8") {
				t.Errorf("the P/D example did not disaggregate every request:\n%s", lastLines(out, 12))
			}
		})
	}
	if n < 5 {
		t.Fatalf("non-vacuity: found %d example scenarios", n)
	}
}

// Explicit simulated Node manifests give each kernel-placed Pod geography
// without guessing it from network bandwidth. The same declared topology is
// read by both run and replay, and the real EPP chooses the in-zone decoder.
func TestScenarioRouter_DeclaredZoneRunReplayParity(t *testing.T) {
	const epp = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: prefill-filter
- type: decode-filter
- type: max-score-picker
- type: topology-extractor
- type: topology-affinity-filter
  parameters:
    minAffinity: zone
- type: disagg-profile-handler
  parameters:
    stageOrder: prefill-first
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: max-score-picker
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: topology-affinity-filter
  - pluginRef: max-score-picker
`
	config := writeRouterFile(t, "zone-epp.yaml", epp)
	const nodes = `apiVersion: v1
kind: Node
metadata:
  name: blis-node-0
  labels:
    topology.kubernetes.io/zone: zone-a
    topology.kubernetes.io/region: region-1
---
apiVersion: v1
kind: Node
metadata:
  name: blis-node-1
  labels:
    topology.kubernetes.io/zone: zone-a
    topology.kubernetes.io/region: region-1
---
apiVersion: v1
kind: Node
metadata:
  name: blis-node-2
  labels:
    topology.kubernetes.io/zone: zone-a
    topology.kubernetes.io/region: region-1
---
apiVersion: v1
kind: Node
metadata:
  name: blis-node-3
  labels:
    topology.kubernetes.io/zone: zone-b
    topology.kubernetes.io/region: region-1
`
	nodeFile := writeRouterFile(t, "nodes.yaml", nodes)
	dir := routerVariant(t, filepath.Join(pdScenarios, pdDPScenario),
		"router:\n  epp_flags:\n  - --endpoint-selector=llm-d.ai/inference-serving=true\n  - --endpoint-target-ports=8000\n  - --allow-experimental-plugins\n  endpoint_picker_config: zone-epp.yaml\n  manifests:\n  - nodes.yaml\n",
		map[string]string{"zone-epp.yaml": config, "nodes.yaml": nodeFile},
		"router:\n  endpoint_picker_config: router/pd-always.yaml\n", "",
		"  gpus_per_rack: 12\n", "  gpus_per_rack: 4\n")
	common := []string{"--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "2",
		"--prefill-instances", "1", "--decode-instances", "1", "--seed", "7", "--horizon", "600000000"}
	trace := filepath.Join(t.TempDir(), "trace")
	out, stderr, err := runKernelCLI(t, append([]string{"run", "--num-requests", "8", "--rate", "4", "--trace-output", trace}, common...)...)
	if err != nil {
		t.Fatalf("zone topology run: %v\n%s", err, lastLines(stderr, 12))
	}
	if got := instanceMetricInt(t, out, "instance_2", "completed_requests"); got != 8 {
		t.Fatalf("in-zone decoder completed %d of 8 requests", got)
	}
	if got := instanceMetricInt(t, out, "instance_3", "completed_requests"); got != 0 {
		t.Fatalf("out-of-zone decoder completed %d of 8 requests", got)
	}
	replayed, stderr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", trace + ".yaml", "--trace-data", trace + ".csv"}, common...)...)
	if err != nil {
		t.Fatalf("zone topology replay: %v\n%s", err, lastLines(stderr, 12))
	}
	if replayed != out {
		t.Error("declared zone topology changed run/replay output")
	}
}
