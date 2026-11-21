package llmd

import (
	"fmt"
	"strings"
	"testing"

	"sigs.k8s.io/controller-runtime/pkg/metrics"
)

// renderFailures is the router's token-producer render-failure count (every backend, plugin and
// reason), read from the registry the router writes it to. It accumulates over the process, so a
// test reads the difference over its run.
func renderFailures(t *testing.T) float64 {
	t.Helper()
	families, err := metrics.Registry.Gather()
	if err != nil {
		t.Fatal(err)
	}
	var n float64
	for _, f := range families {
		if strings.HasSuffix(f.GetName(), "token_producer_render_failures_total") {
			for _, m := range f.GetMetric() {
				n += m.GetCounter().GetValue()
			}
		}
	}
	return n
}

// TestRenderCalls_AnswerAsTheEngineStatesEnableScaleOut: vLLM answers the router's render calls
// only when the server runs with --enable-scale-out; otherwise each is HTTP 403
// (register_scale_out_api_routers, vllm/entrypoints/scale_out/factories.py). With the setting
// unset a router config with a render token-producer gets no token IDs: the router counts render
// failures and routes as a config without the producer does. With it true the same config
// routes on prefix (inference-sim#2129).
func TestRenderCalls_AnswerAsTheEngineStatesEnableScaleOut(t *testing.T) {
	loadOnly := `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: max-score-picker
`
	run := func(stated, config string) ([]string, float64) {
		dep := testDeployment(4)
		dep.NewEngine = compatEngineFactoryStating(false, pdTicks, stated)
		before := renderFailures(t)
		r := runPrecise(t, dep, prefixGroups(3, 15, 256, 30000), config, Options{})
		var at []string
		for _, q := range r.reqs {
			at = append(at, q.ID+"@"+q.AssignedInstance)
		}
		return at, renderFailures(t) - before
	}
	groupsKept := func(at []string) bool {
		where := map[string]map[string]bool{}
		for _, a := range at {
			id, inst, _ := strings.Cut(a, "@")
			g := id[:strings.Index(id, "_")]
			if where[g] == nil {
				where[g] = map[string]bool{}
			}
			where[g][inst] = true
		}
		for _, w := range where {
			if len(w) != 1 {
				return false
			}
		}
		return true
	}

	unset, failed := run("", preciseConfig(""))
	if failed == 0 {
		t.Errorf("enable_scale_out unset: the router counted no render failures; vLLM answers 403")
	}
	if groupsKept(unset) {
		t.Errorf("enable_scale_out unset: prefix groups stayed together; the router has no token IDs and ranks on load alone")
	}
	if base, _ := run("", loadOnly); fmt.Sprint(base) != fmt.Sprint(unset) {
		t.Errorf("enable_scale_out unset: endpoint choices differ from a load-only config\n got  %v\n want %v", unset, base)
	}
	if at, failed := run("enable_scale_out: false\n", preciseConfig("")); failed == 0 || fmt.Sprint(at) != fmt.Sprint(unset) {
		t.Errorf("enable_scale_out false behaves unlike unset (render failures %v)", failed)
	}

	set, failed := run("enable_scale_out: true\n", preciseConfig(""))
	if failed != 0 {
		t.Errorf("enable_scale_out true: %v render failures, want none", failed)
	}
	if !groupsKept(set) {
		t.Errorf("enable_scale_out true: prefix groups spread; the router routes on prefix: %v", set)
	}
}
