package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// vLLM answers the router's render calls only when the server runs with --enable-scale-out
// (engine.enable_scale_out); otherwise each is refused with 403 and the router's token producer
// goes on without token IDs, so a prefix-aware configuration routes on load alone
// (inference-sim#2129). The engine block's flag decides it end to end: the engine states it
// (blis-vllm), the simulated model server answers accordingly (sim/llmd).
func TestRender_EngineScaleOutFlagDecidesWhetherThePrefixRouterSeesTokens(t *testing.T) {
	horizon := "9223372036854775807"
	run := func(router, engineLine string) map[string]any {
		var edits []string
		if engineLine != "" {
			edits = []string{"    engine:\n", "    engine:\n" + engineLine}
		}
		dir := routerVariant(t, filepath.Join(pdScenarios, capScenario), "router:\n  endpoint_picker_config: epp.yaml\n",
			map[string]string{"epp.yaml": router}, edits...)
		out, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml", "--num-instances", "4",
			"--workload-spec", "../examples/prefix-affinity-demo.yaml", "--seed", "1", "--horizon", horizon,
			"--kv-event-delay", "3000")
		if err != nil {
			t.Fatalf("run: %v\n%s", err, lastLines(stderr, 8))
		}
		// The measured figures: what routing changes. The recorded router requirements differ
		// between configurations that name different plugins, whatever they route.
		var result struct {
			Metrics json.RawMessage `json:"metrics"`
		}
		if err := json.Unmarshal([]byte(out), &result); err != nil || len(result.Metrics) == 0 {
			t.Fatalf("result has no metrics: %v", err)
		}
		var m map[string]any
		if err := json.Unmarshal(result.Metrics, &m); err != nil {
			t.Fatal(err)
		}
		return m
	}
	unset := run("testdata/epp/precise.yaml", "")
	// The same picker and profile handler with no scorer: what the router does when it has no
	// token IDs to match.
	noScorer := filepath.Join(t.TempDir(), "no-scorer.yaml")
	if err := os.WriteFile(noScorer, []byte("apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n- type: max-score-picker\n"+
		"- type: single-profile-handler\nschedulingProfiles:\n- name: default\n  plugins:\n  - pluginRef: max-score-picker\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	loadOnly := run(noScorer, "")
	served := run("testdata/epp/precise.yaml", "      enable_scale_out: true\n")
	if !reflect.DeepEqual(unset, loadOnly) {
		t.Errorf("enable_scale_out unset: the token-producer router routes differently from one with no scorer, though every render call is refused with 403")
	}
	if reflect.DeepEqual(served, unset) {
		t.Errorf("enable_scale_out true: the router routes exactly as with every render call refused; it never saw token IDs")
	}
}
