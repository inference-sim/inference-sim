package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// p2pFetchScenario is the GLM-5 P/D fixture, written to a fresh directory, with its NIXL handoff
// replaced by vLLM's p2p KV tier: the deployment's offload block states a CPU tier and, in its
// connector options, a p2p tier, as vLLM states it (kv_connector_extra_config.secondary_tiers), so
// both pools build engines with the tier. The fixture itself is only read.
func p2pFetchScenario(t *testing.T) (dir, name string) {
	t.Helper()
	name = "glm-5-h200-3p1d-ib.yaml"
	raw, err := os.ReadFile(filepath.Join(pdScenarios, name))
	if err != nil {
		t.Fatal(err)
	}
	const nixl = "pd_transfer:\n  connector: NixlConnector\n"
	s := string(raw)
	if !strings.Contains(s, nixl) {
		t.Fatal("the fixture's pd_transfer block changed")
	}
	s = strings.Replace(s, nixl, "pd_transfer:\n  connector: OffloadingConnector\noffload:\n  connector: OffloadingConnector\n  spec: TieringOffloadingSpec\n"+
		"  tiers:\n    - {tier: cpu_dram, bytes: 549755813888}\n"+
		"  connector_options:\n    secondary_tiers: [{type: p2p, port: 5710}]\n", 1)
	if !strings.Contains(s, "cost-model-offload-tiers") {
		s = strings.Replace(s, "coefficients: [", "coefficients: [cost-model-offload-tiers, ", 1)
	}
	dir = withPDRouter(t, t.TempDir())
	if err := os.WriteFile(filepath.Join(dir, name), []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir, name
}

// figure finds a named figure anywhere in a Result document.
func figure(v any, name string) (float64, bool) {
	switch x := v.(type) {
	case map[string]any:
		if f, ok := x[name].(float64); ok {
			return f, true
		}
		for _, e := range x {
			if f, ok := figure(e, name); ok {
				return f, ok
			}
		}
	case []any:
		for _, e := range x {
			if f, ok := figure(e, name); ok {
				return f, ok
			}
		}
	}
	return 0, false
}

// keysWithPrefix lists the figure names anywhere in a Result document that start with prefix.
func keysWithPrefix(v any, prefix string, out map[string]bool) {
	switch x := v.(type) {
	case map[string]any:
		for k, e := range x {
			if strings.HasPrefix(k, prefix) {
				out[k] = true
			}
			keysWithPrefix(e, prefix, out)
		}
	case []any:
		for _, e := range x {
			keysWithPrefix(e, prefix, out)
		}
	}
}

// A P/D run of real blis-vllm engines hands its requests off through vLLM's p2p KV tier
// (blis-vllm#204): the decode engine loads each prompt from the prefill instance's CPU tier, whose
// WRITEs cross the fabric from the prefill node to the decode node. Every request completes, the
// engines' control messages are carried and none is dropped, and the WRITEs load the fabric's links
// between the two nodes.
func TestP2PFetch_APDRunHandsItsRequestsOffThroughTheP2PTier(t *testing.T) {
	dir, name := p2pFetchScenario(t)
	for _, top := range []struct{ prefill, decode int }{{1, 1}, {2, 1}, {3, 1}} {
		t.Run(fmt.Sprintf("%dP%dD", top.prefill, top.decode), func(t *testing.T) {
			out := filepath.Join(t.TempDir(), "result.json")
			_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", name,
				"--num-instances", fmt.Sprint(top.prefill+top.decode), "--prefill-instances", fmt.Sprint(top.prefill),
				"--decode-instances", fmt.Sprint(top.decode), "--num-requests", "6", "--rate", "2", "--seed", "1", "--output", out)
			if err != nil {
				t.Fatalf("the p2p P/D run: %v\n%s", err, lastLines(stderr, 8))
			}
			var doc any
			if err := json.Unmarshal([]byte(readFile(t, out)), &doc); err != nil {
				t.Fatal(err)
			}
			sent, ok := figure(doc, "peer_messages_sent")
			if !ok || sent == 0 {
				t.Fatalf("the Result states no peer messages (%v, %v): the engines did not talk", sent, ok)
			}
			delivered, _ := figure(doc, "peer_messages_delivered")
			dropped, _ := figure(doc, "peer_messages_dropped")
			if delivered != sent || dropped != 0 {
				t.Errorf("peer messages: %v sent, %v delivered, %v dropped; want every one delivered", sent, delivered, dropped)
			}
			links := map[string]bool{}
			keysWithPrefix(doc, "link_peak_concurrent_transfers_", links)
			crossed := false
			for k := range links {
				crossed = crossed || strings.Contains(k, "nic")
			}
			t.Logf("peer messages %v sent, %v delivered, %v dropped; links %v", sent, delivered, dropped, links)
			if !crossed {
				t.Errorf("no WRITE crossed a NIC between the two nodes; links used: %v", links)
			}
			if n, _ := figure(doc, "disaggregated_requests"); n != 6 {
				t.Errorf("%v disaggregated requests, want all 6", n)
			}
			if n, _ := figure(doc, "dropped_at_decode_kv"); n != 0 {
				t.Errorf("%v requests dropped at the decode's KV", n)
			}
		})
	}
}
