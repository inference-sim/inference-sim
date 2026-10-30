package vllmengine

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
)

// DeepSeek V4's 0.32.0 deployments keep their packed full-attention group at the engine's block
// size (V4 Pro at 256 beside 64-token windows, V4.1 Flash at 128 beside 32-token windows), so a
// router keying blocks at the engine's block size indexes their KV events, as it does Muse
// Spark's (core/vllm Engine.KVEventsRefusal).
func TestDashboardDeploymentsKVEventsAreIndexable(t *testing.T) {
	Register()
	for _, c := range []struct {
		scenario string
		refused  bool
	}{
		{"d000-032.yaml", false},
		{"d027-032.yaml", false},
		{"muse-h100-roce-armC.yaml", false},
	} {
		eng, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, c.scenario)})
		if err != nil {
			t.Fatalf("%s: %v", c.scenario, err)
		}
		r, ok := eng.(interface{ KVEventsRefusal() error })
		if !ok {
			t.Fatalf("%s: the engine %T states no KVEventsRefusal", c.scenario, eng)
		}
		err = r.KVEventsRefusal()
		if c.refused != (err != nil) || (err != nil && !strings.Contains(err.Error(), "block size")) {
			t.Errorf("%s: KVEventsRefusal() = %v, want refused %t", c.scenario, err, c.refused)
		}
	}
}
