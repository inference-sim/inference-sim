// Original to blis-vllm.

package build

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/engine"
	_ "github.com/inference-sim/blis-vllm/compat/lora" // the LoRA cost model and registry, as the simulator links them
)

func f64(v float64) *float64 { return &v }

// The LoRA HBM reservation is 0 for an inert config, and capacity x max rank x footprint per
// rank once adapters are declared; a cost config the model refuses is an error naming the
// reservation.
func TestAdapterReservedBytes(t *testing.T) {
	if got, err := AdapterReservedBytes(engine.LoRAConfig{}); err != nil || got != 0 {
		t.Errorf("inert config: %d, %v; want 0", got, err)
	}
	capacity := 4
	cfg := engine.LoRAConfig{
		AdapterCapacity: &capacity, LoadBaseLatencyUs: f64(1000), LoadBandwidthBytesUs: f64(2e6),
		FootprintBytesPerRank: f64(2e6),
		StepOverheadTiers:     map[int]engine.StepOverheadTier{8: {K6: f64(0.02), K7: f64(1)}},
		Adapters:              []engine.AdapterSpec{{ID: "a8", Rank: 8}, {ID: "a16", Rank: 16}, {ID: "a32", Rank: 32}},
	}
	if got, err := AdapterReservedBytes(cfg); err != nil || got != int64(4*32*2e6) {
		t.Errorf("populated config: %d, %v; want capacity x max rank x footprint, %d", got, err, int64(4*32*2e6))
	}
	bad := cfg
	bad.LoadBandwidthBytesUs = f64(0)
	if _, err := AdapterReservedBytes(bad); err == nil || !strings.HasPrefix(err.Error(), "Invalid LoRA configuration (HBM reservation): ") {
		t.Errorf("a cost config the model refuses: %v", err)
	}
}

// A config with no adapters has no registry; declared adapters get one that knows them; a
// registry the constructor refuses (two adapters with one id) is an error with the CLI's text.
func TestAdapterRegistryOf(t *testing.T) {
	if r, err := AdapterRegistryOf(engine.LoRAConfig{}); r != nil || err != nil {
		t.Errorf("no adapters: %v, %v; want no registry", r, err)
	}
	r, err := AdapterRegistryOf(engine.LoRAConfig{Adapters: []engine.AdapterSpec{{ID: "a0", Rank: 8}}})
	if err != nil || r == nil {
		t.Fatalf("one adapter: %v, %v", r, err)
	}
	if _, err := AdapterRegistryOf(engine.LoRAConfig{Adapters: []engine.AdapterSpec{{ID: "a0", Rank: 8}, {ID: "a0", Rank: 16}}}); err == nil ||
		!strings.HasPrefix(err.Error(), "Invalid LoRA adapter registry: ") {
		t.Errorf("a duplicate id: %v, want the registry refusal", err)
	}
}

// The speculative configuration is the engine constructor's: K, method and rate as given, and a
// rate stated with nothing drafted refused.
func TestSpeculativeConfigOf(t *testing.T) {
	c, err := SpeculativeConfigOf(3, 0.5, "mtp")
	if err != nil || c.K != 3 || c.Method != "mtp" || c.Acceptance != 0.5 {
		t.Errorf("K 3: %+v, %v", c, err)
	}
	if _, err := SpeculativeConfigOf(0, 0.5, ""); err == nil {
		t.Error("a rate with nothing drafted must be refused")
	}
}
