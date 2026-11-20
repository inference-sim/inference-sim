package deployment

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// An offload's connector options are the engine's, held unopened like a pool's engine block:
// they load as written, misspelled keys included, and are written back unchanged. The
// engine model checks them (rules.EngineRules.CheckConnectorOptions).
func TestOffloadConnectorOptionsAreHeldUnopened(t *testing.T) {
	const doc = "tiers:\n    - tier: cpu-dram\n      bytes: 1\nconnector_options:\n    load_asynch: true\n    blocks_per_chunk: -1\n"
	var o Offload
	dec := yaml.NewDecoder(strings.NewReader(doc))
	dec.KnownFields(true)
	if err := dec.Decode(&o); err != nil {
		t.Fatalf("the options are the engine's to check, but decoding refused them: %v", err)
	}
	if o.ConnectorOptions.IsZero() {
		t.Fatal("stated options decoded as absent")
	}
	out, err := yaml.Marshal(o)
	if err != nil || string(out) != doc {
		t.Errorf("the options did not round-trip:\n%s\nwant\n%s (%v)", out, doc, err)
	}
	d := &Deployment{Kind: "Deployment", Name: "d", Pools: []Pool{{
		Role: RoleColocated, Nodes: 1, Parallel: Parallelism{TP: 1, PP: 1, DP: 1}}}, Offload: &o}
	if p := d.Validate(); !p.OK() {
		t.Errorf("the deployment checked the engine's options itself:\n%s", p.Error())
	}
	var absent Offload
	if err := yaml.Unmarshal([]byte("tiers: [{tier: cpu-dram, bytes: 1}]\n"), &absent); err != nil || !absent.ConnectorOptions.IsZero() {
		t.Errorf("absent options: %+v, %v", absent.ConnectorOptions, err)
	}
}
