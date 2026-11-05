package scenario

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/trace"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// multiNode is the immutable problem behind a prefill/decode cluster: a model on 60
// H200 nodes behind a 400G fabric. The mutable layout that runs on it is a Deployment,
// which this package no longer carries.
func multiNode() *Scenario {
	return &Scenario{
		Kind: "Scenario", Name: "glm-5.3-60node-h200",
		Model: "glm-5.3", Coefficients: []string{"cost-model-primitives-h200"},
		EngineVersion: "0.29.0",
		Cluster: Cluster{Hardware: "h200", Fabric: "ib-400g", Nodes: 60,
			GPUsPerNode: 8, Storage: []string{"cpu_dram", "nvme_gen4"}},
	}
}

// singleNode is the colocated problem the published Granite runs pose: eight GPUs on
// one node, no fabric because nothing crosses a node.
func singleNode() *Scenario {
	return &Scenario{
		Kind: "Scenario", Name: "granite-230b-h100",
		Model: "granite-5-230b", Coefficients: []string{"cost-model-primitives-h100"},
		EngineVersion: "0.29.0",
		Cluster:       Cluster{Hardware: "h100", Nodes: 1, GPUsPerNode: 8},
	}
}

func TestValidScenariosPass(t *testing.T) {
	for _, s := range []*Scenario{multiNode(), singleNode()} {
		if p := s.Validate(); !p.OK() {
			t.Fatalf("%s rejected:\n%s", s.Name, p.Error())
		}
	}
}

// TestScenarioWorkloadBinding covers the "what traffic" slot the Scenario now carries: a
// capacity-only scenario omits it, each arm is accepted, and an ill-formed binding
// surfaces through the Scenario's own validation (the exactly-one law is the binding's,
// and the Scenario must not swallow it).
func TestScenarioWorkloadBinding(t *testing.T) {
	traceArm := &workload.Binding{Trace: &workload.TraceRef{
		Data:   "traces/run.csv",
		Header: trace.Header{Version: trace.Version, TimeUnit: "microseconds", Mode: trace.ModeConverted},
	}}
	pass := map[string]*workload.Binding{
		"no workload (capacity question)": nil,
		"shape arm":                       {Shape: "chatbot"},
		"trace arm":                       traceArm,
	}
	for name, b := range pass {
		t.Run(name, func(t *testing.T) {
			s := singleNode()
			s.Workload = b
			if p := s.Validate(); !p.OK() {
				t.Fatalf("expected a valid scenario, got:\n%s", p.Error())
			}
		})
	}

	fail := map[string]*workload.Binding{
		"both arms":     {Shape: "chatbot", Trace: traceArm.Trace},
		"empty binding": {},
		"invalid trace": {Trace: &workload.TraceRef{Data: "t.csv"}}, // header missing required fields
	}
	for name, b := range fail {
		t.Run(name, func(t *testing.T) {
			s := singleNode()
			s.Workload = b
			if p := s.Validate(); p.OK() {
				t.Fatal("expected the scenario to reject an ill-formed workload binding")
			}
		})
	}
}

func TestClusterGPUs(t *testing.T) {
	if got, err := (Cluster{Hardware: "h200", Nodes: 60, GPUsPerNode: 8}).GPUs(hgx()); err != nil || got != 480 {
		t.Errorf("GPUs() = %d, %v, want 480", got, err)
	}
	if got, err := (Cluster{Hardware: "gb200-nvl72", Nodes: 18}).GPUs(nvl72()); err != nil || got != 72 {
		t.Errorf("GPUs() on NVL72 trays = %d, %v, want 72", got, err)
	}
	if _, err := (Cluster{Hardware: "gb200-nvl72", Nodes: 18, GPUsPerNode: 8}).GPUs(nvl72()); err == nil {
		t.Error("GPUs() on a cluster contradicting its chip: got no error")
	}
}

func TestRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Scenario)
	}{
		{"wrong kind", func(s *Scenario) { s.Kind = "Deployment" }},
		{"no model", func(s *Scenario) { s.Model = "" }},
		{"no hardware", func(s *Scenario) { s.Cluster.Hardware = "" }},
		{"no engine version", func(s *Scenario) { s.EngineVersion = "" }},
		{"no coefficients", func(s *Scenario) { s.Coefficients = nil }},
		{"zero nodes", func(s *Scenario) { s.Cluster.Nodes = 0 }},
		{"negative gpus per node", func(s *Scenario) { s.Cluster.GPUsPerNode = -8 }},
		{"negative gpus per rack", func(s *Scenario) { s.Cluster.GPUsPerRack = -72 }},
		{"multi-node without a fabric", func(s *Scenario) { s.Cluster.Fabric = "" }},
		{"rack not a multiple of node", func(s *Scenario) { s.Cluster.GPUsPerRack = 70 }},
		{"empty storage class", func(s *Scenario) {
			s.Cluster.Storage = []string{""}
		}},
		{"duplicate storage class", func(s *Scenario) {
			s.Cluster.Storage = []string{"cpu_dram", "cpu_dram"}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := multiNode()
			tc.mutate(s)
			if p := s.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

// The acceptance pin is optional, and a present one must be well formed.
func TestScenarioAcceptancePin(t *testing.T) {
	length := 2.8
	for _, c := range []struct {
		name string
		pin  *acceptance.Pin
		ok   bool
	}{
		{"omitted", nil, true},
		{"a named set", &acceptance.Pin{Profiles: "inferencex-golden-al"}, true},
		{"an inline value", &acceptance.Pin{Value: &acceptance.Value{NumSpecTokens: 3, AcceptanceLength: &length}}, true},
		{"empty", &acceptance.Pin{}, false},
		{"set and value", &acceptance.Pin{Profiles: "x",
			Value: &acceptance.Value{NumSpecTokens: 3, AcceptanceLength: &length}}, false},
		{"value above K+1", &acceptance.Pin{Value: &acceptance.Value{NumSpecTokens: 1, AcceptanceLength: &length}}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := singleNode()
			s.Acceptance = c.pin
			if p := s.Validate(); p.OK() != c.ok {
				t.Errorf("OK = %v, want %v:\n%s", p.OK(), c.ok, p.Error())
			}
		})
	}
}

// TestEngine covers SIM-1's scenario side: engine {kind, version} is the field, the
// deprecated engine_version is accepted alone or agreeing with it until M4, and a scenario
// naming no engine is refused.
func TestEngine(t *testing.T) {
	withEngine := func(e *Engine, version string) *Scenario {
		s := singleNode()
		s.Engine, s.EngineVersion = e, version
		return s
	}
	vllm := &Engine{Kind: "vllm", Version: "0.32.0"}
	for name, s := range map[string]*Scenario{
		"engine alone":         withEngine(vllm, ""),
		"engine_version alone": withEngine(nil, "0.32.0"),
		"both, agreeing":       withEngine(vllm, "0.32.0"),
	} {
		if p := s.Validate(); !p.OK() {
			t.Errorf("%s: rejected:\n%s", name, p.Error())
		}
	}
	if got := withEngine(vllm, "").EngineRelease(); got != "0.32.0" {
		t.Errorf("EngineRelease with engine = %q", got)
	}
	if got := withEngine(nil, "0.31.0").EngineRelease(); got != "0.31.0" {
		t.Errorf("EngineRelease with engine_version = %q", got)
	}
	for _, c := range []struct {
		name, path string
		s          *Scenario
	}{
		{"neither", "engine", withEngine(nil, "")},
		{"no kind", "engine.kind", withEngine(&Engine{Version: "0.32.0"}, "")},
		{"no version", "engine.version", withEngine(&Engine{Kind: "vllm"}, "")},
		{"disagreeing", "engine_version", withEngine(vllm, "0.31.0")},
	} {
		p := c.s.Validate()
		found := false
		for _, it := range p.Errors() {
			found = found || it.Path == c.path
		}
		if !found {
			t.Errorf("%s: no problem at %s:\n%s", c.name, c.path, p.Error())
		}
	}
}
