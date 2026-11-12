package latencykernel

import (
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/load"
)

// handoffOn builds a two-node B300 scenario on the named fabric and prices one request's KV
// handoff between the nodes. The error is the scenario's refusal, if any.
func handoffOn(t *testing.T, fabric string, tokens int) (kernel.TransferDemand, error) {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	body := fmt.Sprintf(`kind: Scenario
name: b300-handoff
model: llama-3.1-8b-instruct
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-memory, cost-model-fabric]
engine_version: "0.32.0"
cluster:
  hardware: b300
  fabric: %s
  nodes: 2
  gpus_per_node: 8
---
kind: Deployment
name: tp1
pools:
  - role: colocated
    nodes: 2
    parallel: {tp: 1, pp: 1, dp: 1}
`, fabric)
	r, name := openRepos(t, d, body)
	r.Rules = func(string) rules.EngineRules { return inertRules{} }
	r.EngineSettings = func(*deployment.Deployment, int) (enginesettings.Settings, error) {
		return enginesettings.Settings{BlockSize: 16, MaxNumSeqs: 64, MaxNumBatchedTokens: 8192,
			MaxModelLen: 65536}, nil
	}
	in, err := OpenInputs(name, r, 0)
	if err != nil {
		return kernel.TransferDemand{}, err
	}
	k, err := New(in)
	if err != nil {
		return kernel.TransferDemand{}, err
	}
	return k.PDTransferDemand(tokens, kernel.Placement{Node: 0, Pool: deployment.RolePrefill},
		kernel.Placement{Node: 1, Pool: deployment.RoleDecode}), nil
}

// A KV handoff between nodes of a B300 pool on ib-800g crosses the NIC at twice the rate it
// does on ib-400g, so a pull large enough to be bandwidth-bound takes half as long. The
// efficiency is the one the registry states for each class; ib-800g's is assumed equal to
// ib-400g's, so the ratio is exactly two (a measured 800G efficiency would move it, and this
// test with it).
func TestHandoffOverIB800GTakesHalfTheTimeOfIB400G(t *testing.T) {
	const tokens = 60000
	slow, err := handoffOn(t, "ib-400g", tokens)
	if err != nil {
		t.Fatal(err)
	}
	fast, err := handoffOn(t, "ib-800g", tokens)
	if err != nil {
		t.Fatalf("a scenario naming ib-800g was refused: %v", err)
	}
	if slow.Bytes != fast.Bytes || slow.Bytes == 0 {
		t.Fatalf("payload differs between fabrics: %d vs %d bytes", slow.Bytes, fast.Bytes)
	}
	if len(slow.Links) == 0 || len(fast.Links) == 0 {
		t.Fatal("a cross-node handoff crossed no NIC link")
	}
	if ratio := fast.Links[0].BytesPerSecond / slow.Links[0].BytesPerSecond; ratio < 1.999 || ratio > 2.001 {
		t.Errorf("NIC rate over ib-800g is %.4g times ib-400g's, want 2", ratio)
	}
	tSlow, tFast := slow.AloneTime(), fast.AloneTime()
	if tSlow <= slow.Floor || tFast <= fast.Floor {
		t.Fatalf("the pull is floor-bound (%v, %v, floor %v); the test must use a larger transfer",
			tSlow, tFast, slow.Floor)
	}
	if r := float64(tSlow) / float64(tFast); r < 1.99 || r > 2.01 {
		t.Errorf("ib-400g takes %v, ib-800g %v: ratio %.3f, want 2", tSlow, tFast, r)
	}
}

// A fabric class the catalog does not define is still refused, and the refusal names it.
func TestAnUnknownFabricClassIsStillRefused(t *testing.T) {
	_, err := handoffOn(t, "ib-1600g", 1000)
	if err == nil {
		t.Fatal("a scenario naming a fabric the catalog lacks was accepted")
	}
	if !strings.Contains(err.Error(), "ib-1600g") {
		t.Errorf("refusal %q does not name the fabric", err)
	}
}
