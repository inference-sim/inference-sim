package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/load"
)

// The kernel prices the cluster's resolved packaging (SUB-3), and its document check (the
// field layer of contracts.Validate, which validateDocuments runs) refuses a scenario
// that contradicts the chip's catalog entry before anything is priced.
// A GB200 NVL72 scenario that omits gpus_per_node and gpus_per_rack resolves to the
// entry's 4 and 72, as one that restates them does; pricing the two alike follows, since
// the layout reads only the resolved values (resolve.ResolveLayout). The engine models'
// golden suites price such scenarios end to end.
func TestKernelReadsTheResolvedPackaging(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	const body = `kind: Scenario
name: nvl72-tp8
model: llama-3.1-8b-instruct
coefficients: [cost-model-primitives, cost-model-host-overheads]
engine_version: "0.32.0"
cluster:
  hardware: gb200-nvl72
  fabric: ib-400g
  nodes: 2
PACKAGING---
kind: Deployment
name: tp8
pools:
  - role: colocated
    nodes: 2
    parallel: {tp: 8, pp: 1, dp: 1}
`
	// The field layer validateDocuments runs, without the engine rules this test has none of.
	check := func(in Inputs) error {
		rep := contracts.Validate(contracts.Bundle{Scenario: in.Scenario, Deployment: in.Deployment,
			Chip: in.Chip, Fabric: in.Fabric})
		if !rep.Field.OK() {
			return rep.Field
		}
		return nil
	}
	inputs := func(packaging string) Inputs {
		r, name := openRepos(t, d, strings.Replace(body, "PACKAGING", packaging, 1))
		in, err := OpenInputs(name, r, 0)
		if err != nil {
			t.Fatal(err)
		}
		return in
	}
	for _, packaging := range []string{"", "  gpus_per_node: 4\n  gpus_per_rack: 72\n"} {
		in := inputs(packaging)
		if err := check(in); err != nil {
			t.Errorf("%q: refused: %v", packaging, err)
		}
		pk, err := in.Scenario.Cluster.Packaging(in.Chip)
		if err != nil || pk.GPUsPerNode != 4 || pk.GPUsPerRack != 72 {
			t.Errorf("%q: resolved to %+v, %v; want the entry's 4 and 72", packaging, pk, err)
		}
	}
	for _, c := range []struct{ packaging, field string }{
		{"  gpus_per_node: 8\n", "gpus_per_node"},
		{"  gpus_per_rack: 12\n", "gpus_per_rack"},
	} {
		err := check(inputs(c.packaging))
		if err == nil || !strings.Contains(err.Error(), "scenario.cluster."+c.field) ||
			!strings.Contains(err.Error(), "hardware/gb200-nvl72.yaml") {
			t.Errorf("%q: got %v, want a refusal at scenario.cluster.%s naming the entry", c.packaging, err, c.field)
		}
	}
}
