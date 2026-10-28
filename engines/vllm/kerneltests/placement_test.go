// Moved from blis-latency-kernel/placement_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"path/filepath"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/kernel"
)

// INSIDE ONE NVLINK RACK, A TRAY BOUNDARY IS NOT A FABRIC CROSSING. An NVL72-class part puts
// four GPUs on a tray and 72 on one NVSwitch fabric at the full per-GPU rate, so a tp=8
// group across two trays never reaches the inter-node fabric. Contrasted with the same
// layout on a part whose rack is not one domain, where the same group does cross.
func TestAGroupInsideOneNVLinkRackDoesNotCrossTheFabric(t *testing.T) {
	build := func(rackDomain bool) *Kernel {
		t.Helper()
		in := fixtureInputs(t, "minimax-m25-gb300-tp4.yaml")
		pool := &in.Deployment.Pools[0]
		pool.Parallel.TP = 8
		pool.Nodes, in.Scenario.Cluster.Nodes = 2, 2
		in.Scenario.Cluster.Fabric = "ib-400g"
		f, err := schemas.LoadFabric(filepath.Join(catalogRoot, "networks", "ib-400g.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		in.Fabric = f
		if !rackDomain {
			chip := *in.Chip
			chip.GPUsPerRack, chip.IntraRackBwGBps = 0, 0
			in.Chip = &chip
		}
		k, err := newKernel(in)
		if err != nil {
			t.Fatalf("rack domain %v: %v", rackDomain, err)
		}
		return k
	}
	b := decodeBatch(64, 1, 4096)
	if nic := build(true).StepTime(b).PerResource[kernel.ResourceNIC]; nic != 0 {
		t.Errorf("a tp=8 group inside one NVLink rack charged %v to the fabric", nic)
	}
	if nic := build(false).StepTime(b).PerResource[kernel.ResourceNIC]; nic == 0 {
		t.Error("without a rack domain the same tp=8 group across two trays charged " +
			"nothing to the fabric; the contrast proves nothing")
	}
}
