// Moved from blis-latency-kernel/placement_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-substrate/kernel"

	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// INSIDE ONE NVLINK RACK, A TRAY BOUNDARY IS NOT A FABRIC CROSSING. An NVL72-class part puts
// four GPUs on a tray and 72 on one NVSwitch fabric at the full per-GPU rate, so a tp=8
// group across two trays never reaches the inter-node fabric. Contrasted with the same
// layout on a part whose rack is not one domain, where the same group does cross.
func TestAGroupInsideOneNVLinkRackDoesNotCrossTheFabric(t *testing.T) {
	build := func(rackDomain bool) *Kernel {
		t.Helper()
		in := fixtureInputs(t, "muse-gb200-tp8ep8.yaml")
		// The fixture omits gpus_per_node and gpus_per_rack, so the 4 and 72 below come from the
		// catalog entry through the resolver, not from the scenario.
		if !rackDomain {
			// The contrast also goes through the resolver: a copy of the chip whose entry
			// states no rack, which Cluster.Packaging resolves to GPUsPerRack 0 (the scenario
			// states none either). The node size stays the entry's 4.
			chip := *in.Chip
			chip.GPUsPerRack, chip.IntraRackBwGBps = 0, 0
			if pk, err := in.Scenario.Cluster.Packaging(&chip); err != nil || pk.GPUsPerRack != 0 || pk.GPUsPerNode != 4 {
				t.Fatalf("the rackless chip resolved to %+v, %v; want 4 per node and no rack", pk, err)
			}
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
