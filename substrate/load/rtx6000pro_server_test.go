package load

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// The RTX PRO 6000 Blackwell Server Edition is a PCIe card. Its entry resolves from the
// embedded catalog, passes the chip validator, and states what the datasheet states: dense
// peaks, a 64 GB/s PCIe link and no NVLink domain, so a scenario sizes its node and may not
// size a rack (SUB-3).
func TestRTXPro6000ServerResolvesAsAPCIePart(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	chip, err := d.Chip("rtx6000pro-server")
	if err != nil {
		t.Fatalf("the catalog does not resolve the part InferenceX labels rtx6000pro: %v", err)
	}
	if p := chip.Validate(); !p.OK() {
		t.Fatalf("the entry does not validate: %v", p)
	}
	if major, ok := chip.CapabilityMajor(); !ok || major != 12 {
		t.Errorf("compute capability major = %d, %v; NVIDIA lists 12.0", major, ok)
	}
	// Dense figures, not the sparse 1, 2 and 4 PFLOPS the product page prints.
	if chip.BF16Peak >= 1000 || chip.FP8Peak != 2*chip.BF16Peak || chip.NVFP4Peak != 2*chip.FP8Peak {
		t.Errorf("peaks BF16 %v, FP8 %v, FP4 %v: want dense figures in the ratio 1:2:4",
			chip.BF16Peak, chip.FP8Peak, chip.NVFP4Peak)
	}
	if chip.IntraNodeBwGBps != 64 || chip.GPUsPerNode != 0 || chip.GPUsPerRack != 0 || chip.IntraRackBwGBps != 0 {
		t.Errorf("link %v GB/s, %d per node, %d per rack: want PCIe Gen5 x16 and packaging left to the scenario",
			chip.IntraNodeBwGBps, chip.GPUsPerNode, chip.GPUsPerRack)
	}
	if src, err := chip.SourceOf("MemoryGiB"); err != nil || src.Kind != vocab.FieldMeasured {
		t.Errorf("MemoryGiB source = %+v, %v; want the measured device total", src, err)
	}

	pk, err := scenario.Cluster{Hardware: "rtx6000pro-server", GPUsPerNode: 8}.Packaging(chip)
	if err != nil || pk.GPUsPerNode != 8 || pk.GPUsPerNodeSource != scenario.SourceStated || pk.GPUsPerRack != 0 {
		t.Errorf("an 8-GPU server: %+v, %v; want 8 GPUs a node, stated, and no rack", pk, err)
	}
	if _, err := (scenario.Cluster{Hardware: "rtx6000pro-server"}).Packaging(chip); err == nil ||
		!strings.Contains(err.Error(), "gpus_per_node") {
		t.Errorf("a cluster that leaves the node size open: %v; want gpus_per_node required", err)
	}
	if _, err := (scenario.Cluster{Hardware: "rtx6000pro-server", GPUsPerNode: 8, GPUsPerRack: 72}).Packaging(chip); err == nil ||
		!strings.Contains(err.Error(), "gpus_per_rack") {
		t.Errorf("a cluster that sizes a rack: %v; want gpus_per_rack refused", err)
	}
}
