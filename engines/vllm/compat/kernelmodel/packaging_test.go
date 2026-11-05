// Original to blis-vllm.

package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
)

// Placement reads the cluster's packaging as blis-substrate resolves it from the scenario
// and the chip's catalog entry (scenario.Cluster.Packaging, design SUB-3). The P/D + DP
// fixture states neither gpus_per_node nor gpus_per_rack, so its trays and racks are
// GB200 NVL72's 4 and 72: its 19 trays span two racks, and its two decode ranks sit on
// trays 17 and 18, either side of the rack boundary.
func TestPlacementReadsTheCatalogsPackaging(t *testing.T) {
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	const fixture = "minimax-m2.5-gb200-1p1d-dp2.yaml"
	dec, err := OpenRole(fixture, r, deployment.RoleDecode)
	if err != nil {
		t.Fatal(err)
	}
	if dec.id.gpusPerNode != 4 || dec.id.gpusPerRack != 72 {
		t.Fatalf("decode pool placed with %d GPUs per node and %d per rack, want the catalog's 4 and 72",
			dec.id.gpusPerNode, dec.id.gpusPerRack)
	}
	a, b := dec.PlacementOf(0), dec.PlacementOf(1)
	if a.Node != 17 || a.Rack != 0 || b.Node != 18 || b.Rack != 1 {
		t.Errorf("decode ranks placed at %+v and %+v, want tray 17 in rack 0 and tray 18 in rack 1", a, b)
	}
	if _, _, declared := dec.EndpointPlacement(0); !declared {
		t.Error("the catalog's rack is not reported as declared, so no rack label reaches the router")
	}
}

// A scenario contradicting the chip's catalog entry does not open, and the refusal names
// the field and the entry (SIM-9's pair: the fixture above, restating nothing, opens).
func TestOpenRefusesPackagingThatContradictsTheCatalog(t *testing.T) {
	for _, c := range []struct{ insert, field string }{
		{"  gpus_per_rack: 12\n", "gpus_per_rack"},
		{"  gpus_per_node: 8\n", "gpus_per_node"},
	} {
		dir := variant(t, "minimax-m2.5-gb200-1p1d-dp2.yaml", "  nodes: 19\n", "  nodes: 19\n"+c.insert)
		r := repos()
		r.Scenarios = dir
		_, err := OpenRole("variant.yaml", r, deployment.RoleDecode)
		if err == nil || !strings.Contains(err.Error(), c.field) || !strings.Contains(err.Error(), "hardware/gb200-nvl72.yaml") {
			t.Errorf("%q: got %v, want a refusal naming %s and the entry", c.insert, err, c.field)
		}
	}
}

// variant writes a copy of the committed scenario src, with old replaced by new once, as
// variant.yaml in a fresh directory beside a link to the fixture's router directory.
func variant(t *testing.T, src, old, new string) string {
	t.Helper()
	scenarios, err := filepath.Abs(filepath.Join("..", "..", "testdata", "scenarios"))
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(filepath.Join(scenarios, src))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), old) {
		t.Fatalf("%s has no %q to replace", src, old)
	}
	dir := t.TempDir()
	if err := os.Symlink(filepath.Join(scenarios, "router"), filepath.Join(dir, "router")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(strings.Replace(string(body), old, new, 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}
