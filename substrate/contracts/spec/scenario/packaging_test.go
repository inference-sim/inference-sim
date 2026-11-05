package scenario

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
)

// nvl72 is a chip whose product fixes its packaging, as hardware/gb200-nvl72.yaml states
// it: four GPUs per tray and a 72-GPU NVLink rack.
func nvl72() *hardware.Chip {
	return &hardware.Chip{Name: "gb200-nvl72", GPUsPerNode: 4, GPUsPerRack: 72, IntraRackBwGBps: 900}
}

// hgx is a chip whose entry leaves its packaging open, as hardware/h100.yaml does: the
// node size is the user's, and the part has no rack tier.
func hgx() *hardware.Chip { return &hardware.Chip{Name: "h200"} }

func cluster(hw string, gpn, gpr int) Cluster {
	return Cluster{Hardware: hw, Fabric: "ib-400g", Nodes: 2, GPUsPerNode: gpn, GPUsPerRack: gpr}
}

// TestPackagingResolves covers what Packaging accepts and where each value comes from: an
// omitted field takes the entry's value, a restated one is accepted, and a chip that
// leaves the node open takes the scenario's.
func TestPackagingResolves(t *testing.T) {
	for _, c := range []struct {
		name string
		c    Cluster
		chip *hardware.Chip
		want Packaging
	}{
		{"NVL72, both omitted", cluster("gb200-nvl72", 0, 0), nvl72(),
			Packaging{4, SourceCatalog, 72, SourceCatalog}},
		{"NVL72, both restated", cluster("gb200-nvl72", 4, 72), nvl72(),
			Packaging{4, SourceCatalog, 72, SourceCatalog}},
		{"NVL72, node restated, rack omitted", cluster("gb200-nvl72", 4, 0), nvl72(),
			Packaging{4, SourceCatalog, 72, SourceCatalog}},
		{"HGX, node stated, no rack", cluster("h200", 8, 0), hgx(),
			Packaging{8, SourceStated, 0, ""}},
		{"a 4-GPU server of an open chip", cluster("h200", 4, 0), hgx(),
			Packaging{4, SourceStated, 0, ""}},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := c.c.Packaging(c.chip)
			if err != nil {
				t.Fatalf("refused: %v", err)
			}
			if got != c.want {
				t.Errorf("Packaging = %+v, want %+v", got, c.want)
			}
			if p := got.Validate(); !p.OK() {
				t.Errorf("the resolved packaging does not validate:\n%s", p.Error())
			}
			if p := got.ValidateAgainst(c.c); !p.OK() {
				t.Errorf("the resolved packaging disagrees with its cluster:\n%s", p.Error())
			}
		})
	}
}

// TestPackagingRefusals is SIM-9's pair for each refusal: the smallest cluster that
// triggers it, and the closest one that does not. Each message names the field, both
// values and the catalog entry. Each refusing case is also a known-bad input for the
// packaging check (X18).
func TestPackagingRefusals(t *testing.T) {
	for _, c := range []struct {
		name      string
		bad, good Cluster
		chip      *hardware.Chip
		field     string
		mentions  []string
	}{
		{"a node size other than the entry's", cluster("gb200-nvl72", 8, 0), cluster("gb200-nvl72", 4, 0), nvl72(),
			"gpus_per_node", []string{"is 8", "hardware/gb200-nvl72.yaml", "at 4"}},
		{"a rack size other than the entry's", cluster("gb200-nvl72", 0, 12), cluster("gb200-nvl72", 0, 72), nvl72(),
			"gpus_per_rack", []string{"is 12", "hardware/gb200-nvl72.yaml", "at 72"}},
		{"no node size on a chip that states none", cluster("h200", 0, 0), cluster("h200", 8, 0), hgx(),
			"gpus_per_node", []string{"required", "hardware/h200.yaml", "got 0"}},
		{"a rack on a chip with no rack tier", cluster("h200", 8, 16), cluster("h200", 8, 0), hgx(),
			"gpus_per_rack", []string{"is 16", "hardware/h200.yaml", "no GPUsPerRack"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := c.good.Packaging(c.chip); err != nil {
				t.Fatalf("the closest accepted cluster is refused: %v", err)
			}
			_, err := c.bad.Packaging(c.chip)
			if err == nil {
				t.Fatal("accepted")
			}
			p, ok := err.(*validate.Problems)
			if !ok {
				t.Fatalf("error is %T, want *validate.Problems", err)
			}
			if errs := p.Errors(); len(errs) != 1 || errs[0].Path != c.field {
				t.Fatalf("want one problem at %s, got:\n%s", c.field, p.Error())
			}
			for _, m := range c.mentions {
				if !strings.Contains(p.Error(), m) {
					t.Errorf("the message does not mention %q:\n%s", m, p.Error())
				}
			}
			if p := c.bad.ValidatePackaging(c.chip); p.OK() {
				t.Error("ValidatePackaging accepts what Packaging refuses")
			}
		})
	}
}

// The resolver refuses a chip that is not the one the cluster names, and no chip at all,
// rather than resolve against the wrong entry.
func TestPackagingNeedsTheClustersChip(t *testing.T) {
	if _, err := cluster("h200", 8, 0).Packaging(nil); err == nil {
		t.Error("resolved with no chip")
	}
	if _, err := cluster("h200", 8, 0).Packaging(nvl72()); err == nil {
		t.Error("resolved an h200 cluster against the gb200-nvl72 entry")
	}
}

// A recorded packaging is checked on its own and against the cluster it was resolved from.
func TestRecordedPackaging(t *testing.T) {
	for _, c := range []struct {
		name string
		pk   Packaging
	}{
		{"no node size", Packaging{0, SourceStated, 0, ""}},
		{"unknown source", Packaging{8, "default", 0, ""}},
		{"a rack stated by the scenario", Packaging{4, SourceCatalog, 72, SourceStated}},
		{"a rack source without a rack", Packaging{8, SourceStated, 0, SourceCatalog}},
		{"a rack not a multiple of a node", Packaging{8, SourceStated, 12, SourceCatalog}},
	} {
		if c.pk.Validate().OK() {
			t.Errorf("%s: accepted %+v", c.name, c.pk)
		}
	}
	pk := Packaging{4, SourceCatalog, 72, SourceCatalog}
	if p := pk.ValidateAgainst(cluster("gb200-nvl72", 8, 0)); p.OK() {
		t.Error("a recorded node size other than the cluster's stated one is accepted")
	}
	if p := (Packaging{8, SourceStated, 0, ""}).ValidateAgainst(cluster("h200", 0, 0)); p.OK() {
		t.Error("a node size recorded as stated, over a cluster that states none, is accepted")
	}
}
