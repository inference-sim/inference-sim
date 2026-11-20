package scenario

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
)

// Who owns a cluster's packaging (design §2.2, SUB-3).
//
// A chip's catalog entry owns GPUsPerNode and GPUsPerRack when its product fixes them. A
// GB200 NVL72 compute tray holds four GPUs, and its rack is one NVLink domain of 72: those
// are facts of the product, cited in hardware/gb200-nvl72.yaml, and no scenario can change
// them. The scenario's cluster owns what the user provisions: the node count, and the GPUs
// per node of a chip whose entry states none, such as an 8-GPU HGX H100 server.
//
// A rack, in BLIS, is an NVLink domain that spans nodes: the third topology tier, whose
// bandwidth is the entry's IntraRackBwGBps. The cost model prices it only where the entry
// states it. The kernel's RackIsOneDomain reads GPUsPerRack and IntraRackBwGBps from the
// chip. An engine places ranks into racks only so the cost model can price that tier.
// hardware.Chip.Validate refuses IntraRackBwGBps without GPUsPerRack, so an entry that
// states no GPUsPerRack has no rack tier at all, and there is no rack for the scenario to
// size. A scenario's gpus_per_rack on such a chip describes an NVLink domain the product
// does not have, and Packaging refuses it.
//
// Packaging is the one function that resolves the two (X15). Every reader calls it, or
// reads a Packaging it returned: the cost model's layout, the engines' placement, the
// validators' cluster-fit checks and the Result's inputs. A reader that reads
// Cluster.GPUsPerNode or Cluster.GPUsPerRack directly reads what the scenario stated, which
// is zero when the scenario leaves the value to the catalog.

// PackagingSource says which document supplied a resolved packaging value.
type PackagingSource string

const (
	// SourceStated: the chip's entry leaves the value open, and the scenario states it.
	SourceStated PackagingSource = "stated"
	// SourceCatalog: the chip's entry states the value. The scenario omitted it or
	// restated it unchanged.
	SourceCatalog PackagingSource = "catalog"
)

// Packaging is a cluster's resolved packaging: the GPUs in one node and in one rack, and
// the source of each.
type Packaging struct {
	// GPUsPerNode is the number of GPUs in one node. Always at least 1.
	GPUsPerNode int `yaml:"gpus_per_node"`
	// GPUsPerNodeSource says whether the scenario or the chip's entry supplied it.
	GPUsPerNodeSource PackagingSource `yaml:"gpus_per_node_source"`
	// GPUsPerRack is the number of GPUs in one rack, an NVLink domain that spans nodes.
	// Zero means the chip has no rack tier, and nodes meet only through the fabric.
	GPUsPerRack int `yaml:"gpus_per_rack,omitempty"`
	// GPUsPerRackSource is SourceCatalog when GPUsPerRack is set, and empty otherwise:
	// only a chip's entry states a rack.
	GPUsPerRackSource PackagingSource `yaml:"gpus_per_rack_source,omitempty"`
}

// Packaging resolves the cluster's packaging against chip, the catalog entry its Hardware
// names.
//
// A field the scenario omits takes the entry's value. A field it restates unchanged is
// accepted. A different value is refused, with a message naming the field, both values and
// the entry. gpus_per_node is required when the entry states no GPUsPerNode, and
// gpus_per_rack is refused when the entry states no GPUsPerRack. The error is a
// *validate.Problems located at the cluster's fields.
func (c Cluster) Packaging(chip *hardware.Chip) (Packaging, error) {
	pk, p := c.resolvePackaging(chip)
	if !p.OK() {
		return Packaging{}, p
	}
	return pk, nil
}

// ValidatePackaging reports, at the cluster's fields, every way the cluster's packaging
// disagrees with chip. It is Packaging's check without its result, for a validator that
// accumulates problems.
func (c Cluster) ValidatePackaging(chip *hardware.Chip) *validate.Problems {
	_, p := c.resolvePackaging(chip)
	return p
}

func (c Cluster) resolvePackaging(chip *hardware.Chip) (Packaging, *validate.Problems) {
	p := &validate.Problems{}
	if chip == nil {
		p.Field("hardware", "the packaging of a %q cluster resolves from its catalog entry, and none was given", c.Hardware)
		return Packaging{}, p
	}
	if chip.Name != "" && c.Hardware != "" && chip.Name != c.Hardware {
		p.Field("hardware", "is %q, but its packaging was resolved against the catalog entry %q", c.Hardware, chip.Name)
		return Packaging{}, p
	}
	entry := "hardware/" + chip.Name + ".yaml"
	var pk Packaging

	switch {
	case chip.GPUsPerNode > 0 && c.GPUsPerNode != 0 && c.GPUsPerNode != chip.GPUsPerNode:
		p.Field("gpus_per_node", "is %d, but %s fixes GPUsPerNode at %d: the product packages %d GPUs in a node. "+
			"Omit gpus_per_node, or state %d", c.GPUsPerNode, entry, chip.GPUsPerNode, chip.GPUsPerNode, chip.GPUsPerNode)
	case chip.GPUsPerNode > 0:
		pk.GPUsPerNode, pk.GPUsPerNodeSource = chip.GPUsPerNode, SourceCatalog
	case c.GPUsPerNode < 1:
		p.Field("gpus_per_node", "required: %s states no GPUsPerNode, so the scenario states how many GPUs each node holds (at least 1), got %d",
			entry, c.GPUsPerNode)
	default:
		pk.GPUsPerNode, pk.GPUsPerNodeSource = c.GPUsPerNode, SourceStated
	}

	switch {
	case chip.GPUsPerRack > 0 && c.GPUsPerRack != 0 && c.GPUsPerRack != chip.GPUsPerRack:
		p.Field("gpus_per_rack", "is %d, but %s fixes GPUsPerRack at %d: the product's NVLink domain is %d GPUs. "+
			"Omit gpus_per_rack, or state %d", c.GPUsPerRack, entry, chip.GPUsPerRack, chip.GPUsPerRack, chip.GPUsPerRack)
	case chip.GPUsPerRack > 0:
		pk.GPUsPerRack, pk.GPUsPerRackSource = chip.GPUsPerRack, SourceCatalog
	case c.GPUsPerRack != 0:
		p.Field("gpus_per_rack", "is %d, but %s states no GPUsPerRack: the product has no NVLink domain that spans nodes, "+
			"so there is no rack to size. Remove gpus_per_rack", c.GPUsPerRack, entry)
	}

	if p.OK() && pk.GPUsPerRack > 0 && pk.GPUsPerRack%pk.GPUsPerNode != 0 {
		p.Field("gpus_per_node", "%d does not divide the %d GPUs of a rack that %s states", pk.GPUsPerNode, pk.GPUsPerRack, entry)
	}
	return pk, p
}

// Validate checks a resolved packaging on its own, as a document that records one (a
// Result's inputs) carries it: the counts are positive, a rack is a whole number of nodes,
// and each source is one the resolver gives.
func (pk Packaging) Validate() *validate.Problems {
	p := &validate.Problems{}
	if pk.GPUsPerNode < 1 {
		p.Field("gpus_per_node", "must be at least 1, got %d", pk.GPUsPerNode)
	}
	switch pk.GPUsPerNodeSource {
	case SourceStated, SourceCatalog:
	default:
		p.Field("gpus_per_node_source", "%q is not one of %s, %s", pk.GPUsPerNodeSource, SourceStated, SourceCatalog)
	}
	switch {
	case pk.GPUsPerRack < 0:
		p.Field("gpus_per_rack", "must not be negative, got %d", pk.GPUsPerRack)
	case pk.GPUsPerRack == 0 && pk.GPUsPerRackSource != "":
		p.Field("gpus_per_rack_source", "is %q, but no rack is recorded", pk.GPUsPerRackSource)
	case pk.GPUsPerRack > 0 && pk.GPUsPerRackSource != SourceCatalog:
		p.Field("gpus_per_rack_source", "must be %s: only a chip's catalog entry states a rack, got %q", SourceCatalog, pk.GPUsPerRackSource)
	case pk.GPUsPerRack > 0 && pk.GPUsPerNode > 0 && pk.GPUsPerRack%pk.GPUsPerNode != 0:
		p.Field("gpus_per_rack", "%d is not a multiple of gpus_per_node %d", pk.GPUsPerRack, pk.GPUsPerNode)
	}
	return p
}

// ValidateAgainst checks that a recorded packaging is one the cluster could resolve to: a
// value the cluster states is the recorded value, and a value recorded as stated is one the
// cluster states. It cannot check the catalog's half, which needs the chip; Packaging does.
func (pk Packaging) ValidateAgainst(c Cluster) *validate.Problems {
	p := &validate.Problems{}
	if c.GPUsPerNode != 0 && c.GPUsPerNode != pk.GPUsPerNode {
		p.Field("gpus_per_node", "is %d, but the scenario's cluster states %d", pk.GPUsPerNode, c.GPUsPerNode)
	}
	if pk.GPUsPerNodeSource == SourceStated && c.GPUsPerNode == 0 {
		p.Field("gpus_per_node_source", "is %s, but the scenario's cluster states no gpus_per_node", SourceStated)
	}
	if c.GPUsPerRack != 0 && c.GPUsPerRack != pk.GPUsPerRack {
		p.Field("gpus_per_rack", "is %d, but the scenario's cluster states %d", pk.GPUsPerRack, c.GPUsPerRack)
	}
	return p
}

// String renders the packaging for a message: "4 GPUs per node (catalog), 72 per rack (catalog)".
func (pk Packaging) String() string {
	s := fmt.Sprintf("%d GPUs per node (%s)", pk.GPUsPerNode, pk.GPUsPerNodeSource)
	if pk.GPUsPerRack > 0 {
		s += fmt.Sprintf(", %d per rack (%s)", pk.GPUsPerRack, pk.GPUsPerRackSource)
	}
	return s
}
