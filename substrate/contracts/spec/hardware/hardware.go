// Package hardware defines the three catalog entities that describe physical
// resources: a chip, an inter-node fabric, and a storage device class.
//
// They are three schemas rather than one because they have different owners and
// different lifetimes. A chip's peak rates are a datasheet fact about a die. A
// fabric is a property of the cluster a deployment runs on — blis-catalog states
// this directly, "a fabric is a property of the cluster/pool, not of the GPU die" —
// so the same chip appears behind InfiniBand in one cluster and RoCE in another.
// Storage tiers are a third thing again, owned by whoever provisioned the node.
//
// The separation matters to the cost model. Cross-node collective cost
// turns on the ratio of intra-node to inter-node bandwidth, and that ratio is a
// property of a (chip, fabric) pairing rather than of either alone. A schema that
// put both numbers on the chip would make the ratio look like a die fact and would
// force a new hardware file for every fabric a chip is ever paired with.
package hardware

import (
	"regexp"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// The catalog's files carry free-text keys prefixed with "_comment" — _comment,
// _comment_interconnect, _comment_sm, _comment_pd_transfer — holding the provenance
// narrative for each number.
// They are deliberate content rather than stray fields, so a strict decoder must accept
// them while still rejecting a misspelled real field.
//
// The prefix is "_comment" specifically, not any underscore: hardware/ carries only
// dimensioned physical quantities, and a fitted or dimensionless factor must not enter —
// not even disguised under an underscore. A "_mfu: 0.85" is therefore an unknown field,
// not a comment, and fails like any other, matching the rule blis-catalog's deleted
// validate_catalog.py enforced.
//
// A consequence worth stating: these comment keys are STRIPPED here, before decoding, so
// no provenance prose survives onto the struct. The deleted gate's SM-count citation rule
// (#32, item 4) — every SMCount must carry a chaseable source URL — therefore cannot live
// in Chip.Validate(), which never sees the prose. That rule is a catalog-provenance policy
// (does the prose cite a source), not a schema-structural invariant (is the value a sane
// datasheet figure), so it belongs in a catalog-side lint over the raw files rather than
// here. It is deliberately NOT enforced by this package and nowhere else today; issue #32
// is the tracked home for the decision on where that lint lives, and stays open until it
// exists. This is the explicit tracking #32 asks for in lieu of a silent drop.
//
// The files also carry no name: identity is the filename. Chip.Name and Fabric.Name are
// tagged `yaml:"-"`, so yaml.v3 binds no `name` key to them and the strict decoder rejects
// a stray one — an in-file name cannot become a second source of truth for an identity the
// filename already fixes. A loader sets Name from the path stem (its only source), and
// validation requires it, so a chip that reaches a cost model without one is an error
// rather than an anonymous descriptor.
//
// The strip itself is validate.StripCatalogComments, shared with spec/workload (#28) so the
// one convention is implemented once; the hardware-specific policy above (what counts as a
// comment, why a bare underscore does not) still lives here because it is where the catalog's
// hardware/ namespace documents it.

// Chip is what a GPU die can do. Every field is a vendor figure; nothing here is
// fitted. A fitted quantity belongs in blis-registry, where it can carry a method
// and a scope.
type Chip struct {
	// Name is the chip's identity, and it is a filename fact, not a file field. The
	// `yaml:"-"` tag means yaml.v3 binds no `name:` key to it, so under the strict decoder
	// an in-file `name` is an unknown field and is rejected; LoadChip stamps the field from
	// the path stem as its only source. This matches workload.Shape (#25) and the package
	// doc above, which the field previously contradicted by carrying `yaml:"name"` (#27).
	Name       string           `yaml:"-"`
	Provenance vocab.Provenance `yaml:"Provenance"`

	// Peak dense rates, in TFLOP/s. FP8Peak is zero on parts without native FP8.
	BF16Peak float64 `yaml:"TFlopsPeak"`
	FP8Peak  float64 `yaml:"TFlopsFP8"`
	// NVFP4Peak is zero where the format is emulated rather than native, which is
	// the honest encoding: a model must not read a peak that the hardware reaches
	// only through a dequantize path.
	NVFP4Peak float64 `yaml:"TFlopsNVFP4,omitempty"`

	// MemoryBandwidthTBs is the HBM peak, nominal. The empirical derate that turns
	// it into an achievable rate is a coefficient, not a chip fact.
	MemoryBandwidthTBs float64 `yaml:"BwPeakTBs"`
	MemoryGiB          float64 `yaml:"MemoryGiB"`

	// IntraNodeBwGBps is per-GPU unidirectional on-node bandwidth: NVLink where
	// present. It stays on the chip because it is a die-and-package property.
	IntraNodeBwGBps float64 `yaml:"IntraNodeBwGBps"`

	// SMCount is the streaming-multiprocessor count. A model needs it to size a
	// withheld-SM derate as a fraction rather than an absolute, and an engine reads
	// it from the driver rather than hardcoding it per part. It is REQUIRED and must
	// be positive: no shipped chip has zero SMs, so a missing or zero count is an
	// omission, not a legitimate value — hence no omitempty, and Validate rejects it.
	SMCount int `yaml:"SMCount"`

	// ComputeCapability is the CUDA compute capability, "major.minor": "9.0" for H100
	// and H200, "10.0" for B200 and GB200, "10.3" for B300 and GB300. An engine selects
	// kernels, and with them what some weights and caches are resident as, by its major
	// version (the engine's is_device_capability_family), so a model graph's arch_overrides are
	// keyed by it. Empty means not stated: a graph that is the same on every chip still
	// resolves, and one that distinguishes architectures cannot be resolved for this chip.
	ComputeCapability string `yaml:"ComputeCapability,omitempty"`

	// GPUsPerNode and GPUsPerRack describe packaging. GPUsPerRack is non-zero only
	// where a rack boundary is a distinct link from a node boundary, which is the
	// third topology tier a two-tier model cannot express.
	GPUsPerNode int `yaml:"GPUsPerNode,omitempty"`
	GPUsPerRack int `yaml:"GPUsPerRack,omitempty"`

	// IntraRackBwGBps applies between nodes inside one rack, where that differs
	// from the inter-rack fabric. Zero means no such tier exists on this part.
	IntraRackBwGBps float64 `yaml:"IntraRackBwGBps,omitempty"`
}

// Fabric is an inter-node network class, reusable across chips.
type Fabric struct {
	// Name is a filename fact, tagged `yaml:"-"` for the same reason as Chip.Name: identity
	// is the file's stem, LoadFabric stamps it, and an in-file `name` is a rejected unknown
	// field rather than a competing source of truth.
	Name       string           `yaml:"-"`
	Provenance vocab.Provenance `yaml:"Provenance"`

	// InterNodeBwGBps is per-GPU unidirectional, and NOMINAL: a line rate divided
	// by eight bits, not a measurement. The measured fraction of it is a
	// per-deployment coefficient in blis-registry. Treating this figure as achieved
	// overstates transfer bandwidth, which is why the field name says nothing about
	// effectiveness and this comment says it plainly.
	InterNodeBwGBps float64 `yaml:"InterNodeBwGBps"`

	// RDMA distinguishes a fabric that keeps the CPU out of the data path from one
	// that does not. It changes how far below nominal a transfer lands, and a name
	// like "ethernet-100gbe" does not say which side of the line the fabric is on.
	RDMA bool `yaml:"RDMA,omitempty"`
}

// StorageDevice is one offload tier class: CPU memory, an NVMe class, an object
// store. Read and write bandwidths are separate because they differ materially on
// flash, and pricing an eviction at read bandwidth understates it.
type StorageDevice struct {
	// Name is NOT tagged `yaml:"-"` like Chip.Name and Fabric.Name, and that is deliberate
	// (#27): a storage tier has no per-device file whose stem could supply an identity. It
	// is loaded from a mapping of tier name to facts (LoadStorageDevices), where the map KEY
	// is the identity and the loader stamps Name from it, overwriting whatever the value
	// held. So the enforcement differs from a chip's: an in-file `name` here is accepted and
	// then clobbered by the key rather than rejected, which is harmless because the key — not
	// the field — always wins. There is no second-source-of-truth risk to reject.
	Name string `yaml:"name"`
	// The wire keys carry their units — _mb_s, _us — because the YAML key is the only
	// thing a catalog author writes, and a unit stated only in this Go field name or in a
	// file comment is not a contract: a device whose datasheet quotes ms, entered as if it
	// were µs, would make that tier look 1000x faster and pass every validator (#18). The
	// consumer (blis-latency-kernel) reads BaseLatencyUs as microseconds, so the key says so.
	ReadBandwidthMBs  float64 `yaml:"read_bandwidth_mb_s"`
	WriteBandwidthMBs float64 `yaml:"write_bandwidth_mb_s"`
	// BaseLatencyUs is the fixed per-transfer cost, which dominates small
	// transfers and is three orders of magnitude apart across the tiers.
	BaseLatencyUs float64 `yaml:"base_latency_us"`
}

// UnmarshalYAML accepts the catalog's `_comment`-prefixed comment keys while leaving
// every other unknown field — including any other underscore-prefixed one — an error.
func (c *Chip) UnmarshalYAML(node *yaml.Node) error {
	if err := validate.RejectUnknownKeys(node, Chip{}, validate.IgnoreCatalogComments); err != nil {
		return err
	}
	validate.StripCatalogComments(node)
	type shape Chip
	var raw shape
	if err := validate.DecodeNode(node, &raw); err != nil {
		return err
	}
	*c = Chip(raw)
	return nil
}

// UnmarshalYAML accepts the catalog's comment keys, as Chip does.
func (f *Fabric) UnmarshalYAML(node *yaml.Node) error {
	if err := validate.RejectUnknownKeys(node, Fabric{}, validate.IgnoreCatalogComments); err != nil {
		return err
	}
	validate.StripCatalogComments(node)
	type shape Fabric
	var raw shape
	if err := validate.DecodeNode(node, &raw); err != nil {
		return err
	}
	*f = Fabric(raw)
	return nil
}

// UnmarshalYAML accepts the catalog's comment keys, as Chip does.
func (d *StorageDevice) UnmarshalYAML(node *yaml.Node) error {
	if err := validate.RejectUnknownKeys(node, StorageDevice{}, validate.IgnoreCatalogComments); err != nil {
		return err
	}
	validate.StripCatalogComments(node)
	type shape StorageDevice
	var raw shape
	if err := validate.DecodeNode(node, &raw); err != nil {
		return err
	}
	*d = StorageDevice(raw)
	return nil
}

// CapabilityMajor returns the major version of ComputeCapability, and false when it is
// unstated or malformed.
func (c *Chip) CapabilityMajor() (int, bool) {
	if !capabilityForm.MatchString(c.ComputeCapability) {
		return 0, false
	}
	major, _, _ := strings.Cut(c.ComputeCapability, ".")
	m, err := strconv.Atoi(major)
	return m, err == nil
}

// capabilityForm is major.minor with no sign or leading zero on the major version.
var capabilityForm = regexp.MustCompile(`^[1-9][0-9]*\.[0-9]+$`)
