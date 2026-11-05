// Package scenario describes the immutable problem a user is handed: which model, on
// which available hardware, measured against which workload (a distributional shape or a
// concrete captured trace), and fitted against which coefficient and engine-version
// references.
//
// A scenario is a file rather than a flag list so that an estimate is reproducible
// from committed artifacts. It names catalog and registry entries by identity. What it
// deliberately does NOT carry is the tunable configuration an optimizer sweeps —
// parallelism, how many prefill vs decode pods, engine knobs, offload layout, PD
// transfer; those are a separate Deployment (spec/deployment). The split keeps the
// immutable problem apart from the mutable choices made against it: one `blis run` is
// one Scenario and one Deployment, and the operating point (load level) is a run-level
// sweep axis that is neither.
package scenario

import (
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// Scenario is the immutable problem: a model, the hardware it is available on, the
// workload it is measured against, and the references that identify what a prediction is
// fitted to.
type Scenario struct {
	Kind string `yaml:"kind"` // "Scenario"
	Name string `yaml:"name"`

	// Model names a catalog model graph.
	Model string `yaml:"model"`

	// Coefficients names blis-registry sets, applied in order. Sets fitted for a
	// different cost model do not belong here, however similar their entry names.
	Coefficients []string `yaml:"coefficients"`

	// Workload is the "what traffic" slot: a sum type binding either a catalog workload
	// shape (distributional) or a reference to an external captured trace (concrete) —
	// see workload.Binding. A scenario without one describes a problem but no traffic,
	// which is enough for a capacity question and not enough for a throughput one, so it
	// is a pointer and omitting it is valid. When present it must choose exactly one arm.
	Workload *workload.Binding `yaml:"workload,omitempty"`

	// Acceptance pins speculative acceptance for the run, above the workload's and the
	// registry's: by naming an acceptance profile set in the registry (as Coefficients
	// names coefficient sets) or by an inline value. To score against a run that forced
	// its acceptance, pin the value it forced. Omitted, the workload and the registry
	// decide (acceptance.Resolve).
	Acceptance *acceptance.Pin `yaml:"acceptance,omitempty"`

	// Routing pins expert-parallel routing imbalance for the run, above the workload's and
	// the registry's: by naming a routing profile set in the registry (a sensitivity set is
	// usable only this way) or by an inline value. Omitted, the workload and the registry's
	// default sets decide, and absent both the ranks are priced as evenly loaded
	// (routing.Resolve).
	Routing *routing.Pin `yaml:"routing,omitempty"`

	// Engine names the engine that runs the scenario, by kind and upstream release
	// (SIM-1). The simulator chooses the engine by it and refuses a release it does not
	// run. It is distinct from the per-pool engine settings a Deployment states: those
	// move between releases, this identifies which release.
	Engine *Engine `yaml:"engine,omitempty"`

	// EngineVersion is the field Engine replaces. Deprecated: by M1, in the first substrate release with Engine;
	// removed in M4 (plan §3.3). A scenario may state it alone, or beside Engine with
	// the same version; EngineRelease reads either.
	EngineVersion string `yaml:"engine_version,omitempty"`

	// Cluster is the available-hardware inventory: the accelerator every node carries,
	// the inter-node fabric, how the nodes are packaged, and the storage classes a
	// deployment may offload onto.
	Cluster Cluster `yaml:"cluster"`
}

// Engine is an engine by kind and upstream release, for example {kind: vllm, version:
// 0.32.0}.
type Engine struct {
	// Kind names the engine, as its engine repository registers it: vllm or sglang.
	Kind string `yaml:"kind"`
	// Version is the upstream release the engine models, for example "0.32.0".
	Version string `yaml:"version"`
}

// EngineRelease returns the upstream release the scenario names: Engine.Version when
// Engine is set, else the deprecated EngineVersion.
func (s *Scenario) EngineRelease() string {
	if s.Engine != nil {
		return s.Engine.Version
	}
	return s.EngineVersion
}

// Cluster is the available-hardware inventory: the hardware a problem is handed and a
// deployment cannot change. Hardware and Fabric name catalog entries; Fabric is kept
// separate from Hardware because the inter-node bandwidth it carries is a cluster
// property, and the intra-to-inter ratio a collective pays is a property of the
// pairing. Storage completes the inventory — the device classes a deployment's offload
// layout may draw tiers from.
type Cluster struct {
	// Hardware names the catalog accelerator (GPU die) every node is populated with.
	// Folded in from the former loose top-level reference.
	Hardware string `yaml:"hardware"`
	// Fabric names the catalog inter-node network. Folded in from the former loose
	// top-level reference. Omitted for single-node clusters, where nothing crosses a
	// node boundary.
	Fabric string `yaml:"fabric,omitempty"`

	Nodes       int `yaml:"nodes"`
	GPUsPerNode int `yaml:"gpus_per_node"`
	// GPUsPerRack is stated only where a rack boundary is a distinct link from a
	// node boundary. Zero means the two-tier model applies.
	GPUsPerRack int `yaml:"gpus_per_rack,omitempty"`

	// Storage names the catalog storage-device classes provisioned in the cluster:
	// the inventory a deployment's offload layout may draw tiers from. Declaring it
	// here, rather than reaching it only through a deployment's offload block,
	// completes the available-hardware picture.
	Storage []string `yaml:"storage,omitempty"`
}

// GPUs returns the total device count.
func (c Cluster) GPUs() int { return c.Nodes * c.GPUsPerNode }
