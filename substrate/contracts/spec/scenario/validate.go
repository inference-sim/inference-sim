package scenario

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// Validate performs FIELD-LEVEL validation: required fields, positive counts, and the
// structural invariants that hold whatever engine version a scenario names.
//
// A structural invariant is one that follows from what the problem is, not from an
// engine's current choices. That a cluster declaring more than one node needs a fabric
// to price cross-node cost is such an invariant. Version-scoped rules — whether a
// backend name is supported, what the collective thresholds default to — do not belong
// here; they move between releases, and a rules pack keyed by engine version owns them.
//
// The mutable configuration a deployment chooses against this problem — pools,
// parallelism, engine knobs, offload, PD transfer — is validated by the Deployment
// type (spec/deployment), not here.
func (s *Scenario) Validate() *validate.Problems {
	p := &validate.Problems{}

	if s.Kind != "Scenario" {
		p.Field("kind", "must be %q, got %q", "Scenario", s.Kind)
	}
	for field, v := range map[string]string{"name": s.Name, "model": s.Model} {
		if v == "" {
			p.Field(field, "required")
		}
	}
	s.validateEngine(p)
	if len(s.Coefficients) == 0 {
		p.Field("coefficients",
			"at least one set is required; an estimate with no coefficients is not an estimate")
	}
	// A set is applied in order and named once: a blank entry names no registry set, and
	// a duplicate is a copy-paste that would apply the same set twice.
	seen := map[string]bool{}
	for i, name := range s.Coefficients {
		at := fmt.Sprintf("coefficients[%d]", i)
		if name == "" {
			p.Field(at, "names a registry coefficient set and must not be empty")
		} else if seen[name] {
			p.Field(at, "duplicate coefficient set %q", name)
		}
		seen[name] = true
	}

	// A workload binding is optional (a capacity-only scenario carries none), but a
	// binding that is present must be well formed: exactly one of its two arms, and that
	// arm valid. The exactly-one check lives on the binding, not here.
	if s.Workload != nil {
		p.Merge("workload", s.Workload.Validate())
	}
	if s.Acceptance != nil {
		p.Merge("acceptance", s.Acceptance.Validate())
	}
	if s.Routing != nil {
		p.Merge("routing", s.Routing.Validate())
	}

	s.validateCluster(p)
	return p
}

// validateEngine checks the engine a scenario names (SIM-1). Until M4 the deprecated
// engine_version is accepted alone, or beside engine with the same version (plan §3.3).
func (s *Scenario) validateEngine(p *validate.Problems) {
	e := s.Engine
	switch {
	case e == nil && s.EngineVersion == "":
		p.Field("engine", "required: the engine that runs the scenario, as engine: {kind, version}")
		return
	case e == nil:
		return
	}
	if e.Kind == "" {
		p.Field("engine.kind", "required: the engine kind, for example vllm")
	}
	if e.Version == "" {
		p.Field("engine.version", "required: the upstream release the engine models, for example 0.32.0")
	}
	if s.EngineVersion != "" && e.Version != "" && s.EngineVersion != e.Version {
		p.Field("engine_version", "is %q but engine.version is %q: engine_version is deprecated; state the release once, in engine.version",
			s.EngineVersion, e.Version)
	}
}

func (s *Scenario) validateCluster(p *validate.Problems) {
	p.Merge("cluster", s.Cluster.Validate())
}

// Validate performs field-level validation of a cluster inventory on its own, for a
// document that states a cluster without the rest of a scenario: a measured run records
// the cluster it ran on, and the coefficients a scenario names are BLIS's choice, not a
// fact of the measurement.
//
// It does not check the cluster's packaging against its chip, which needs the chip's
// catalog entry: Packaging does, and every validator that holds the entry calls it.
func (c Cluster) Validate() *validate.Problems {
	p := &validate.Problems{}
	if c.Hardware == "" {
		p.Field("hardware",
			"required: names the catalog accelerator the cluster is built from")
	}
	if c.Nodes < 1 {
		p.Field("nodes", "must be at least 1")
	}
	// gpus_per_node and gpus_per_rack may be omitted: whether a value is required, and
	// whether a stated one is accepted, depends on the chip's catalog entry, which only
	// Packaging sees. What can be refused without the entry is a value no entry accepts.
	if c.GPUsPerNode < 0 {
		p.Field("gpus_per_node", "must be at least 1 when stated, got %d", c.GPUsPerNode)
	}
	if c.GPUsPerRack < 0 {
		p.Field("gpus_per_rack", "must be at least 1 when stated, got %d", c.GPUsPerRack)
	}
	if c.GPUsPerRack > 0 && c.GPUsPerNode > 0 && c.GPUsPerRack%c.GPUsPerNode != 0 {
		p.Field("gpus_per_rack", "%d is not a multiple of gpus_per_node %d",
			c.GPUsPerRack, c.GPUsPerNode)
	}
	// A multi-node cluster moves bytes between nodes, and the bandwidth for that is a
	// fabric property. Without one, every cross-node cost is unpriced.
	if c.Nodes > 1 && c.Fabric == "" {
		p.Field("fabric",
			"required for a %d-node cluster: cross-node cost resolves from the fabric, not the chip",
			c.Nodes)
	}
	// Storage is optional inventory, but a named class must be nameable and listed
	// once: a blank entry names nothing, and a duplicate is a copy-paste that would
	// make a deployment's tier reference ambiguous.
	seen := map[string]bool{}
	for i, dev := range c.Storage {
		at := fmt.Sprintf("storage[%d]", i)
		if dev == "" {
			p.Field(at, "names a catalog storage-device class and must not be empty")
		} else if seen[dev] {
			p.Field(at, "duplicate storage class %q", dev)
		}
		seen[dev] = true
	}
	return p
}
