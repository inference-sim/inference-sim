package workload

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// ValidateAdapterReferences cross-checks every adapter id referenced by the workload
// spec (clients and cohorts) against the declared adapter registry (#1464, US1):
//
//   - a non-empty adapter id MUST be a registry key (completeness), and
//   - if the adapter declares a base model, it MUST equal the referencing
//     client/cohort model (an adapter cannot attach to the wrong base model).
//
// An omitted adapter ("") is always valid — a base-model-only request, the no-op
// default (INV-6). A nil registry means no adapters are configured; in that case any
// non-empty reference is an error (a workload cannot reference adapters that were
// never declared). With no references and no registry the spec is inert.
//
// This is a pure validation query (no mutation). The CLI boundary decides fatality
// (cmd/ -> logrus.Fatalf); the library returns an error.
func ValidateAdapterReferences(spec *WorkloadSpec, reg sim.AdapterRegistry) error {
	if spec == nil {
		return nil
	}

	check := func(kind, id, adapter, model string) error {
		if adapter == "" {
			return nil // base-model-only request
		}
		if reg == nil {
			return fmt.Errorf("%s %q references adapter %q but no adapters are declared", kind, id, adapter)
		}
		if !reg.Has(adapter) {
			return fmt.Errorf("%s %q references unknown adapter %q (not in registry)", kind, id, adapter)
		}
		if baseModel, _ := reg.BaseModelOf(adapter); baseModel != "" && baseModel != model {
			return fmt.Errorf("%s %q model %q does not match adapter %q base model %q",
				kind, id, model, adapter, baseModel)
		}
		return nil
	}

	for i := range spec.Clients {
		c := &spec.Clients[i]
		if err := check("client", c.ID, c.Adapter, c.Model); err != nil {
			return err
		}
	}
	for i := range spec.Cohorts {
		h := &spec.Cohorts[i]
		if err := check("cohort", h.ID, h.Adapter, h.Model); err != nil {
			return err
		}
	}
	return nil
}
