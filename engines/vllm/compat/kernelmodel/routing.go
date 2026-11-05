// Moved from inference-sim/sim/kernelmodel/routing.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
	latencykernel "github.com/inference-sim/blis-substrate/kernel"
	"github.com/inference-sim/blis-substrate/load"

	sim "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// Expert-parallel routing imbalance is resolved here, per pool, by blis-schemas'
// routing.Resolve from five layers, lowest to highest: the documented default (evenly
// loaded), the registry's default-role routing/ sets for (model, workload category, phase,
// width), the workload's own value, the scenario's pin, and the command line. The kernel
// prices the busiest rank from the answer; this file only gathers the layers.

// RoutingOverrides are the layers a caller supplies beyond the scenario file: a command
// line value and a workload that is not the scenario's own binding (a catalog preset).
type RoutingOverrides struct {
	CLI      *routing.Value
	Workload *routing.Workload
}

// routingOverrides is set by the command layer before a pool opens (SetRoutingOverrides).
var routingOverrides RoutingOverrides

// SetRoutingOverrides records the caller's routing layers for the pools opened after it.
func SetRoutingOverrides(o RoutingOverrides) { routingOverrides = o }

// resolveRouting resolves one pool's routing imbalance. A miss falls to the default and is
// not an error; a scenario's named set that is missing or lacks the key is.
func resolveRouting(in latencykernel.Inputs, engine engineconfig.Engine, data *load.Data, o RoutingOverrides) (*routing.Resolution, error) {
	pool := in.Deployment.Pools[in.PoolIndex]
	req := routing.Request{
		Model:          in.Scenario.Model,
		ExpertParallel: pool.Parallel.ExpertParallelWidth(),
		EPLB:           engine.EPLB != nil && engine.EPLB.Enabled,
		Hardware:       in.Chip.Name,
		Workload:       bindingRoutingTerms(in.Scenario.Workload),
		Scenario:       in.Scenario.Routing,
		CLI:            o.CLI,
	}
	if o.Workload != nil {
		req.Workload = *o.Workload
	}
	if req.ExpertParallel < 2 {
		r, err := routing.Resolve(req)
		return &r, err
	}
	sets := data.RoutingSets()
	req.Registry = sets
	if pin := req.Scenario; pin != nil && pin.Profiles != "" {
		for _, s := range sets {
			if s.Name == pin.Profiles {
				req.ScenarioSet = s
			}
		}
	}
	r, err := routing.Resolve(req)
	if err != nil {
		return nil, fmt.Errorf("routing imbalance: %w", err)
	}
	data.RecordRouting(&r)
	return &r, nil
}

func bindingRoutingTerms(b *workload.Binding) routing.Workload {
	switch {
	case b == nil:
		return routing.Workload{}
	case b.Trace != nil:
		return b.Trace.RoutingTerms()
	case b.Replay != nil:
		return b.Replay.RoutingTerms()
	}
	return routing.Workload{}
}

// routingRecordOf is the results-file record of a resolution; nil when none was made.
func routingRecordOf(r *routing.Resolution) *sim.RoutingRecord {
	if r == nil {
		return nil
	}
	rec := &sim.RoutingRecord{SourceLayer: string(r.Layer), Set: r.Set, Source: r.Source,
		Reason: r.Reason, MeasuredOn: r.MeasuredOn, FallbackFrom: r.FallbackFrom}
	for _, p := range []*routing.Profile{r.Decode, r.Prefill} {
		if p != nil {
			rec.Keys = append(rec.Keys, p.Key().String())
		}
	}
	if v := r.Value; v != nil {
		rec.TokenImbalance, rec.ExpertImbalance = v.TokenImbalance, v.Expert()
	}
	return rec
}

// CurrentRoutingOverrides returns the layers SetRoutingOverrides last recorded.
func CurrentRoutingOverrides() RoutingOverrides { return routingOverrides }
