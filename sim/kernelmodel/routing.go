package kernelmodel

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	blisschemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/spec/routing"
	"github.com/inference-sim/blis-schemas/spec/workload"

	sim "github.com/inference-sim/inference-sim/sim/engine"
)

// Expert-parallel routing imbalance is resolved here, per pool, by blis-schemas'
// routing.Resolve from five layers, lowest to highest: the documented default (evenly
// loaded), the registry's default-role routing/ sets for (model, workload category, phase,
// width), the workload's own value, the scenario's pin, and the command line. The kernel
// prices the busiest rank from the answer; this file only gathers the layers.

// registryRoutingDir is the registry's routing-profile namespace, beside coefficients/ and
// acceptance/.
const registryRoutingDir = "routing"

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
func resolveRouting(in latencykernel.Inputs, registry string, o RoutingOverrides) (*routing.Resolution, error) {
	pool := in.Deployment.Pools[in.PoolIndex]
	req := routing.Request{
		Model:          in.Scenario.Model,
		ExpertParallel: pool.Parallel.ExpertParallelWidth(),
		EPLB:           pool.Engine.EPLB != nil && pool.Engine.EPLB.Enabled,
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
	sets, err := loadRegistryRoutingSets(registry)
	if err != nil {
		return nil, err
	}
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

// loadRegistryRoutingSets loads and validates every routing-profile set under the
// registry's routing/ namespace, in path order. No namespace means no sets; a set that
// fails to load or validate is an error, never skipped.
func loadRegistryRoutingSets(registry string) ([]*routing.ProfileSet, error) {
	if registry == "" {
		return nil, nil
	}
	dir := filepath.Join(registry, registryRoutingDir)
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	var paths []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.Type().IsRegular() && (strings.HasSuffix(p, ".yaml") || strings.HasSuffix(p, ".yml")) {
			paths = append(paths, p)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", dir, err)
	}
	sort.Strings(paths)
	sets := make([]*routing.ProfileSet, 0, len(paths))
	for _, p := range paths {
		s, err := blisschemas.LoadRoutingProfileSet(p)
		if err != nil {
			return nil, fmt.Errorf("routing profile set %s: %w", p, err)
		}
		if problems := s.Validate(); !problems.OK() {
			return nil, fmt.Errorf("routing profile set %s: %s", p, problems.Error())
		}
		sets = append(sets, s)
	}
	return sets, nil
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
