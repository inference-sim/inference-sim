package creation

import "github.com/inference-sim/inference-sim/sim"

// prePlacement seeds each instance's cluster-resolved adapter subset resident at
// t=0 (D3/D4), reproducing a static placement without any run-time cold load for
// the placed adapters (SC-002). It consumes ONLY the per-instance subset the
// cluster resolves into CreationContext.Assigned — it never sees the cross-instance
// placement map (Principle I). Seeding is uncharged (the caller Stores each id
// without a load-count increment or cold-load latency, INV-L3).
//
// Stateless. It trivially satisfies the starvation-freedom obligation on
// sim.CreationPolicy: OnResidentMiss always admits, so a request that misses on a
// non-seeded adapter still cold-loads on demand exactly as pre-B-5 — pre-placement
// only front-loads the declared subset, it does not restrict the miss path.
type prePlacement struct{}

// Initial returns a defensive copy of the assigned subset so a caller mutating the
// returned slice cannot corrupt the placement source. An empty (or nil) assignment
// returns nil, keeping the caller's uncharged-seed loop a clean no-op.
func (prePlacement) Initial(ctx sim.CreationContext) []string {
	if len(ctx.Assigned) == 0 {
		return nil
	}
	seed := make([]string, len(ctx.Assigned))
	copy(seed, ctx.Assigned)
	return seed
}

func (prePlacement) OnResidentMiss(sim.CreationContext) bool { return true }
