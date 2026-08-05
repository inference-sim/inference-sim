package creation

import "github.com/inference-sim/inference-sim/sim"

// onDemand is the shipped default creation policy — byte-identical to pre-B-5
// behavior (INV-L1). It seeds nothing at t=0 (Initial returns empty) and admits
// every cold load (OnResidentMiss always true), so residency is driven purely by
// the on-request cold-load gate exactly as before the creation seam existed.
// Stateless. It trivially satisfies the starvation-freedom obligation on
// sim.CreationPolicy because it never returns false from OnResidentMiss.
type onDemand struct{}

func (onDemand) Initial(sim.CreationContext) []string { return nil }

func (onDemand) OnResidentMiss(sim.CreationContext) bool { return true }
