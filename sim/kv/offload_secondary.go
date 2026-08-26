package kv

import "github.com/inference-sim/inference-sim/sim/internal/kvkey"

// secondaryTier models one ordered secondary offload tier (an "fs" tier). It
// tracks only which block keys it HOLDS — not token payloads: like the CPU tier,
// content is addressed by key and reload is driven by the requesting request's
// own prefix tokens. Secondary tiers are unbounded in H1 (#1587 defines no
// per-secondary capacity).
//
// Writes are eager parallel replicas across all tiers (BC-C7a): OffloadCache.cascade
// submits a Write job to every secondary tier, and each tier records the block only
// when its Write job completes (SetClock). Lookups are ORDERED (tier 0 first) and
// stop at the first holding tier. A tier's own drop never touches another tier —
// eviction is not a spill-chain.
type secondaryTier struct {
	holds map[kvkey.BlockKey]struct{}
}

func newSecondaryTier() *secondaryTier {
	return &secondaryTier{holds: make(map[kvkey.BlockKey]struct{})}
}

func (s *secondaryTier) store(key kvkey.BlockKey) { s.holds[key] = struct{}{} }
func (s *secondaryTier) drop(key kvkey.BlockKey)  { delete(s.holds, key) }

func (s *secondaryTier) has(key kvkey.BlockKey) bool {
	_, ok := s.holds[key]
	return ok
}

// lookupSecondary returns the index of the lowest-indexed tier that holds key, or
// (0, false) if no tier holds it. Order depends only on tier index, never on
// arrival order (BC-C6, INV-6). It is a free function over the tier slice so it
// carries no reference to the composing OffloadCache.
func lookupSecondary(tiers []*secondaryTier, key kvkey.BlockKey) (int, bool) {
	for i, tier := range tiers {
		if tier.has(key) {
			return i, true
		}
	}
	return 0, false
}
