package kv

import "github.com/inference-sim/inference-sim/sim/internal/kvkey"

// secondaryTier models one ordered secondary offload tier (an "fs" tier). It
// tracks only which block keys it HOLDS — not token payloads: like the CPU tier,
// content is addressed by key and reload is driven by the requesting request's
// own prefix tokens. Secondary tiers are unbounded in H1 (#1587 defines no
// per-secondary capacity).
//
// Writes are eager parallel replicas across all tiers (BC-C7a): a block landing
// in CPU is copied to every secondary tier via fanOutSecondary. Lookups are
// ORDERED (tier 0 first) and stop at the first holding tier. A tier's own drop
// never touches another tier — eviction is not a spill-chain.
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

// fanOutSecondary writes key into every secondary tier (the write-through cascade
// target set, BC-C7a) and returns the tier indices written, in order.
func fanOutSecondary(tiers []*secondaryTier, key kvkey.BlockKey) []int {
	written := make([]int, 0, len(tiers))
	for i, tier := range tiers {
		tier.store(key)
		written = append(written, i)
	}
	return written
}
