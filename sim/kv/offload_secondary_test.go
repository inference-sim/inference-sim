package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/internal/kvkey"
)

// BC-C6: secondary-tier lookup order is a deterministic function of tier index,
// independent of the order in which keys were inserted / requests arrived.
func TestSecondary_OrderedLookup(t *testing.T) {
	tiers := []*secondaryTier{newSecondaryTier(), newSecondaryTier(), newSecondaryTier()}
	kA, kB := cpuTestKey(1), cpuTestKey(2)

	// kA lives in tiers 1 and 2; kB only in tier 2. Insert in a scrambled order.
	tiers[2].store(kA)
	tiers[1].store(kA)
	tiers[2].store(kB)

	if idx, ok := lookupSecondary(tiers, kA); !ok || idx != 1 {
		t.Fatalf("kA must be found first in tier 1 (lowest holding index), got idx=%d ok=%v", idx, ok)
	}
	if idx, ok := lookupSecondary(tiers, kB); !ok || idx != 2 {
		t.Fatalf("kB must be found in tier 2, got idx=%d ok=%v", idx, ok)
	}
	if _, ok := lookupSecondary(tiers, cpuTestKey(9)); ok {
		t.Fatalf("absent key must miss all tiers")
	}

	// Determinism: repeated lookups return the same index regardless of call order.
	for i := 0; i < 100; i++ {
		if idx, _ := lookupSecondary(tiers, kA); idx != 1 {
			t.Fatalf("lookup not deterministic on iteration %d", i)
		}
	}
}

// storeAll records a key in every tier — the in-memory effect the write-through
// cascade produces once its per-tier Write jobs complete (BC-C7a). Production
// cascade lives in OffloadCache.cascade (submits the Write jobs) + SetClock
// (records the block on completion); this helper drives the tier struct directly.
func storeAll(tiers []*secondaryTier, k kvkey.BlockKey) {
	for _, tier := range tiers {
		tier.store(k)
	}
}

// BC-C7a: every secondary tier can hold a fanned-out key (eager parallel replicas).
func TestSecondary_FanOut(t *testing.T) {
	tiers := []*secondaryTier{newSecondaryTier(), newSecondaryTier()}
	k := cpuTestKey(7)
	storeAll(tiers, k)
	for i, tier := range tiers {
		if !tier.has(k) {
			t.Fatalf("tier %d must hold the fanned-out key", i)
		}
	}
}

// BC-C7a (negative): a tier's own drop never mutates another tier's holdings —
// eviction is not a spill-chain.
func TestSecondary_DropIsIsolated(t *testing.T) {
	tiers := []*secondaryTier{newSecondaryTier(), newSecondaryTier()}
	k := cpuTestKey(3)
	storeAll(tiers, k)
	tiers[0].drop(k)
	if tiers[0].has(k) {
		t.Fatalf("dropped key must be gone from tier 0")
	}
	if !tiers[1].has(k) {
		t.Fatalf("tier 1 must be untouched by tier 0's drop (no spill-chain)")
	}
}
