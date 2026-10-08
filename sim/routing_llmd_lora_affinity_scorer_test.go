package sim

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Tier table for the faithful core, against llm-d-router d4b8afd3
// loraaffinity.LoraAffinityScorer.Score. active/waiting are passed separately here
// so the 0.6 tier — unreachable through the snapshot, where both are
// ActiveAdapters — is exercised.
func TestLLMDLoRAAffinityTier_LLMDSemantics(t *testing.T) {
	set := func(ids ...string) map[string]int {
		m := make(map[string]int, len(ids))
		for _, id := range ids {
			m[id] = 0 // llm-d stores 0 for every key
		}
		return m
	}
	cases := []struct {
		name            string
		target          string
		active, waiting map[string]int
		maxActive       int
		want            float64
	}{
		// Tier 1.0: target active — wins even when the server is full or over-full.
		{"active, free slots", "A", set("A"), set("A"), 4, 1.0},
		{"active, full", "A", set("A", "B"), set("A", "B"), 2, 1.0},
		{"active, over-full", "A", set("A", "B", "C"), nil, 2, 1.0},
		{"active, max 0", "A", set("A"), nil, 0, 1.0},
		// Tier 0.8: not active, union below max.
		{"cold, empty server", "A", nil, nil, 4, 0.8},
		{"cold, one below max", "A", set("B", "C", "D"), set("B"), 4, 0.8},
		// Boundary: union == max is NOT < max ⇒ falls through.
		{"cold, union == max", "A", set("B", "C"), set("B", "C"), 2, 0.0},
		// Union de-duplicates: active {B}, waiting {B,C} ⇒ 2, not 3.
		{"union dedup below max", "A", set("B"), set("B", "C"), 3, 0.8},
		{"union counts waiting-only keys", "A", set("B"), set("C"), 2, 0.0},
		// Tier 0.6: waiting only (not active), union at/over max.
		{"waiting only, full", "A", set("B"), set("A"), 2, 0.6},
		// waiting-only but slots free ⇒ 0.8 precedes 0.6 (switch order).
		{"waiting only, free slots", "A", set("B"), set("A"), 3, 0.8},
		// Tier 0.0.
		{"cold, full", "A", set("B", "C"), nil, 2, 0.0},
		{"LoRA off (max 0, nothing active)", "A", nil, nil, 0, 0.0},
		// Base-model request: "" is never a key ⇒ behaves as cold.
		{"base model, free slots", "", set("B"), set("B"), 4, 0.8},
		{"base model, full", "", set("B", "C"), set("B", "C"), 2, 0.0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, llmdLoRAAffinityTier(tc.target, tc.active, tc.waiting, tc.maxActive))
		})
	}
}

// Scorer level: one snapshot per reachable tier; raw tiers, no normalization.
func TestLLMDLoRAAffinity_ScorerTiers(t *testing.T) {
	snaps := []RoutingSnapshot{
		{ID: "active", ActiveAdapters: map[string]int{"A": 2, "B": 1}, MaxLoras: 2},
		{ID: "free", ActiveAdapters: map[string]int{"B": 5}, MaxLoras: 2},
		{ID: "full", ActiveAdapters: map[string]int{"B": 1, "C": 1}, MaxLoras: 2},
		{ID: "empty", MaxLoras: 2},
	}
	got := scoreLLMDLoRAAffinity(&Request{ID: "r", Adapter: "A"}, snaps)
	assert.Equal(t, map[string]float64{"active": 1.0, "free": 0.8, "full": 0.0, "empty": 0.8}, got)

	// Base model: never active ⇒ 0.8 with free slots, 0.0 when full (llm-d parity,
	// NOT neutral).
	gotBase := scoreLLMDLoRAAffinity(&Request{ID: "b"}, snaps)
	assert.Equal(t, map[string]float64{"active": 0.0, "free": 0.8, "full": 0.0, "empty": 0.8}, gotBase)

	// nil request is treated as a base-model request (no panic).
	assert.Equal(t, gotBase, scoreLLMDLoRAAffinity(nil, snaps))
}

// The scorer must follow the router-observable ActiveAdapters, never the
// ground-truth ResidentAdapters: each snapshot's ResidentAdapters contradicts its
// ActiveAdapters, and the score must follow ActiveAdapters.
func TestLLMDLoRAAffinity_IgnoresResidentAdapters(t *testing.T) {
	snaps := []RoutingSnapshot{
		// Resident but idle, server full of other active adapters: real router sees
		// no A ⇒ 0.0 (a residency reader would say 1.0).
		{ID: "resident-idle", ResidentAdapters: map[string]bool{"A": true},
			ActiveAdapters: map[string]int{"B": 1, "C": 1}, MaxLoras: 2},
		// Queued behind a cold load, not yet resident: router sees A ⇒ 1.0.
		{ID: "queued-cold", ResidentAdapters: map[string]bool{"B": true, "C": true},
			ActiveAdapters: map[string]int{"A": 1}, MaxLoras: 2},
	}
	got := scoreLLMDLoRAAffinity(&Request{ID: "r", Adapter: "A"}, snaps)
	assert.Equal(t, 0.0, got["resident-idle"])
	assert.Equal(t, 1.0, got["queued-cold"])

	// Changing ONLY ResidentAdapters must not change any score.
	for i := range snaps {
		snaps[i].ResidentAdapters = nil
	}
	assert.Equal(t, got, scoreLLMDLoRAAffinity(&Request{ID: "r", Adapter: "A"}, snaps))
}

// Uniform 0 when LoRA is off: cannot move an argmax.
func TestLLMDLoRAAffinity_InertWhenLoRAOff(t *testing.T) {
	snaps := []RoutingSnapshot{{ID: "i0"}, {ID: "i1"}}
	got := scoreLLMDLoRAAffinity(&Request{ID: "r", Adapter: "A"}, snaps)
	assert.Equal(t, map[string]float64{"i0": 0, "i1": 0}, got)
}

// Wired into the weighted router by name: the tier-1.0 instance wins.
func TestLLMDLoRAAffinity_WeightedRoutingByName(t *testing.T) {
	require.True(t, IsValidScorer("llmd-lora-affinity"))
	policy := NewRoutingPolicy("weighted", []ScorerConfig{{Name: "llmd-lora-affinity", Weight: 1}}, 16, nil)
	state := &RouterState{Snapshots: []RoutingSnapshot{
		{ID: "full", ActiveAdapters: map[string]int{"B": 1, "C": 1}, MaxLoras: 2},
		{ID: "free", MaxLoras: 2},
		{ID: "active", ActiveAdapters: map[string]int{"A": 1, "B": 1}, MaxLoras: 2},
	}}
	d := policy.Route(&Request{ID: "r", Adapter: "A"}, state)
	assert.Equal(t, "active", d.TargetInstance)
	assert.Equal(t, map[string]float64{"full": 0, "free": 0.8, "active": 1.0}, d.Scores)
}
