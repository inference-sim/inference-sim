package embed

import (
	"reflect"
	"testing"
	"time"
)

// TestOldBatchKeepsItsMeaning: a batch built with only the original fields leaves every
// later field at its zero value, which each field defines as absent.
func TestOldBatchKeepsItsMeaning(t *testing.T) {
	b := KVBatch{Timestamp: time.Unix(1, 0), Events: []KVEvent{
		BlockStored{Hashes: []uint64{1, 2}, Parent: 7, Tokens: []uint32{1, 2, 3, 4}, BlockSize: 2, Tier: "cpu"},
		BlockRemoved{Hashes: []uint64{1}, Tier: "cpu"},
	}}
	s := b.Events[0].(BlockStored)
	if s.LoRAID != nil || s.LoRAName != "" || s.ExtraKeys != nil || s.CacheSalt != "" ||
		s.GroupIdx != nil || s.SpecKind != "" || s.SlidingWindow != nil {
		t.Errorf("a stored event with the original fields has optional fields set: %+v", s)
	}
	if r := b.Events[1].(BlockRemoved); r.GroupIdx != nil {
		t.Errorf("GroupIdx = %v, want nil", *r.GroupIdx)
	}
	// The original fields still compose positionally by name.
	if s.Parent != 7 || s.BlockSize != 2 || s.Tier != "cpu" || len(s.Hashes) != 2 {
		t.Errorf("original fields changed: %+v", s)
	}
}

// TestNewFieldsRoundTrip: a batch carrying every new field keeps each one, whether the
// event is held by value or through a type switch.
func TestNewFieldsRoundTrip(t *testing.T) {
	id, g, w := 3, 1, 1024
	want := BlockStored{
		Hashes: []uint64{9}, BlockSize: 16, LoRAID: &id, LoRAName: "ad",
		ExtraKeys: [][]any{{"ad", "mm-1"}}, CacheSalt: "s",
		GroupIdx: &g, SpecKind: SpecSlidingWindow, SlidingWindow: &w,
	}
	var e KVEvent = want
	got, ok := e.(BlockStored)
	if !ok || !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip = %+v, want %+v", got, want)
	}
	if *got.LoRAID != 3 || *got.GroupIdx != 1 || *got.SlidingWindow != 1024 {
		t.Errorf("pointer fields changed: %+v", got)
	}
	// The zero value of a pointer field is distinct from a present zero.
	zero := 0
	if (BlockStored{GroupIdx: &zero}).GroupIdx == nil {
		t.Error("group 0 must be present")
	}
}

// TestAllBlocksClearedTier: the clear names a tier, or every tier at its zero value.
func TestAllBlocksClearedTier(t *testing.T) {
	var all KVEvent = AllBlocksCleared{}
	if all.(AllBlocksCleared).Tier != "" {
		t.Error("the zero clear must mean every tier")
	}
	one := KVEvent(AllBlocksCleared{Tier: "cpu"})
	if one.(AllBlocksCleared).Tier != "cpu" {
		t.Error("tier lost")
	}
}
