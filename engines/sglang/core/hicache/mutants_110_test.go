// Original to blis-sglang.

package hicache

import (
	"slices"
	"testing"
)

// Behavioural tests of the layer counter and the storage-writer rule (blis-sglang#110): each
// fails on a generated mutant of counter.go or hicache.go that the golden traces do not reach.

func TestLayerCompleteRefusesAnyLayerOutsideTheEvents(t *testing.T) {
	clock := &Clock{}
	l := &LayerLoadingEvent{Start: NewEvent(clock), LoadEvents: []*Event{NewEvent(clock), NewEvent(clock)}}
	for _, layer := range []int{-1, -5, 2, 7} {
		func() {
			defer func() {
				if r := recover(); r != "hicache: layer index out of range" {
					t.Errorf("Complete(%d) of 2 layers panicked with %v, want the layer-range refusal", layer, r)
				}
			}()
			l.Complete(layer)
		}()
	}
	l.Complete(0)
	l.Complete(1)
	for i, e := range l.LoadEvents {
		if e.Query() {
			t.Errorf("layer %d event done before a boundary passed", i)
		}
	}
	clock.boundaries++
	if !l.Finish().Query() {
		t.Error("last layer event not done one boundary after Complete")
	}
}

func TestWaitIsRecordedFromTheFirstConsumerOnly(t *testing.T) {
	c := NewLayerDoneCounter(&Clock{}, 4)
	c.WaitUntil(1) // no consumer yet: nothing to wait for
	if len(c.Waits) != 0 {
		t.Fatalf("a wait with no consumer was recorded: %v", c.Waits)
	}
	c.SetConsumer(0) // the first event of the ring is a consumer too
	c.WaitUntil(2)
	c.SetConsumer(2)
	c.WaitUntil(3)
	if !slices.Equal(c.Waits, []int{2, 3}) {
		t.Fatalf("waits = %v, want [2 3]", c.Waits)
	}
	c.Reset()
	c.WaitUntil(0)
	if !slices.Equal(c.Waits, []int{2, 3}) {
		t.Fatalf("a wait after Reset was recorded: %v", c.Waits)
	}
}

func TestOnlyTheDCPShardLeaderOfAnMLAModelWritesStorage(t *testing.T) {
	for _, tc := range []struct {
		cfg  StorageConfig
		want bool
	}{
		{StorageConfig{IsMLAModel: false, TPRank: 0, DCPRank: 0}, true},
		{StorageConfig{IsMLAModel: false, TPRank: 3, DCPRank: 1}, true},
		{StorageConfig{IsMLAModel: true, TPRank: 2, DCPRank: 2}, true},
		{StorageConfig{IsMLAModel: true, TPRank: 3, DCPRank: 1}, false},
	} {
		if got := tc.cfg.IsStorageWriter(); got != tc.want {
			t.Errorf("%+v: storage writer = %v, want %v", tc.cfg, got, tc.want)
		}
	}
}
