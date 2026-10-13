package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// An injected tier ServiceTime is asked for the direction of the job it prices: a store
// (the write-through cascade, CPU -> secondary) is priced with write=true and a load (a
// promotion, secondary -> CPU) with write=false, each for the job's bytes. The glue in
// NewOffloadCache translates the station's kvtransfer.Direction into that flag; a swapped
// translation would price every store as a load and vice versa.
func TestOffload_ServiceTimeIsAskedForTheJobsDirection(t *testing.T) {
	type call struct {
		write bool
		bytes int64
	}
	var calls []call
	cfg := enabledOffloadCfg(1<<20, 4096, 1)
	cfg.Tiers[0].ServiceTime = func(write bool, bytes int64, inService int) int64 {
		if inService < 1 {
			t.Errorf("priced at in-service depth %d; a job in service is at least depth 1", inService)
		}
		calls = append(calls, call{write, bytes})
		return 10
	}
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, cfg)
	oc.SetClock(1)

	stored := blockKeysFor([]sim.TokenID{7, 8}, 2)[0]
	if !oc.cpu.store(stored) {
		t.Fatal("could not seed the CPU tier")
	}
	oc.cascade(stored)
	if len(calls) == 0 {
		t.Fatal("a store was never priced by the tier's ServiceTime")
	}
	for _, c := range calls {
		if !c.write || c.bytes != 4096 {
			t.Errorf("a store was priced as write=%t for %d bytes; want write=true for one 4096-byte block", c.write, c.bytes)
		}
	}

	calls = nil
	tokens := []sim.TokenID{1, 2, 3, 4}
	for _, k := range blockKeysFor(tokens, 2) {
		oc.secondary[0].store(k)
	}
	oc.consultAndReload(tokens, 0, "")
	if oc.promotionsFired != 1 || len(calls) == 0 {
		t.Fatalf("a load was not priced (promotions %d, priced %d)", oc.promotionsFired, len(calls))
	}
	for _, c := range calls {
		if c.write || c.bytes <= 0 {
			t.Errorf("a load was priced as write=%t for %d bytes; want write=false for its blocks", c.write, c.bytes)
		}
	}
}
