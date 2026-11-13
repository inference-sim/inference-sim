package vllm

import (
	"testing"

	"github.com/inference-sim/blis-substrate/load"
)

// recordingPricer counts the Data it is registered with.
type recordingPricer struct {
	FixedPricer
	got []*load.Data
}

func (p *recordingPricer) RecordSelected(d *load.Data) { p.got = append(p.got, d) }

// Engine.RecordSelected hands the pricer the host's Data once, and does nothing for a nil Data
// or a pricer that has no such method.
func TestEngineRecordSelectedRegistersThePricer(t *testing.T) {
	cfg, err := decodeConfig(block(t, "max_num_batched_tokens: 32\n"))
	if err != nil {
		t.Fatal(err)
	}
	st, err := resolve(cfg)
	if err != nil {
		t.Fatal(err)
	}
	p := &recordingPricer{FixedPricer: FixedPricer(100)}
	e, err := newEngine(st, p)
	if err != nil {
		t.Fatal(err)
	}
	host := &load.Data{}
	e.RecordSelected(nil)
	if len(p.got) != 0 {
		t.Fatalf("a nil Data registered the pricer %d times", len(p.got))
	}
	e.RecordSelected(host)
	if len(p.got) != 1 || p.got[0] != host {
		t.Fatalf("the pricer was registered with %v, want the host's Data once", p.got)
	}
	plain, err := newEngine(st, FixedPricer(100))
	if err != nil {
		t.Fatal(err)
	}
	plain.RecordSelected(host) // a pricer that selects nothing records nothing, and no error
}

// A data-parallel group shares one pricer across its ranks, so it registers it once.
func TestDPGroupRecordSelectedRegistersTheSharedPricerOnce(t *testing.T) {
	cfg, err := decodeConfig(block(t, "max_num_batched_tokens: 32\ndata_parallel_size: 2\n"))
	if err != nil {
		t.Fatal(err)
	}
	st, err := resolve(cfg)
	if err != nil {
		t.Fatal(err)
	}
	p := &recordingPricer{FixedPricer: FixedPricer(100)}
	g, err := newDPGroup(st, p)
	if err != nil {
		t.Fatal(err)
	}
	if len(g.ranks) != 2 {
		t.Fatalf("group has %d ranks, want 2", len(g.ranks))
	}
	host := &load.Data{}
	g.RecordSelected(nil)
	g.RecordSelected(host)
	if len(p.got) != 1 || p.got[0] != host {
		t.Fatalf("the shared pricer was registered %d times, want once with the host's Data", len(p.got))
	}
}
