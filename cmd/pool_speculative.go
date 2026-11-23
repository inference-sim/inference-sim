package cmd

import (
	"github.com/inference-sim/inference-sim/sim"
)

// Per-pool speculative decoding (inference-sim#2125). vLLM takes each server's own
// speculative_config and the P/D connector's fingerprint carries none of its fields, so a prefill
// pool and a decode pool may draft differently (an MTP of 1 token in prefill and 3 in decode). Each
// pool's engine already drafts by its own scenario block; what is run-wide is the host's reading of
// it: the acceptance that advances a request, and the Result's acceptance record. When the pools
// state the same speculation the run resolves it once, as it always has; when they differ each
// pool resolves its own.

// poolSpeculative is one P/D pool's resolved speculation.
type poolSpeculative struct {
	Config speculativeConfig
	// Record is the pool's resolved acceptance, nil when the pool drafts nothing.
	Record *sim.AcceptanceRecord
}

// poolSpeculation is each P/D pool's speculation by role ("prefill", "decode"), set only for a run
// whose pools draft differently; nil otherwise (the run-wide config and acceptanceRecord apply).
var poolSpeculation map[string]poolSpeculative

// poolAcceptanceRecords are the per-role acceptance records of a run whose pools draft differently,
// for the Result (a pool that drafts nothing has none); nil otherwise.
func poolAcceptanceRecords() map[string]*sim.AcceptanceRecord {
	var out map[string]*sim.AcceptanceRecord
	for role, ps := range poolSpeculation {
		if ps.Record == nil {
			continue
		}
		if out == nil {
			out = map[string]*sim.AcceptanceRecord{}
		}
		out[role] = ps.Record
	}
	return out
}
