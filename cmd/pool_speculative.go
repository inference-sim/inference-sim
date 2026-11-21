package cmd

import (
	"reflect"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/engines/vllm/core/engineconfig"
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

// sameDrafting reports whether two speculative blocks draft alike: no block and a block of no
// draft tokens both draft nothing.
func sameDrafting(a, b *engineconfig.Speculative) bool {
	tokens := func(s *engineconfig.Speculative) int {
		if s == nil {
			return 0
		}
		return s.NumSpecTokens
	}
	if tokens(a) == 0 && tokens(b) == 0 {
		return true
	}
	return reflect.DeepEqual(a, b)
}

// poolsDraftDifferently reports whether the P/D pools state different speculative blocks.
func poolsDraftDifferently(p *kernelPools) bool {
	if p == nil {
		return false
	}
	pre, err := p.prefill.Engine()
	if err != nil {
		logrus.Fatalf("reading the prefill pool's speculative settings: %v", err)
	}
	dec, err := p.decode.Engine()
	if err != nil {
		logrus.Fatalf("reading the decode pool's speculative settings: %v", err)
	}
	return !sameDrafting(pre.Speculative, dec.Speculative)
}

// resolvePoolSpeculation resolves each P/D pool's speculation from its own block (resolveSpeculation)
// and records it in poolSpeculation. It returns the run-wide config, the first drafting pool's,
// which names the run in the places that state one speculation. The command line's acceptance
// flags apply to every pool.
func resolvePoolSpeculation(cmd *cobra.Command, p *kernelPools) speculativeConfig {
	poolSpeculation = map[string]poolSpeculative{}
	var run speculativeConfig
	var have bool
	for _, pool := range []struct {
		role string
		m    *engineModel
	}{{"prefill", p.prefill}, {"decode", p.decode}} {
		s, err := pool.m.Settings()
		if err != nil {
			logrus.Fatalf("reading the %s pool's speculative settings: %v", pool.role, err)
		}
		var c speculativeConfig
		var rec *sim.AcceptanceRecord
		if s.SpeculativeTokens == 0 {
			// A pool that drafts nothing takes no acceptance (the other pool's selectors and
			// length are not refused for it).
			c = speculativeConfigOf(0, 0, s.SpeculativeMethod)
		} else {
			c, rec = resolveSpeculation(cmd, pool.m, s.SpeculativeTokens, s.SpeculativeMethod, " in the "+pool.role+" pool")
		}
		poolSpeculation[pool.role] = poolSpeculative{Config: c, Record: rec}
		if !have && c.IsEnabled() {
			run, have = c, true
		}
	}
	if !have {
		return poolSpeculation["prefill"].Config
	}
	return run
}

// adoptPoolSpeculation puts each pool's own speculation on its overrides when the pools draft
// differently; a no-op otherwise, so same-config pools and colocated runs are unchanged.
func adoptPoolSpeculation(prefill, decode *poolOverrides) {
	if poolSpeculation == nil {
		return
	}
	pre, dec := poolSpeculation["prefill"].Config, poolSpeculation["decode"].Config
	prefill.Speculative, decode.Speculative = &pre, &dec
}

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
