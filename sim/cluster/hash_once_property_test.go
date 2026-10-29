package cluster

import (
	"fmt"
	"reflect"
	"testing"

	"pgregory.net/rapid"

	hash "github.com/inference-sim/blis-vllm/compat/prefix"
	sim "github.com/inference-sim/inference-sim/sim"
)

// INV-K11: each distinct prompt block is hashed at most once per engine, whatever the
// workload: random prompt lengths, shared prefixes, multi-turn growth, P/D on or off, a CPU
// tier on or off. The memoised run equals the uncached one exactly.
func TestHashOnce_EachPromptBlockHashedAtMostOncePerEngine(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		pd := rapid.Bool().Draw(rt, "pd")
		tier := rapid.Bool().Draw(rt, "tier")
		n := rapid.IntRange(2, 10).Draw(rt, "requests")
		var reqs []*sim.Request
		for i := 0; i < n; i++ {
			id := fmt.Sprintf("r%d", i)
			arrival := int64(i) * rapid.Int64Range(0, 3_000_000).Draw(rt, "gap")
			out := rapid.IntRange(1, 40).Draw(rt, "out")
			if i > 0 && rapid.Bool().Draw(rt, "extends") { // a later turn, or a shared prefix
				base := reqs[rapid.IntRange(0, i-1).Draw(rt, "base")]
				reqs = append(reqs, extend(id, base, arrival, rapid.IntRange(0, 200).Draw(rt, "extra"), out))
				continue
			}
			reqs = append(reqs, holdTestRequest(id, i+1, arrival, rapid.IntRange(1, 400).Draw(rt, "in"), out))
		}
		blocks := rapid.Int64Range(40, 200).Draw(rt, "blocks")
		run := func(off bool) (*ClusterSimulator, int64) {
			hash.SetChainCacheOff(off)
			defer hash.SetChainCacheOff(false)
			cfg := newTestDeploymentConfig(1)
			if pd {
				cfg = newTestDisaggDeploymentConfig(2, 1, 1)
				cfg.PDPrefillKVHold = true
				editPrefill(&cfg, func(c *simConfig) { c.TotalKVBlocks = blocks })
				if tier {
					editPrefill(&cfg, func(c *simConfig) { c.Offload = *cpuTier() })
				}
			} else {
				editEngine(&cfg, func(c *simConfig) { c.TotalKVBlocks = blocks })
				if tier {
					editEngine(&cfg, func(c *simConfig) { c.Offload = *cpuTier() })
				}
			}
			before := hash.PromptBlockHashes()
			cs := newTestCluster(cfg, NewSliceRequestSource(cloneRequests(reqs)), nil)
			if err := cs.Run(); err != nil {
				rt.Fatalf("run: %v", err)
			}
			return cs, hash.PromptBlockHashes() - before
		}
		ref, _ := run(true)
		got, hashes := run(false)
		for i := range ref.Instances() {
			if a, b := ref.Instances()[i].Metrics(), got.Instances()[i].Metrics(); !reflect.DeepEqual(a, b) {
				rt.Fatalf("instance %d: memoised metrics differ from the uncached run", i)
			}
		}
		distinct := map[string]bool{}
		for _, r := range reqs {
			for _, h := range hash.ComputeBlockHashes(16, tokenIDsOf(r.InputTokens)) {
				distinct[h] = true
			}
		}
		engines := int64(len(got.Instances()))
		if limit := engines * int64(len(distinct)); hashes > limit {
			rt.Fatalf("%d prompt block hashes for %d distinct prompt blocks on %d engines (limit %d)",
				hashes, len(distinct), engines, limit)
		}
	})
}
