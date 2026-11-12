// Original to blis-vllm.

package vllmengine

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// A model with recurrent layers (Nemotron-3-Ultra's Mamba-2, Kimi-K3's KDA) builds, sizes its KV
// pool over vLLM's hybrid groups and runs requests to completion at vLLM's default settings: prefix
// caching on, so the recurrent cache runs in align mode, and the hybrid block-size alignment
// replaces the stated block size. Before the recurrent state was mapped to a MambaSpec these were
// refused with "recurrent state is not mapped to a KV cache spec".
func TestRecurrentModelsBuildAndRun(t *testing.T) {
	Register()
	for _, scenario := range []string{"recurrent-nemotron-3-ultra-h100-tp8.yaml", "recurrent-kimi-k3-h200-dep16.yaml"} {
		t.Run(scenario, func(t *testing.T) {
			t.Parallel()
			spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Config: poolEngine(t, scenario),
				Roots: roots(t, scenario), Pool: 0}
			e, err := engine.New(spec)
			if err != nil {
				t.Fatal(err)
			}
			if e.Snapshot().KVBlocksTotal < 2 {
				t.Fatalf("a recurrent model's pool has %d blocks", e.Snapshot().KVBlocksTotal)
			}
			tr, err := enginetest.Drive(func() (engine.Engine, error) { return engine.New(spec) },
				[]enginetest.Arrival{{At: 0, Req: req("a", 512, 64)}, {At: 5000, Req: req("b", 3000, 32)}},
				enginetest.Options{})
			if err != nil {
				t.Fatal(err)
			}
			if len(tr.Finished) != 2 {
				t.Fatalf("finished %d of 2 requests", len(tr.Finished))
			}
		})
	}
}
