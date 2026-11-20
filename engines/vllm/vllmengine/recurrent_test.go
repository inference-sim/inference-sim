// Original to blis-vllm.

package vllmengine

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
)

// A model with recurrent layers (Nemotron-3-Ultra's Mamba-2, Kimi-K3's KDA) builds, sizes its KV
// pool over vLLM's hybrid groups and runs requests to completion at vLLM's default settings: prefix
// caching on, so the recurrent cache runs in align mode, and the hybrid block-size alignment
// replaces the stated block size. Before the recurrent state was mapped to a MambaSpec these were
// refused with "recurrent state is not mapped to a KV cache spec". The Qwen fixtures are the
// cells the cluster matrix refused: a 256-wide attention head beside the recurrent layers
// (blis-vllm#309) and a QSA model whose pool page was derived from the memory model's bytes
// (blis-vllm#310).
func TestRecurrentModelsBuildAndRun(t *testing.T) {
	Register()
	for _, scenario := range []string{"recurrent-nemotron-3-ultra-h100-tp8.yaml", "recurrent-kimi-k3-h200-dep16.yaml",
		"recurrent-qwen3-8-flash-next-h200-tp4.yaml", "recurrent-qwen3-6-35b-a3b-h200-tp8.yaml", "recurrent-qwen3-8-27b-h200-tp4.yaml"} {
		t.Run(scenario, func(t *testing.T) {
			t.Parallel()
			spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.34.0+0fda86a35419", Config: poolEngine(t, scenario),
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
