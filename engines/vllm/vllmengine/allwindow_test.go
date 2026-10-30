// Original to blis-vllm.

package vllmengine

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"

	"github.com/inference-sim/blis-vllm/core/vllm"
)

// A model whose every KV layer is bounded -- muse-spark-open-dspark's five attention layers are
// all sliding_window 2048 -- has no growing-context page (SequenceVariableBytes 0), and vLLM
// still starts it: its pool is cut from the window groups' own pages
// (get_kv_cache_config_from_groups, vllm/v1/core/kv_cache_utils.py:1622-1628@5df63a72), at pp 1
// and, per pipeline rank, at pp 2 (get_kv_cache_configs, :2595-2680). The engine sizes it from
// those pages, its block count is the binding rank's allocatable bytes over that rank's bytes a
// block, and it runs a request through, at both. Before, both refused with "per-block KV
// resolved to 0 bytes".
func TestAnAllWindowModelSizesItsPoolFromItsGroupsPages(t *testing.T) {
	Register()
	for _, c := range []struct {
		scenario string
		pp       int
	}{{"muse-h100-tp8-allwindow.yaml", 1}, {"muse-h100-pp2tp8-allwindow.yaml", 2}} {
		t.Run(c.scenario, func(t *testing.T) {
			// As open opens it: SettingsCheckedByGroups, not the compatibility engine's Settings,
			// whose one full-attention group an all-window model has none of.
			p, err := open(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, c.scenario)})
			if err != nil {
				t.Fatalf("open: %v", err)
			}
			if got := p.model.Kernel().SequenceVariableBytes(16); got != 0 {
				t.Fatalf("SequenceVariableBytes(16) = %d; the fixture must be all-window (0)", got)
			}
			if got := p.model.PipelineStages(); got != c.pp {
				t.Fatalf("pp %d, want %d", got, c.pp)
			}
			runs, err := p.model.KVLayers()
			if err != nil || len(runs) == 0 {
				t.Fatalf("KV layers %v, %v", runs, err)
			}
			sp, err := stagePoolsOf(p.model, runs)
			if err != nil {
				t.Fatalf("stagePoolsOf: %v", err)
			}
			if sp.blocks <= 0 || sp.perBlock[sp.binding] <= 0 {
				t.Fatalf("%d blocks of %d bytes on binding rank %d", sp.blocks, sp.perBlock[sp.binding], sp.binding)
			}
			if want := sp.avail[sp.binding] / sp.perBlock[sp.binding]; sp.blocks != want {
				t.Errorf("%d blocks, want the binding rank's %d bytes over its %d a block = %d",
					sp.blocks, sp.avail[sp.binding], sp.perBlock[sp.binding], want)
			}
			if per, err := offloadBytesPerBlock(p.model); err != nil || per != sp.perBlock[0] {
				t.Errorf("offload bytes a block %d (%v), want rank 0's %d", per, err, sp.perBlock[0])
			}

			spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, c.scenario)}
			done, err := prepare(spec)
			if err != nil {
				t.Fatalf("prepare: %v", err)
			}
			var block struct {
				Layers []vllm.KVLayerRun `yaml:"kv_layers"`
			}
			if err := done.Config.Decode(&block); err != nil {
				t.Fatal(err)
			}
			if len(block.Layers) == 0 {
				t.Fatal("the completed engine block states no kv_layers")
			}
			tr, err := enginetest.Drive(func() (engine.Engine, error) { return engine.New(spec) },
				[]enginetest.Arrival{{At: 0, Req: req("a", 4096, 16)}}, enginetest.Options{})
			if err != nil {
				t.Fatal(err)
			}
			if len(tr.Finished) != 1 {
				t.Fatalf("finished %d of 1 request", len(tr.Finished))
			}
		})
	}
}
