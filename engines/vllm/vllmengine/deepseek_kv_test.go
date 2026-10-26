// Original to blis-vllm.

package vllmengine

import (
	"reflect"
	"strconv"
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"

	"github.com/inference-sim/blis-vllm/core/kv"
	vllm "github.com/inference-sim/blis-vllm/core/vllm"
)

// DeepSeek V4's and V4.1's KV cache layers, mapped from the catalog's model graph, group as vLLM
// v0.32.0 groups them: the groups core/kv's TestPackedGroupsAreVLLMs pins as vLLM's own
// get_kv_cache_groups output for those layers, from its own spec classes, and the drafter group
// TestPackedGroupsFlagTheDrafterGroup pins. deepseek-v4-pro has two windows at 64, the compressed
// caches at 256 and the two compressor states at 8 and 4, with MTP's layer in the second window
// group; deepseek-v4.1-flash at block 64 with DSpark eight windows at 32 (bounded replay on, as
// the V2 runner DSpark needs leaves it), the compressed caches at 64 and the ring, with DSpark's
// windows in the third window group.
func TestDeepSeekKVLayersGroupAsVLLMDoes(t *testing.T) {
	v4 := []kv.Group{{Kind: kv.SlidingWindowMLA, Window: 128, BlockSize: 64}, {Kind: kv.SlidingWindowMLA, Window: 128, BlockSize: 64},
		{Kind: kv.FullAttention, BlockSize: 256}, {Kind: kv.SlidingWindowMLA, Window: 128, BlockSize: 8},
		{Kind: kv.SlidingWindowMLA, Window: 8, BlockSize: 4}}
	v4mtp := append([]kv.Group(nil), v4...)
	v4mtp[1].Eagle = true
	var v41 []kv.Group
	for range 8 {
		v41 = append(v41, kv.Group{Kind: kv.SlidingWindowMLA, Window: 128, BlockSize: 32, BoundedReplay: true})
	}
	v41 = append(v41, kv.Group{Kind: kv.FullAttention, BlockSize: 64}, kv.Group{Kind: kv.CircularBuffer, BlockSize: 8})
	v41[2].Eagle = true
	for _, c := range []struct {
		scenario string
		want     []kv.Group
	}{
		{"d000-032.yaml", v4},    // deepseek-v4-pro b200 tp8
		{"d001-032.yaml", v4mtp}, // the same with MTP
		{"d010-032.yaml", v4},    // deepseek-v4-pro h200 tp8
		{"d034-032.yaml", v41},   // deepseek-v4.1-flash h200 tp4, block 64, DSpark K 2
	} {
		t.Run(c.scenario, func(t *testing.T) {
			p := openPool(t, c.scenario)
			runs, err := p.model.KVLayers()
			if err != nil {
				t.Fatal(err)
			}
			e, err := p.model.Engine()
			if err != nil {
				t.Fatal(err)
			}
			got, err := vllm.KVGroups(vllmRuns(runs), false, e.Speculative != nil && e.Speculative.NumSpecTokens > 0)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("groups\n%v\nwant\n%v", got, c.want)
			}
		})
	}
}

// On an SM90 part, where FlashMLA's V4 record runs (584 B, 576 B pages), the layers mapped from the
// graph are core/kv's fixtures of vLLM's own modules exactly: deepseekV4ProLayers(256, ratios, mtp)
// and deepseekV41FlashLayers(64, 2, replay) in core/kv/packed_test.go, which TestPackedGroupsAreVLLMs
// pins to vLLM's own grouping, restated here layer by layer as those fixtures build them (core/kv's
// test helpers are not importable).
func TestDeepSeekKVLayersAreVLLMsModules(t *testing.T) {
	for _, c := range []struct {
		scenario string
		want     []kv.Layer
	}{
		{"d010-032.yaml", deepseekV4ProFixture(256, false)},
		{"d011-032.yaml", deepseekV4ProFixture(256, true)},
		{"d034-032.yaml", deepseekV41FlashFixture(64, 2, true)},
	} {
		t.Run(c.scenario, func(t *testing.T) {
			p := openPool(t, c.scenario)
			runs, err := p.model.KVLayers()
			if err != nil {
				t.Fatal(err)
			}
			got, err := vllm.KVLayers(vllmRuns(runs))
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(c.want) {
				t.Fatalf("%d layers, want %d", len(got), len(c.want))
			}
			for i := range got {
				if got[i] != c.want[i] {
					t.Errorf("layer %d: %+v, want %+v", i, got[i], c.want[i])
				}
			}
		})
	}
}

// deepseekV4ProFixture is core/kv's deepseekV4ProLayers (packed_test.go): per layer of
// deepseek-v4-pro's compress_ratios, a ratio-4 layer's indexer key cache and its compressor
// state, the sliding window, then a compressed layer's MLA cache and compressor state; mtp adds
// the MTP layer's window.
func deepseekV4ProFixture(blockSize int, mtp bool) []kv.Layer {
	ratios := []int{128, 128}
	for range 29 {
		ratios = append(ratios, 4, 128)
	}
	ratios = append(ratios, 4)
	swa := kv.Layer{Spec: kv.LayerSlidingWindowMLA, Window: 128, KVHeads: 1, HeadSize: 512, BlockSize: 64, StateBytes: 584, Alignment: 576}
	var out []kv.Layer
	n := len(ratios)
	if mtp {
		n++
	}
	for i := 0; i < n; i++ {
		cr := 1
		if i < len(ratios) {
			cr = ratios[i]
		}
		if cr == 4 {
			out = append(out,
				kv.Layer{Spec: kv.LayerMLA, KVHeads: 1, HeadSize: 132, BlockSize: blockSize, TokensPerState: 4, Alignment: 576},
				kv.Layer{Spec: kv.LayerSlidingWindowMLA, Window: 8, KVHeads: 1, HeadSize: 512, BlockSize: 4, ElemBytes: 4, Alignment: 576})
		}
		out = append(out, swa)
		if cr > 1 {
			coff, block := 1, 8
			if cr == 4 {
				coff, block = 2, 4
			}
			out = append(out,
				kv.Layer{Spec: kv.LayerMLA, KVHeads: 1, HeadSize: 512, BlockSize: blockSize, TokensPerState: cr, StateBytes: 584, Alignment: 576},
				kv.Layer{Spec: kv.LayerSlidingWindowMLA, Window: coff * cr, KVHeads: 1, HeadSize: 2 * coff * 512, BlockSize: block, ElemBytes: 4, Alignment: 576})
		}
	}
	return out
}

// deepseekV41FlashFixture is core/kv's deepseekV41FlashLayers (packed_test.go): per layer, a kv
// source's indexer key cache, the sliding window (block 32), then a kv source's MLA cache and, at
// ratio 2, its compressor ring; then a DSpark drafter's three windows.
func deepseekV41FlashFixture(blockSize, numSpec int, replay bool) []kv.Layer {
	ratios := []int{0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	kvSource := map[int]bool{2: true, 8: true, 14: true, 20: true}
	ring := 8
	for ring < numSpec+2 {
		ring *= 2
	}
	swa := kv.Layer{Spec: kv.LayerSlidingWindowMLA, Window: 128, KVHeads: 1, HeadSize: 512, BlockSize: 32, StateBytes: 584, Alignment: 576, BoundedReplay: replay}
	var out []kv.Layer
	for i, cr := range ratios {
		if kvSource[i] {
			out = append(out, kv.Layer{Spec: kv.LayerMLA, KVHeads: 1, HeadSize: 132, BlockSize: blockSize, TokensPerState: max(cr, 1), Alignment: 576})
		}
		out = append(out, swa)
		if kvSource[i] {
			out = append(out, kv.Layer{Spec: kv.LayerMLA, KVHeads: 1, HeadSize: 512, BlockSize: blockSize, TokensPerState: max(cr, 1), StateBytes: 584, Alignment: 576})
			if cr > 1 {
				out = append(out, kv.Layer{Spec: kv.LayerCircularBuffer, KVHeads: 1, HeadSize: 1024, BlockSize: ring, ElemBytes: 4})
			}
		}
	}
	if numSpec > 0 {
		out = append(out, swa, swa, swa)
	}
	return out
}

// DeepSeek V4 and V4.1 build on 0.32.0 through engine.New, with their packed KV cache groups,
// their drafters and a pool sized over the groups as vLLM sizes it, and run requests to
// completion: d000 (deepseek-v4-pro), d001 (with MTP) and d027 (deepseek-v4.1-flash with DSpark).
//
// The pool is the KV memory over vLLM's bytes per block, the widest group's pages on the
// block-outermost layout DeepSeek V4 runs (get_kv_cache_config_from_groups,
// vllm/v1/core/kv_cache_utils.py:1521-1553, 1622-1628@5df63a72). For deepseek-v4-pro that is over
// core/kv's fixture of vLLM's own modules. For d027, on SM100, V4.1's MXFP8 window record (528 B)
// and NVFP4 compressed record (288 B) run, at 512 B pages (vllm/models/deepseek_v41/attention.py:
// 510-523@5df63a72), and the compressed and indexer caches take the engine's block, 128
// (:1150-1151, :1213-1214): the widest group is the MLA group, three ratio-2 kv sources of an
// indexer page (64 x 132 B, to 8704) and a compressed page (64 x 288 B, 18432) and the ratio-1 one
// (128 x 132 B, 16896, and 128 x 288 B, 36864), 135168 B.
func TestDashboardDeepSeekDeploymentsBuildAndRun(t *testing.T) {
	Register()
	fixture := func(layers []kv.Layer) int64 {
		per, err := kv.PoolBytesPerBlock(layers, kv.GroupOptions{}, 256)
		if err != nil {
			t.Fatal(err)
		}
		return per
	}
	for _, c := range []struct {
		scenario      string
		bytesPerBlock int64
	}{
		{"d000-032.yaml", fixture(deepseekV4ProFixture(256, false))},
		{"d001-032.yaml", fixture(deepseekV4ProFixture(256, true))},
		{"d027-032.yaml", 3*(8704+18432) + 16896 + 36864},
	} {
		t.Run(c.scenario, func(t *testing.T) {
			spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, c.scenario)}
			done, err := prepare(spec)
			if err != nil {
				t.Fatal(err)
			}
			blocks, err := strconv.ParseInt(stated(done.Config, "num_gpu_blocks"), 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			b, err := openPool(t, c.scenario).model.KVBudget()
			if err != nil {
				t.Fatal(err)
			}
			if want := b.AllocatableBytes / c.bytesPerBlock; blocks != want {
				t.Errorf("num_gpu_blocks %d, want %d (%d B of KV memory over vLLM's %d B a block)",
					blocks, want, b.AllocatableBytes, c.bytesPerBlock)
			}
			tr, err := enginetest.Drive(func() (engine.Engine, error) { return engine.New(spec) },
				[]enginetest.Arrival{{At: 0, Req: req("a", 512, 64)}, {At: 5000, Req: req("b", 1024, 32)}},
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
