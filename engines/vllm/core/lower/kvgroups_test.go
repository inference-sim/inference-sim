// Original to blis-vllm.

package lower

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"

	"github.com/inference-sim/blis-vllm/core/kv"
)

func layers(spec kv.LayerSpec, n, window int) []kv.Layer {
	out := make([]kv.Layer, n)
	for i := range out {
		out[i] = kv.Layer{Spec: spec, Window: window, KVHeads: 8, HeadSize: 64, ElemBytes: 2}
	}
	return out
}

// aligned sets every layer's stride alignment, the bytes a stride between physical blocks must be
// a multiple of (get_block_stride_alignment): a page below it is padded up in a layer-outer layout.
func aligned(ls []kv.Layer, a int) []kv.Layer {
	for i := range ls {
		ls[i].StrideAlignment = a
	}
	return ls
}

// A hybrid model's pages are the groups' own, and the pool's stride is the widest. gpt-oss-120b
// at TP1 in bf16 with 16-token blocks: 8 KV heads of 64, keys and values, 2 bytes, is 32768 bytes
// a layer-block, so two groups of 18 layers hold 18 pages each.
func TestThePlanStatesEachGroupsPage(t *testing.T) {
	ls := append(layers(kv.LayerSlidingWindow, 18, 128), layers(kv.LayerFull, 18, 0)...)
	got, err := KVGroups(ls, kv.GroupOptions{}, 16)
	if err != nil {
		t.Fatal(err)
	}
	const page = 2 * 8 * 64 * 2 * 16
	want := []plan.KVGroup{{BytesPerPage: 18 * page, TokensPerPage: 16, Window: 128}, {BytesPerPage: 18 * page, TokensPerPage: 16}}
	if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("groups %+v, want %+v", got, want)
	}
	stride, err := kv.PoolBytesPerBlock(ls, kv.GroupOptions{}, 16)
	if err != nil {
		t.Fatal(err)
	}
	var widest int64
	for _, g := range got {
		widest = max(widest, int64(g.BytesPerPage))
	}
	if widest != stride {
		t.Errorf("widest group %d, pool stride %d", widest, stride)
	}
}

// The stated pages agree with the pool's stride for every shape the engine groups, including the
// packed layout of DeepSeek V4.
func TestTheWidestStatedPageIsThePoolStride(t *testing.T) {
	v4 := []kv.Layer{}
	for range 4 {
		v4 = append(v4, kv.Layer{Spec: kv.LayerSlidingWindowMLA, Window: 128, KVHeads: 1, HeadSize: 512, BlockSize: 64,
			StateBytes: 584, ElemBytes: 1, Alignment: 576, StrideAlignment: 1})
		v4 = append(v4, kv.Layer{Spec: kv.LayerMLA, KVHeads: 1, HeadSize: 512, BlockSize: 64, TokensPerState: 4,
			StateBytes: 584, ElemBytes: 1, Alignment: 576, StrideAlignment: 1})
	}
	for name, ls := range map[string][]kv.Layer{
		"one full group":      layers(kv.LayerFull, 12, 0),
		"hybrid":              append(layers(kv.LayerSlidingWindow, 30, 128), layers(kv.LayerFull, 20, 0)...),
		"deepseek v4 packed":  v4,
		"indexer beside full": append(layers(kv.LayerFull, 6, 0), layers(kv.LayerIndexer, 3, 0)...),
		"stride alignment":    aligned(layers(kv.LayerFull, 5, 0), 4096),
	} {
		got, err := KVGroups(ls, kv.GroupOptions{}, 16)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		stride, err := kv.PoolBytesPerBlock(ls, kv.GroupOptions{}, 16)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		var widest int64
		for _, g := range got {
			widest = max(widest, int64(g.BytesPerPage))
		}
		// Every layout's pool stride is the widest group page, the packed DeepSeek V4 one included
		// (SUB-14: the bytes a plan states are the bytes the engine allocates).
		if widest != stride {
			t.Errorf("%s: widest group page %d, pool stride %d", name, widest, stride)
		}
	}
}

// A model with no KV layers has no groups to state; a plan with a group of no bytes is refused by
// the plan's own Validate, so KVGroups never returns one.
func TestNoLayersIsAnError(t *testing.T) {
	if _, err := KVGroups(nil, kv.GroupOptions{}, 16); err == nil {
		t.Error("a model with no KV layers gave groups")
	}
}

// In a layer-outer layout each layer's block stride is its page padded to its alignment, so a
// group of five 32768-byte pages (2 x 8 heads x 64 x 2 bytes x 16 tokens) at an alignment of 12288
// holds 5 x 36864 bytes, not 5 x 32768.
func TestAPageIsPaddedToItsStrideAlignment(t *testing.T) {
	got, err := KVGroups(aligned(layers(kv.LayerFull, 5, 0), 12288), kv.GroupOptions{}, 16)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].BytesPerPage != 5*36864 {
		t.Errorf("groups %+v; five 32768-byte pages padded to 12288 are 5 x 36864", got)
	}
}
