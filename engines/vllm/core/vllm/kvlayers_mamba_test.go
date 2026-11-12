// Original to blis-vllm.

package vllm

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-vllm/core/kv"
)

// 4 attention layers (2 KV heads of 8) beside 8 Mamba layers whose 600-byte state is padded to
// the attention page. The memory model gives the bytes for one block of the attention layers
// alone, 4 x 2 x 16 x 2 x 8 x 2 = 4096 (a Mamba layer has no per-token KV): the attention layers'
// element size is derived from that, 2 bytes, as it is for a model without Mamba layers
// (kv.Layer.StateBytes of a Mamba layer is not a packed state that stops the derivation).
// A pipeline stage holding the four attention layers alone then pages 4 x 1024, not 4 x 512.
func TestMambaLayersDoNotStopTheAttentionElementSizeDerivation(t *testing.T) {
	runs := []KVLayerRun{
		{Count: 4, Spec: "full", NumKVHeads: 2, HeadSize: 8},
		{Count: 8, Spec: "mamba", StateBytes: 600, PageSizePadded: 1024, BlockSize: 16, MambaCacheMode: "align"},
	}
	got, err := KVStagePoolBytesPerBlock(runs, [][2]int{{0, 4}, {4, 12}}, false, 16, 4096)
	if err != nil {
		t.Fatal(err)
	}
	// Stage 1 holds the 8 Mamba layers: two groups of 4 pages of 1024.
	if want := []int64{4096, 4096}; !reflect.DeepEqual(got, want) {
		t.Errorf("stage pool bytes per block %v, want %v", got, want)
	}
	groups, err := KVGroupsWith(runs, kv.GroupOptions{BlockSize: 16})
	if err != nil || len(groups) != 3 || groups[0].Kind != kv.FullAttention || groups[1].Kind != kv.Mamba || groups[2].Kind != kv.Mamba {
		t.Errorf("groups %+v (%v), want full attention and two Mamba groups", groups, err)
	}
}
