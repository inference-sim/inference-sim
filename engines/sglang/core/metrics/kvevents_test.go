// Original to blis-sglang.

package metrics

import (
	"encoding/hex"
	"testing"

	"github.com/inference-sim/blis-sglang/core/cache"
)

// The bytes msgspec.msgpack.Encoder().encode writes for a KVEventBatch of the events the replay
// scenarios do not queue (AllBlocksCleared) and of every integer width, taken from SGLang's own
// classes at the pin (disaggregation/kv_events.py:233-310@297e54fd) with timestamp 0.0 and
// attn_dp_rank 1:
//
//	KVEventBatch(ts=0.0, events=[AllBlocksCleared(), BlockRemoved(block_hashes=[-5, 300, 70000,
//	    -40000, 2**40, -2**40], medium="GPU")], attn_dp_rank=1)
func TestEncodeMatchesMsgspecOnEveryIntegerWidthAndTheUnqueuedEvents(t *testing.T) {
	const want = "93cb00000000000000009281a474797065b0416c6c426c6f636b73436c656172656483a474797065ac426c6f636b52656d6f766564ac626c6f636b5f68617368657396fbcd012cce00011170d2ffff63c0cf0000010000000000d3ffffff0000000000a66d656469756da347505501"
	got := hex.EncodeToString(EncodeKVEventBatch(0, []cache.Event{
		{Type: "AllBlocksCleared"},
		{Type: "BlockRemoved", BlockHashes: []int64{-5, 300, 70000, -40000, 1 << 40, -(1 << 40)}, Medium: "GPU"},
	}, 1))
	if got != want {
		t.Fatalf("encoded\n got  %s\n want %s", got, want)
	}
}
