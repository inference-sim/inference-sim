package latencykernel

import (
	"testing"
)

// DeepEP's own sizing for DeepSeek-V4.1-Flash's shape (hidden 5120, 384 experts, top-6,
// EP8), from LowLatencyLayout and ElasticBuffer::calculate_buffer_size at d4f41e4e93:
// 4 x E x T x (2H + 4H/128) for LL once the experts dominate, and 8 x T x 10,336 B
// for v2, 2 MiB aligned.
func TestDeepEPBufferSizesFollowDeepEP(t *testing.T) {
	ll := lowLatencyRDMABytes(256, 5120, 8, 384)
	want := int64(4*384*256*(2*5120+4*40) + 2*1536)
	want = (want + 128) / 128 * 128
	if ll != want {
		t.Errorf("LL RDMA bytes %d, want %d", ll, want)
	}
	v2 := elasticBufferBytes(8192, 5120, 6, 8)
	if want := int64((8*8192*10336 + (2 << 20) - 1) / (2 << 20) * (2 << 20)); v2 != want {
		t.Errorf("v2 buffer %d, want %d", v2, want)
	}
}
