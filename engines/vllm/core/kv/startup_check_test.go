// Original to blis-vllm.

package kv

import (
	"strings"
	"testing"
)

// One request's blocks over its groups are each group's own: full attention every token, a
// sliding window its admission cap (window - 1 + extra retained + in-flight tokens, plus the block
// the window may start inside), and a circular buffer its one ring block
// (_max_memory_usage_bytes_from_groups, vllm/v1/core/kv_cache_utils.py:2337-2366@5df63a72).
func TestMaxRequestBlocksSumsEveryGroupsAdmissionCap(t *testing.T) {
	limits := RequestLimits{MaxModelLen: 4096, MaxInFlightTokens: 512, ExtraRetained: 2}
	// gpt-oss-style hybrid: a 128-token window and full attention, one group each.
	layers := []Layer{{Spec: LayerSlidingWindow, Window: 128, KVHeads: 8, HeadSize: 64},
		{Spec: LayerFull, KVHeads: 8, HeadSize: 64}}
	got, err := MaxRequestBlocks(layers, GroupOptions{}, 16, limits)
	if err != nil {
		t.Fatal(err)
	}
	// window: cdiv(min(127 + 2 + 512, 4096), 16) + 1 = 41 + 1; full: cdiv(4096, 16) = 256.
	if want := int64(42 + 256); got != want {
		t.Errorf("MaxRequestBlocks = %d, want %d", got, want)
	}
	// A window longer than the model is capped at max_model_len.
	long := []Layer{{Spec: LayerSlidingWindow, Window: 8192, KVHeads: 8, HeadSize: 64},
		{Spec: LayerFull, KVHeads: 8, HeadSize: 64}}
	if got, _ := MaxRequestBlocks(long, GroupOptions{}, 16, limits); got != int64(256+1+256) {
		t.Errorf("a window above max_model_len: %d blocks, want %d", got, 256+1+256)
	}
}

// The check passes a pool that holds one request beside the null block and refuses one block
// fewer, with vLLM's message and the longest window that fits (_check_enough_kv_cache_memory and
// _estimate_max_model_len_from_groups, kv_cache_utils.py:1003-1040, 2369-2412@5df63a72).
func TestCheckEnoughKVCacheMemoryIsVLLMsStartupCheck(t *testing.T) {
	limits := RequestLimits{MaxModelLen: 4096, MaxInFlightTokens: 512}
	layers := []Layer{{Spec: LayerSlidingWindow, Window: 128, KVHeads: 8, HeadSize: 64},
		{Spec: LayerFull, KVHeads: 8, HeadSize: 64}}
	need, err := MaxRequestBlocks(layers, GroupOptions{}, 16, limits)
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckEnoughKVCacheMemory(layers, GroupOptions{}, 16, limits, need+1, 0, 0); err != nil {
		t.Errorf("a pool of %d blocks holds one %d-block request beside the null block: %v", need+1, need, err)
	}
	err = CheckEnoughKVCacheMemory(layers, GroupOptions{}, 16, limits, need, 0, 0)
	if err == nil || !strings.Contains(err.Error(), "max seq len (4096)") {
		t.Fatalf("one block fewer: err %v, want the start-up refusal", err)
	}
	want, err2 := EstimateMaxModelLen(layers, GroupOptions{}, 16, limits, need-1)
	if err2 != nil || want <= 0 || want >= 4096 {
		t.Fatalf("estimated max model len %d (%v), want one in (0, 4096)", want, err2)
	}
	if !strings.Contains(err.Error(), "the estimated maximum model length is "+itoa(want)+".") {
		t.Errorf("the refusal states no estimate %d: %v", want, err)
	}
	// The estimate is the longest window that fits: one token more does not.
	at := func(n int) int64 {
		l := limits
		l.MaxModelLen = n
		b, _ := MaxRequestBlocks(layers, GroupOptions{}, 16, l)
		return b
	}
	if at(want) > need-1 || at(want+1) <= need-1 {
		t.Errorf("estimate %d: %d blocks there and %d one token on, against %d", want, at(want), at(want+1), need-1)
	}
	// With the pool's bytes, the numbers are vLLM's GiB.
	err = CheckEnoughKVCacheMemory(layers, GroupOptions{}, 16, limits, need, 1<<20, need<<20)
	if err == nil || !strings.Contains(err.Error(), "GiB KV cache is needed") {
		t.Errorf("with the pool's bytes: %v, want vLLM's GiB message", err)
	}
	if err := CheckEnoughKVCacheMemory(layers, GroupOptions{}, 16, limits, 1, 0, 0); err == nil ||
		!strings.Contains(err.Error(), "No available memory") {
		t.Errorf("a pool of only the null block: %v, want vLLM's no-memory refusal", err)
	}
}

func itoa(n int) string {
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	if len(b) == 0 {
		return "0"
	}
	return string(b)
}
