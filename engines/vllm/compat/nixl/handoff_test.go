// Original to blis-vllm.

package nixl

import "testing"

// A handoff's prefill release follows how it ended: a decode side that rejected the request
// frees the prefill at once, one that never received it holds until the first lease runs out,
// and one that vanished holds until the lease its last heartbeat granted runs out.
func TestReleaseOnDrop(t *testing.T) {
	for _, c := range []struct {
		o    Outcome
		want Release
	}{
		{DecodeRejected, FreeNow},
		{DecodeNeverReceived, HoldUntilFirstExpiry},
		{DecodeVanished, HoldUntilRenewedExpiry},
	} {
		if got := ReleaseOnDrop(c.o); got != c.want {
			t.Errorf("ReleaseOnDrop(%d) = %d, want %d", c.o, got, c.want)
		}
	}
}

// Each release frees at its own time: now, the first expiry, or the renewed expiry.
func TestReleaseAt(t *testing.T) {
	const lease, prefillEnd = 30_000_000, 1_000_000
	if got := ReleaseAt(FreeNow, prefillEnd, 5_000_000, lease); got != 5_000_000 {
		t.Errorf("FreeNow: %d, want now", got)
	}
	if got := ReleaseAt(HoldUntilFirstExpiry, prefillEnd, 5_000_000, lease); got != prefillEnd+lease {
		t.Errorf("first expiry: %d, want %d", got, prefillEnd+lease)
	}
	// Vanished 25 s after the prefill: the heartbeat's extension (25 s + 20 s) is the later.
	if got := ReleaseAt(HoldUntilRenewedExpiry, prefillEnd, 26_000_000, lease); got != 26_000_000+lease*2/3 {
		t.Errorf("renewed expiry: %d, want %d", got, 26_000_000+lease*2/3)
	}
}

// A handoff no larger than the decode pool is taken; one larger can never be, at the boundary
// too.
func TestDecodeAccepts(t *testing.T) {
	for _, c := range []struct {
		prompt, pool int64
		want         bool
	}{{10, 100, true}, {100, 100, true}, {101, 100, false}, {0, 0, true}} {
		if got := DecodeAccepts(c.prompt, c.pool); got != c.want {
			t.Errorf("DecodeAccepts(%d, %d) = %v, want %v", c.prompt, c.pool, got, c.want)
		}
	}
}

// A completed pull frees the prefill hold and caches the decode's received blocks.
func TestOnPullComplete(t *testing.T) {
	if got := OnPullComplete(); !got.FreePrefillHold || !got.CacheReceivedBlock {
		t.Errorf("OnPullComplete() = %+v, want both effects", got)
	}
}
