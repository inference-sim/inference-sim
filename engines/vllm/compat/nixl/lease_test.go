// Original to blis-vllm.

package nixl

import "testing"

// An unstated lease is vLLM's 30 s default; a stated one is kept.
func TestLease(t *testing.T) {
	if got := Lease(0); got != 30_000_000 {
		t.Errorf("unstated: %d, want vLLM's 30 s", got)
	}
	if got := Lease(-1); got != DefaultLeaseUs {
		t.Errorf("negative: %d, want the default", got)
	}
	if got := Lease(5_000_000); got != 5_000_000 {
		t.Errorf("stated: %d, want it kept", got)
	}
}

// A lease nobody renews runs out one lease after the prefill completed. After a decode side
// vanishes, its last heartbeat (taken at the vanish) extends the lease to 2/3 of a lease from
// then, unless the first deadline is later.
func TestExpiry(t *testing.T) {
	const lease = 30_000_000
	if got := FirstExpiry(1_000, lease); got != 30_001_000 {
		t.Errorf("first expiry: %d", got)
	}
	// Vanished 25 s after the prefill: the heartbeat's 20 s extension wins (45 s > 30 s).
	if got := ExpiryAfterVanish(0, 25_000_000, lease); got != 45_000_000 {
		t.Errorf("late vanish: %d, want 45 s", got)
	}
	// Vanished 1 s after the prefill: the first deadline (30 s) is later than 1 + 20 s.
	if got := ExpiryAfterVanish(0, 1_000_000, lease); got != 30_000_000 {
		t.Errorf("early vanish: %d, want the first deadline, 30 s", got)
	}
}
