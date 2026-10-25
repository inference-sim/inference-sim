// Moved from inference-sim/sim/cluster/pd_events.go at 5bcf93f86a60df138bef28609100f13dfe317910.

package nixl

// The NIXL connector's rules for one P/D handoff: whether the decode side takes it, when the
// prefill side frees its held blocks after a handoff that ends without a pull, and what a
// completed pull does on each side. They moved here from inference-sim's cluster (#2021 B2,
// verification pass 3, gap 1), which keeps the holds, the block reference counts, the waiting
// queue and the events, and asks these functions what to do.

// Release is when the prefill side frees a handoff's held KV after the handoff ends without a
// completed pull.
type Release int

const (
	// FreeNow: the decode engine aborted a request it had received, and its connector notifies
	// the prefill side with an empty read, which frees the blocks at once
	// (nixl/pull_scheduler.py:219-232@5df63a72).
	FreeNow Release = iota
	// HoldUntilFirstExpiry: the decode engine never received the request, so no notice comes and
	// no heartbeat renews the lease; the blocks are freed when it runs out, one lease after the
	// prefill completed (FirstExpiry; _reap_expired_send_leases, nixl/base_worker.py:3014-3040@5df63a72).
	HoldUntilFirstExpiry
	// HoldUntilRenewedExpiry: the decode engine had the request and heartbeated until it
	// vanished; the blocks are freed when the lease its last heartbeat granted runs out
	// (ExpiryAfterVanish).
	HoldUntilRenewedExpiry
)

// Outcome is how a handoff ended without a completed pull, as the host sees it.
type Outcome int

const (
	// DecodeNeverReceived: the decode engine never received the request (its instance was
	// gone or unroutable when the handoff began).
	DecodeNeverReceived Outcome = iota
	// DecodeRejected: the decode engine received the request and rejected it before pulling.
	DecodeRejected
	// DecodeVanished: the decode engine had the request, waiting or pulling, and went away.
	DecodeVanished
)

// ReleaseOnDrop is the prefill release that applies to a handoff that ended without a pull.
func ReleaseOnDrop(o Outcome) Release {
	switch o {
	case DecodeRejected:
		return FreeNow
	case DecodeVanished:
		return HoldUntilRenewedExpiry
	default:
		return HoldUntilFirstExpiry
	}
}

// ReleaseAt is the time a release frees the prefill side's held KV: now for FreeNow, else the
// lease's expiry for the request whose prefill completed at prefillCompleteUs, with the lease
// last renewed at nowUs for HoldUntilRenewedExpiry.
func ReleaseAt(r Release, prefillCompleteUs, nowUs, leaseUs int64) int64 {
	switch r {
	case FreeNow:
		return nowUs
	case HoldUntilRenewedExpiry:
		return ExpiryAfterVanish(prefillCompleteUs, nowUs, leaseUs)
	default:
		return FirstExpiry(prefillCompleteUs, leaseUs)
	}
}

// DecodeAccepts reports whether a decode engine whose whole KV pool is decodeKVBlocks can ever
// take a handoff of promptKVBlocks. One it cannot is rejected outright -- the request is
// aborted before any allocation, which notifies the prefill side to free at once
// (DecodeRejected; nixl/pull_scheduler.py:219-232@5df63a72) -- rather than waiting for KV that
// can never be freed.
func DecodeAccepts(promptKVBlocks, decodeKVBlocks int64) bool {
	return promptKVBlocks <= decodeKVBlocks
}

// PullComplete is what a completed pull does: the decode side's connector reports the request
// in finished_sending, so the prefill side frees the blocks it held
// (nixl/base_worker.py:2985-2997@5df63a72), and the decode scheduler caches the blocks it
// received, which only now may serve another request's prefix
// (_update_waiting_for_remote_kv, sched/scheduler.py:3154-3157@5df63a72).
type PullComplete struct {
	FreePrefillHold    bool
	CacheReceivedBlock bool
}

// OnPullComplete is the effect of a pull that completed on a live decode engine.
func OnPullComplete() PullComplete {
	return PullComplete{FreePrefillHold: true, CacheReceivedBlock: true}
}
