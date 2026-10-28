// Moved from blis-vllm/compat/nixl/handoff.go and lease.go at 635b7bf, with their names
// prefixed NIXL as package sim exported them before: the NIXL connector rules a P/D cluster
// applies around a KV handoff (when a prefill engine's held KV is freed, whether a decode pool
// can ever take a prompt). They stay beside the cluster's holds and lease events, which apply
// them, so the handoff's timing is unchanged (phase 2c).

package cluster

// The NIXL connector's rules for one P/D handoff: whether the decode side takes it, when the
// prefill side frees its held blocks after a handoff that ends without a pull, and what a
// completed pull does on each side. They moved here from inference-sim's cluster (#2021 B2,
// verification pass 3, gap 1), which keeps the holds, the block reference counts, the waiting
// queue and the events, and asks these functions what to do.

// NIXLRelease is when the prefill side frees a handoff's held KV after the handoff ends without a
// completed pull.
type NIXLRelease int

const (
	// NIXLFreeNow: the decode engine aborted a request it had received, and its connector notifies
	// the prefill side with an empty read, which frees the blocks at once
	// (nixl/pull_scheduler.py:219-232@5df63a72).
	NIXLFreeNow NIXLRelease = iota
	// NIXLHoldUntilFirstExpiry: the decode engine never received the request, so no notice comes and
	// no heartbeat renews the lease; the blocks are freed when it runs out, one lease after the
	// prefill completed (NIXLFirstExpiry; _reap_expired_send_leases, nixl/base_worker.py:3014-3040@5df63a72).
	NIXLHoldUntilFirstExpiry
	// NIXLHoldUntilRenewedExpiry: the decode engine had the request and heartbeated until it
	// vanished; the blocks are freed when the lease its last heartbeat granted runs out
	// (NIXLExpiryAfterVanish).
	NIXLHoldUntilRenewedExpiry
)

// NIXLOutcome is how a handoff ended without a completed pull, as the host sees it.
type NIXLOutcome int

const (
	// NIXLDecodeNeverReceived: the decode engine never received the request (its instance was
	// gone or unroutable when the handoff began).
	NIXLDecodeNeverReceived NIXLOutcome = iota
	// NIXLDecodeRejected: the decode engine received the request and rejected it before pulling.
	NIXLDecodeRejected
	// NIXLDecodeVanished: the decode engine had the request, waiting or pulling, and went away.
	NIXLDecodeVanished
)

// NIXLReleaseOnDrop is the prefill release that applies to a handoff that ended without a pull.
func NIXLReleaseOnDrop(o NIXLOutcome) NIXLRelease {
	switch o {
	case NIXLDecodeRejected:
		return NIXLFreeNow
	case NIXLDecodeVanished:
		return NIXLHoldUntilRenewedExpiry
	default:
		return NIXLHoldUntilFirstExpiry
	}
}

// NIXLReleaseAt is the time a release frees the prefill side's held KV: now for NIXLFreeNow, else the
// lease's expiry for the request whose prefill completed at prefillCompleteUs, with the lease
// last renewed at nowUs for NIXLHoldUntilRenewedExpiry.
func NIXLReleaseAt(r NIXLRelease, prefillCompleteUs, nowUs, leaseUs int64) int64 {
	switch r {
	case NIXLFreeNow:
		return nowUs
	case NIXLHoldUntilRenewedExpiry:
		return NIXLExpiryAfterVanish(prefillCompleteUs, nowUs, leaseUs)
	default:
		return NIXLFirstExpiry(prefillCompleteUs, leaseUs)
	}
}

// NIXLDecodeAccepts reports whether a decode engine whose whole KV pool is decodeKVBlocks can ever
// take a handoff of promptKVBlocks. One it cannot is rejected outright -- the request is
// aborted before any allocation, which notifies the prefill side to free at once
// (NIXLDecodeRejected; nixl/pull_scheduler.py:219-232@5df63a72) -- rather than waiting for KV that
// can never be freed.
func NIXLDecodeAccepts(promptKVBlocks, decodeKVBlocks int64) bool {
	return promptKVBlocks <= decodeKVBlocks
}

// NIXLPullComplete is what a completed pull does: the decode side's connector reports the request
// in finished_sending, so the prefill side frees the blocks it held
// (nixl/base_worker.py:2985-2997@5df63a72), and the decode scheduler caches the blocks it
// received, which only now may serve another request's prefix
// (_update_waiting_for_remote_kv, sched/scheduler.py:3154-3157@5df63a72).
type NIXLPullComplete struct {
	FreePrefillHold    bool
	CacheReceivedBlock bool
}

// NIXLOnPullComplete is the effect of a pull that completed on a live decode engine.
func NIXLOnPullComplete() NIXLPullComplete {
	return NIXLPullComplete{FreePrefillHold: true, CacheReceivedBlock: true}
}

// NIXLDefaultLeaseUs is vLLM's default kv_lease_duration, 30 s (nixl/base_scheduler.py:81-85).
const NIXLDefaultLeaseUs int64 = 30_000_000

// NIXLLease is the kv_lease_duration a deployment states, in microseconds; 0 or less takes vLLM's
// default.
func NIXLLease(statedUs int64) int64 {
	if statedUs > 0 {
		return statedUs
	}
	return NIXLDefaultLeaseUs
}

// NIXLFirstExpiry is when a held request's lease runs out if no decode side ever heartbeats: the
// lease from the prefill's completion.
func NIXLFirstExpiry(prefillCompleteUs, leaseUs int64) int64 { return prefillCompleteUs + leaseUs }

// NIXLExpiryAfterVanish is when the lease of a request whose decode engine vanished at now runs
// out: the later of its first deadline and the extension its last heartbeat granted.
// Heartbeats go out every lease/6 and extend to 2/3 of a lease from then
// (nixl/base_scheduler.py:87, base_worker.py:563-567, 3084-3096), so the last one came at most
// lease/6 before now and the true expiry lies in [now + lease/2, now + 2/3 lease] -- 15 s to
// 20 s after the vanish at the default 30 s lease. BLIS takes the upper end, a heartbeat at the
// vanish itself.
func NIXLExpiryAfterVanish(prefillCompleteUs, nowUs, leaseUs int64) int64 {
	return max(NIXLFirstExpiry(prefillCompleteUs, leaseUs), nowUs+leaseUs*2/3)
}
