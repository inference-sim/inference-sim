// Moved from inference-sim/sim/cluster/pd_kv_hold.go at 003f1e0c3ed6aa61f0133d6619154441c68c6c19.

// Package nixl is the compatibility engine's model of vLLM's NIXL connector rules that a P/D
// simulator's cluster applies around a KV handoff: how long a prefill engine holds a finished
// request's KV for the decode side's pull (#1934, INV-PD-7). It moved here from inference-sim's
// cluster (#2021 E4), which keeps the holds, the block reference counts and the lease event,
// and asks this package when a lease expires.
//
// A NIXL prefill engine does not free a finished request's KV blocks: request_finished returns
// delay_free_blocks and records a lease deadline (vllm/distributed/kv_transfer/kv_connector/
// v1/nixl/pull_scheduler.py:247-266 at v0.31.0). The blocks stay allocated, so they count
// against the prefill engine's KV capacity, until one of three things happens:
//
//   - the decode side's pull completes and it notifies the prefill side, which reports the
//     request in finished_sending and the scheduler frees the blocks (nixl/base_worker.py:
//     2902-3031, get_finished);
//   - the decode side aborts the request before pulling: its request_finished still sees
//     do_remote_prefill and queues an empty read "so our worker side will notify and free
//     blocks in the prefill instance" (pull_scheduler.py:219-232);
//   - the lease expires (nixl/base_worker.py:3056-3082, _reap_expired_send_leases). A live
//     decode side heartbeats every lease/6 and each heartbeat extends the lease to 2/3 of a
//     lease from then (nixl/base_scheduler.py:81-87, base_worker.py:563-567 and 3084-3096),
//     so the lease expires only after the decode side has gone.
package nixl

// DefaultLeaseUs is vLLM's default kv_lease_duration, 30 s (nixl/base_scheduler.py:81-85).
const DefaultLeaseUs int64 = 30_000_000

// Lease is the kv_lease_duration a deployment states, in microseconds; 0 or less takes vLLM's
// default.
func Lease(statedUs int64) int64 {
	if statedUs > 0 {
		return statedUs
	}
	return DefaultLeaseUs
}

// FirstExpiry is when a held request's lease runs out if no decode side ever heartbeats: the
// lease from the prefill's completion.
func FirstExpiry(prefillCompleteUs, leaseUs int64) int64 { return prefillCompleteUs + leaseUs }

// ExpiryAfterVanish is when the lease of a request whose decode engine vanished at now runs
// out: the later of its first deadline and the extension its last heartbeat granted.
// Heartbeats go out every lease/6 and extend to 2/3 of a lease from then
// (nixl/base_scheduler.py:87, base_worker.py:563-567, 3084-3096), so the last one came at most
// lease/6 before now and the true expiry lies in [now + lease/2, now + 2/3 lease] -- 15 s to
// 20 s after the vanish at the default 30 s lease. BLIS takes the upper end, a heartbeat at the
// vanish itself.
func ExpiryAfterVanish(prefillCompleteUs, nowUs, leaseUs int64) int64 {
	return max(FirstExpiry(prefillCompleteUs, leaseUs), nowUs+leaseUs*2/3)
}
