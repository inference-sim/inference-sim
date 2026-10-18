package cluster

import (
	"container/heap"
	"fmt"
)

// Prefill-side KV hold (#1934 point 3, INV-PD-7).
//
// A NIXL prefill engine does not free a finished request's KV blocks: request_finished
// returns delay_free_blocks and records a lease deadline (vllm/distributed/kv_transfer/
// kv_connector/v1/nixl/pull_scheduler.py:247-266 at v0.31.0). The blocks stay allocated, so
// they count against the prefill engine's KV capacity, until one of three things happens:
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
//
// BLIS maps these onto its handoff outcomes: a completed transfer frees at its completion; a
// drop at transfer start (the decode side never took the request) frees at once; a decode
// instance that became unroutable mid-transfer is a vanished decode side, so the blocks are
// freed when the lease runs out.

// defaultPDKVLeaseUs is vLLM's default kv_lease_duration, 30 s (nixl/base_scheduler.py:81-85).
const defaultPDKVLeaseUs int64 = 30_000_000

func (c *ClusterSimulator) pdKVLeaseUs() int64 {
	if c.config.PDKVLeaseUs > 0 {
		return c.config.PDKVLeaseUs
	}
	return defaultPDKVLeaseUs
}

// startPrefillHold records that parent's prefill instance keeps its KV for the pull. Called
// when the prefill completes, with the sub-request's blocks still allocated
// (Request.RetainKVOnFinish). No-op when the deployment does not hold.
func (c *ClusterSimulator) startPrefillHold(parent *ParentRequest) {
	if !c.config.PDPrefillKVHold {
		return
	}
	if _, held := c.prefillHolds[parent.ID]; held {
		panic(fmt.Sprintf("startPrefillHold: %s is already held", parent.ID))
	}
	instID := string(parent.PrefillInstanceID)
	inst := c.instanceByID(instID)
	if inst == nil {
		panic(fmt.Sprintf("startPrefillHold: prefill instance %q of %s not found", instID, parent.ID))
	}
	blocks := append([]int64(nil), inst.KVBlocksOf(parent.PrefillSubReq)...)
	c.prefillHolds[parent.ID] = blocks
	refs := c.heldBlockRefs[instID]
	if refs == nil {
		refs = make(map[int64]int)
		c.heldBlockRefs[instID] = refs
	}
	for _, b := range blocks {
		refs[b]++
	}
	tokens := int64(len(refs)) * c.config.BlockSizeTokens
	c.heldKVPeakTokens[instID] = max(c.heldKVPeakTokens[instID], tokens)
}

// releasePrefillHold frees parent's held prefill KV. Idempotent: a parent whose hold already
// ended (or never began) is left alone, so the transfer-complete and lease paths can both
// reach it.
func (c *ClusterSimulator) releasePrefillHold(parent *ParentRequest) {
	blocks, held := c.prefillHolds[parent.ID]
	if !held {
		return
	}
	delete(c.prefillHolds, parent.ID)
	instID := string(parent.PrefillInstanceID)
	refs := c.heldBlockRefs[instID]
	for _, b := range blocks {
		if refs[b]--; refs[b] == 0 {
			delete(refs, b)
		}
	}
	inst := c.instanceByID(instID)
	inst.ReleaseHeldKV(parent.PrefillSubReq)
	inst.WakeIfParked(c.clock)
}

// HeldPrefillKVTokens is the KV, in tokens, that prefill instance id holds right now for pulls
// not yet complete, each block counted once however many held requests share it.
func (c *ClusterSimulator) HeldPrefillKVTokens(id string) int64 {
	return int64(len(c.heldBlockRefs[id])) * c.config.BlockSizeTokens
}

// expireLeaseOfVanishedDecode schedules the release of parent's held KV for when NIXL's lease
// runs out after a decode engine that had the request vanished at now: the later of the
// request's first deadline (prefill completion + lease) and the extension its last heartbeat
// granted. Heartbeats go out every lease/6 and extend to 2/3 of a lease from then
// (nixl/base_scheduler.py:87, base_worker.py:563-567, 3084-3096), so the last one came at most
// lease/6 before now and the true expiry lies in [now + lease/2, now + 2/3 lease] -- 15 s to
// 20 s after the vanish at the default 30 s lease. BLIS takes the upper end, a heartbeat at the
// vanish itself.
func (c *ClusterSimulator) expireLeaseOfVanishedDecode(parent *ParentRequest, now int64) {
	lease := c.pdKVLeaseUs()
	c.expirePrefillLease(parent, max(parent.PrefillCompleteTime+lease, now+lease*2/3))
}

// expirePrefillLease schedules the release of parent's held prefill KV at expiry.
func (c *ClusterSimulator) expirePrefillLease(parent *ParentRequest, expiry int64) {
	if _, held := c.prefillHolds[parent.ID]; !held {
		return
	}
	heap.Push(&c.clusterEvents, clusterEventEntry{
		event: &PrefillKVLeaseExpiredEvent{time: expiry, parentReq: parent},
		seqID: c.nextSeqID(),
	})
}

// PrefillKVLeaseExpiredEvent frees a prefill instance's held KV when NIXL's lease on it runs
// out (_reap_expired_send_leases). Priority 6, beside KVTransferCompletedEvent.
type PrefillKVLeaseExpiredEvent struct {
	time      int64
	parentReq *ParentRequest
}

func (e *PrefillKVLeaseExpiredEvent) Timestamp() int64 { return e.time }
func (e *PrefillKVLeaseExpiredEvent) Priority() int    { return 6 }

// Execute frees the held KV.
func (e *PrefillKVLeaseExpiredEvent) Execute(cs *ClusterSimulator) {
	cs.releasePrefillHold(e.parentReq)
}

// PrefillKVHoldPeakTokens is the most KV, in tokens, that any one prefill instance held at
// once for pulls not yet complete (INV-PD-7), each block counted once however many held
// requests share it through the prefix cache. Zero when the deployment does not hold.
func (c *ClusterSimulator) PrefillKVHoldPeakTokens() int64 {
	var peak int64
	for _, p := range c.heldKVPeakTokens {
		peak = max(peak, p)
	}
	return peak
}

// PrefillKVHeld reports how many parents' prefill KV is held right now.
func (c *ClusterSimulator) PrefillKVHeld() int { return len(c.prefillHolds) }

// instanceByID returns the instance with the given ID, or nil.
func (c *ClusterSimulator) instanceByID(id string) *InstanceSimulator {
	for _, inst := range c.instances {
		if string(inst.ID()) == id {
			return inst
		}
	}
	return nil
}
