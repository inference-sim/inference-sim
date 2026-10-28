package cluster

import (
	"container/heap"
	"fmt"
)

// Prefill-side KV hold (#1934 point 3, INV-PD-7).
//
// A prefill engine under a NIXL handoff keeps a finished request's KV blocks allocated until
// the decode side pulls them, aborts the request, or vanishes and the lease runs out. When a
// lease expires, and which release a handoff's outcome takes, are the connector's rules
// (NIXLLease, NIXLReleaseOnDrop, NIXLReleaseAt, from blis-vllm compat/nixl). The
// cluster keeps the holds, the
// block reference counts and the lease event. It maps the handoff outcomes onto those rules: a
// completed transfer frees at its completion; a drop at transfer start (the decode side never
// took the request) frees at once; a decode instance that became unroutable mid-transfer is a
// vanished decode side, so the blocks are freed when the lease runs out.

func (c *ClusterSimulator) pdKVLeaseUs() int64 { return NIXLLease(c.config.PDKVLeaseUs) }

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
	inst := c.instanceByID(InstanceID(instID))
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
	inst := c.instanceByID(InstanceID(instID))
	inst.ReleaseHeldKV(parent.PrefillSubReq, c.clock)
}

// HeldPrefillKVTokens is the KV, in tokens, that prefill instance id holds right now for pulls
// not yet complete, each block counted once however many held requests share it.
func (c *ClusterSimulator) HeldPrefillKVTokens(id string) int64 {
	return int64(len(c.heldBlockRefs[id])) * c.config.BlockSizeTokens
}

// releasePrefillOnDrop frees parent's held prefill KV as the connector's rule says for a handoff
// that ended without a pull (outcome, at now): at once, or at the lease expiry the rule names
// (NIXLReleaseOnDrop, NIXLReleaseAt). The cluster keeps the hold and its event.
func (c *ClusterSimulator) releasePrefillOnDrop(parent *ParentRequest, outcome NIXLOutcome, now int64) {
	release := NIXLReleaseOnDrop(outcome)
	if release == NIXLFreeNow {
		c.releasePrefillHold(parent)
		return
	}
	c.expirePrefillLease(parent, NIXLReleaseAt(release, parent.PrefillCompleteTime, now, c.pdKVLeaseUs()))
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

// PrefillKVLeaseExpiredEvent frees a prefill instance's held KV when its lease runs out.
// Priority 6, beside KVTransferCompletedEvent.
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
