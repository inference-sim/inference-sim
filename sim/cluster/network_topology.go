package cluster

// recordNodeSpan records how many physical nodes an instance placed on gpuIDs occupies, so
// MaxNodesSpanned reports the widest span in the fleet. `blis run --trace-output` writes it
// into the trace header, and replay refuses a trace whose fleet spanned nodes because node
// pools are run-only and replay cannot reconstruct the placement (#1530).
//
// Called from all three placement sites -- startup, the deferred NodeReadyEvent path, and
// autoscaler scale-up -- right after the per-instance GPU type and hourly cost are resolved
// from the same placement (R23). The span is the count of distinct nodes actually occupied,
// so it is right even when the nodes differ in size.
func (cs *ClusterSimulator) recordNodeSpan(gpuIDs []string) {
	if cs.placement == nil {
		return // no node pools ⇒ no placement ⇒ every instance is single-node
	}
	if span := len(cs.placement.distinctNodesForGPUs(gpuIDs)); span > cs.maxNodesSpanned {
		cs.maxNodesSpanned = span
	}
}
