package cmd

// crossNodeSpanForTrace normalizes a cluster's widest instance node span for the
// TraceV2 header (#1530). A span of 0 (no node pools, hence no placement) or 1 (every
// instance fits on one node) records nothing, keeping the header byte-identical to a
// pre-feature build; only a real multi-node fleet is recorded, and only that value
// makes `blis replay` refuse the trace.
func crossNodeSpanForTrace(maxNodesSpanned int) int {
	if maxNodesSpanned <= 1 {
		return 0
	}
	return maxNodesSpanned
}
