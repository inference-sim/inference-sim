package sim

// RouterState provides cluster-wide state to policy interfaces.
// Built by ClusterSimulator before each policy invocation.
// This is a bridge type in sim/ (not sim/cluster/) to avoid import cycles —
// same pattern as RoutingSnapshot.
//
// USAGE BOUNDARY: In production, only constructed by ClusterSimulator's event
// handlers (via buildRouterState or executeDisaggregatedRouting). Tests may
// construct directly. Single-instance Simulator does not use RouterState —
// instance-level policies (InstanceScheduler) receive parameters directly.
// This prevents import cycles: sim/cluster/ imports sim/, not the reverse.
type RouterState struct {
	// Snapshots carries the routable instances visible to the current policy
	// invocation. For the standard routing path this is all routable instances
	// in the deployment. For the disaggregated routing path this is filtered to
	// the decode-pool members only (parity with llm-d's EPP handing the PD
	// decider a pre-selected decode endpoint).
	Snapshots []RoutingSnapshot
	// PrefillSnapshots carries the prefill-pool members in the disaggregated
	// routing path (one per routable instance whose PoolRole == PoolRolePrefill).
	// Empty for the standard routing path and for disaggregated contexts whose
	// deployment has no prefill pool configured. The three built-in deciders
	// (never, always, prefix-threshold) do not read this field; it exists so
	// future cross-pool-aware and joint D+P policies can observe prefill-pool
	// load and per-prefill-pod cache hits. Issue #1339 (G1).
	PrefillSnapshots []RoutingSnapshot
	// One per Loading instance. Populated fields: ID, Model, GPUType, TPDegree,
	// CostPerHour, TotalKvCapacityTokens. QueueDepth, BatchSize, KVUtilization,
	// FreeKVBlocks, CacheHitRate, InFlightRequests, KvTokensInUse remain zero.
	LoadingSnapshots []RoutingSnapshot
	Clock            int64 // Current simulation clock in microseconds
	// SelectedDecodeInstance is the decode-pod instance ID pre-selected by an
	// upstream caller (typically the decode-routing policy) before invoking a
	// DisaggregationDecider. Empty for contexts where no prior selection has
	// been made (e.g., RoutingPolicy.Route itself receives RouterState with
	// SelectedDecodeInstance == ""). DisaggregationDecider implementations may
	// use this ID to locate the selected pod's state (e.g., in a per-pod cache-
	// query map). Membership in Snapshots is not structurally guaranteed —
	// implementations must tolerate both a zero value ("no selection known")
	// and an ID that has no corresponding entry in Snapshots (e.g., the pod
	// was removed after routing) and guard accordingly.
	//
	// Renamed from SelectedInstance in issue #1339 (G6) to disambiguate now
	// that PrefillSnapshots is visible.
	SelectedDecodeInstance string
	// SelectedPrefillInstance is reserved for future joint D+P policies that
	// select a prefill pod alongside the decode pod. The three built-in
	// deciders leave this empty and downstream prefill-routing code treats
	// an empty value as "no hint — apply normal prefill routing". Added in
	// issue #1339 (G6) to pre-empt a future interface break.
	SelectedPrefillInstance string
}
