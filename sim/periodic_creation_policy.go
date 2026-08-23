package sim

// PeriodicCreationPolicy is the OPTIONAL cluster-scope creation entry point (Spec 3).
// A creation policy that does not implement it never ticks, and a set
// DeploymentConfig.LoRAPeriodicIntervalUs stays inert for that policy (INV-PS3').
//
// OnTick is a PURE query, exactly like CreationPolicy's two methods: it RETURNS
// decisions and the cluster validates and actuates them. A policy cannot load, evict,
// or mutate any cluster or instance state through this interface (Principle I's
// no-mutators rule is unchanged).
//
// A returned decision is a REQUEST, not a command. The cluster drops any decision
// naming an unknown or non-routable instance, an adapter the registry does not have,
// an instance whose load channel is busy, or an instance with a gate-blocked request —
// so a misbehaving policy cannot defeat those deferral rules.
//
// Read "gate-blocked" exactly: the wait-queue HEAD is a cold miss. The rules above are
// therefore narrower than a general "never disturb work in progress" guarantee, and no
// such guarantee is offered. A prefetch may still evict the adapter of a request that is
// merely QUEUED (pins are taken at batch admission, not at enqueue) and so delay it by up
// to one LoadLatency. Widening the rule to skip any instance with a non-empty wait queue
// would suppress prefetching precisely under load — a design decision, deliberately not
// taken here.
type PeriodicCreationPolicy interface {
	OnTick(ctx PeriodicCreationContext) []PrefetchDecision
}

// PeriodicCreationContext is the read-only view a periodic creation policy sees.
//
// It is CLUSTER-scoped, and that is a deliberate, bounded departure from
// CreationContext's Principle I restriction ("the policy never sees the cross-instance
// map", creation_policy.go:43-44) rather than an erosion of it: a decision that places
// adapters ACROSS instances cannot be instance-scoped and still be a placement
// decision. The two instance-scoped entry points (Initial, OnResidentMiss) keep their
// restriction unchanged.
//
// Determinism is structural (INV-6): every collection here is a sorted or
// construction-ordered slice, and demand is an accessor rather than a map, so there is
// no map for a policy to range over.
type PeriodicCreationContext struct {
	// Now is the tick's simulation timestamp in microseconds.
	Now int64
	// Interval is the configured tick interval in microseconds. Exposed so a policy can
	// express a window without a second config knob (keep-warm uses exactly this).
	Interval int64
	// Instances is every routable instance, in cluster construction order (INV-6).
	Instances []InstanceResidency
	// Demand is the read-only per-adapter request-recency accessor. May be nil only when
	// the subsystem is inert; a policy MUST nil-check before use.
	Demand DemandWindow
	// Registry is the read-only adapter registry accessor (Has/RankOf), the same
	// accessor CreationContext carries. May be nil when inert; nil-check before use.
	Registry AdapterRegistry
}

// InstanceResidency is one instance's residency and load-channel state.
type InstanceResidency struct {
	// ID is the instance id a PrefetchDecision must name.
	ID string
	// ConstructionIndex is this instance's position in the CLUSTER'S construction
	// order — the same key space PlacementScheduleEntry.Placement and
	// --lora-adapter-placement's DeploymentConfig.LoRAAdapterPlacement use (D9).
	//
	// It is NOT the index of this entry within the Instances slice: a non-routable
	// instance is omitted from Instances (buildContext), so slice position and
	// construction index diverge exactly when an instance is skipped. A policy that
	// keys off PlacementScheduleEntry.Placement MUST index by this field, never by a
	// range index over Instances — ranging positionally silently hands one instance
	// another instance's target set the moment any earlier instance is non-routable.
	ConstructionIndex int
	// Resident is the currently-resident adapter ids, sorted (INV-6).
	Resident []string
	// Unpinned is the eviction seam's candidate set, sorted (INV-6). A prefetch into a
	// full instance is only possible when this is non-empty, because every resident
	// adapter being pinned means no victim can be selected.
	Unpinned []string
	// Capacity is TOTAL resident slots, not free ones: free = Capacity - len(Resident).
	Capacity int
	// Loading is the adapter id currently occupying this instance's serialized load
	// channel, or "" when the channel is free. A decision naming a busy instance is
	// dropped at actuation.
	Loading string
	// GateBlocked reports that a cold-miss request is waiting at this instance's gate.
	// A decision naming such an instance is dropped at actuation (demand priority).
	GateBlocked bool
}

// PrefetchDecision asks the cluster to start a charged prefetch of Adapter on Instance.
type PrefetchDecision struct {
	Instance string
	Adapter  string
}

// DemandWindow is the read-only per-adapter demand accessor. It is an accessor rather
// than a map for the same reason CreationContext.Registry is: ranging a map would break
// INV-6 non-deterministically, and only under some seeds.
type DemandWindow interface {
	// LastRequestedAt returns the simulation time of the most recent request for the
	// adapter, and false when it has never been requested.
	LastRequestedAt(adapter string) (int64, bool)
	// RequestedSince returns the adapter ids requested at or after since, sorted (INV-6).
	RequestedSince(since int64) []string
}
