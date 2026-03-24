package sim

import "fmt"

// DisaggregationDecision encapsulates the prefill-decode disaggregation decision for a request.
type DisaggregationDecision struct {
	Disaggregate bool // true = route to prefill pool, false = route to shared/decode pool
}

// DisaggregationDecider decides whether a request should be disaggregated
// (sent to a dedicated prefill pool) or handled by the default routing pipeline.
// Used by ClusterSimulator's event pipeline when pool topology is configured.
//
// Implementations must not read Request.OutputTokens (INV-9 oracle boundary);
// use len(req.InputTokens) and req.MaxOutputLen only.
//
// req is guaranteed non-nil; implementations may assume a non-nil pointer.
type DisaggregationDecider interface {
	Decide(req *Request) DisaggregationDecision
}

// NeverDisaggregate always returns Disaggregate=false.
// Default decider when PD disaggregation is not configured.
type NeverDisaggregate struct{}

func (n *NeverDisaggregate) Decide(_ *Request) DisaggregationDecision {
	return DisaggregationDecision{Disaggregate: false}
}

// AlwaysDisaggregate always returns Disaggregate=true.
// Test-oriented decider for validating disaggregation pipeline wiring.
type AlwaysDisaggregate struct{}

func (a *AlwaysDisaggregate) Decide(_ *Request) DisaggregationDecision {
	return DisaggregationDecision{Disaggregate: true}
}

// NewDisaggregationDecider creates a disaggregation decider by name.
// Valid names are defined in validDisaggregationDeciders (bundle.go).
// An empty string defaults to NeverDisaggregate.
// Panics on unrecognized names, on "prefix-threshold" (use NewPrefixThresholdDecider
// directly), or on "direct-to-decode" (use NewDirectToDecodeDecider directly).
// Both parameterized deciders require a caller-supplied threshold.
func NewDisaggregationDecider(name string) DisaggregationDecider {
	if !IsValidDisaggregationDecider(name) {
		panic(fmt.Sprintf("unknown disaggregation decider %q", name))
	}
	switch name {
	case "", "never":
		return &NeverDisaggregate{}
	case "always":
		return &AlwaysDisaggregate{}
	case "prefix-threshold":
		panic("use NewPrefixThresholdDecider(threshold, blockSize) to construct prefix-threshold decider")
	case "direct-to-decode":
		panic("use NewDirectToDecodeDecider(threshold) to construct direct-to-decode decider")
	default:
		panic(fmt.Sprintf(
			"disaggregation decider %q is registered in validDisaggregationDeciders (bundle.go) "+
				"but has no case in NewDisaggregationDecider; add a case here",
			name))
	}
}

// globalVirtualInstance is the single key used in PrefixThresholdDecider's PrefixCacheIndex
// to represent cluster-wide prefix knowledge. All requests update the same virtual instance
// so the decider tracks the global set of recently-seen prefixes.
const globalVirtualInstance = "__global__"

// DisaggregationObserver is an optional interface for stateful DisaggregationDeciders that need
// to learn from routing decisions. ClusterSimulator calls ObserveRouting after each routing
// decision: after standard (non-disaggregated) routing in cluster_event.go, and after prefill
// routing in the disaggregated path in pd_events.go. It is not called for decode routing.
//
// Signal freshness (R17): ObserveRouting is called synchronously within the event loop,
// so the cache state is always current at decision time.
//
// req is guaranteed non-nil. Implementations must treat req as read-only.
type DisaggregationObserver interface {
	ObserveRouting(req *Request, instanceID string)
}

// PrefixThresholdDecider disaggregates a request when its non-cached token count exceeds
// the configured threshold. Maintains a router-side prefix cache (globalVirtualInstance)
// to estimate how many input tokens are already cached cluster-wide.
//
// Non-cached token count: len(req.InputTokens) - cachedBlocks * blockSize.
// Decision: Disaggregate = (nonCachedTokens > threshold).
//
// Signal freshness (R17, INV-7): Uses router-side PrefixCacheIndex updated synchronously
// via ObserveRouting after each routing decision -- always fresh.
type PrefixThresholdDecider struct {
	threshold    int
	blockSize    int
	idx          *PrefixCacheIndex
	cachedHashes []string
	cachedReqID  string
}

// NewPrefixThresholdDecider creates a PrefixThresholdDecider with the given threshold and block size.
// threshold must be >= 0; blockSize must be > 0. Panics otherwise (R3).
func NewPrefixThresholdDecider(threshold, blockSize int) *PrefixThresholdDecider {
	if threshold < 0 {
		panic(fmt.Sprintf("NewPrefixThresholdDecider: threshold must be >= 0, got %d", threshold))
	}
	if blockSize <= 0 {
		panic(fmt.Sprintf("NewPrefixThresholdDecider: blockSize must be > 0, got %d", blockSize))
	}
	return &PrefixThresholdDecider{
		threshold: threshold,
		blockSize: blockSize,
		idx:       NewPrefixCacheIndex(blockSize, defaultLRUCapacity),
	}
}

// Decide returns Disaggregate=true when non-cached token count exceeds the threshold.
// Empty requests (len(InputTokens) == 0) always return Disaggregate=false.
// Caches block hashes for reuse by ObserveRouting when the same request is routed next.
func (p *PrefixThresholdDecider) Decide(req *Request) DisaggregationDecision {
	if len(req.InputTokens) == 0 {
		return DisaggregationDecision{Disaggregate: false}
	}
	p.cachedHashes = p.idx.ComputeBlockHashes(req.InputTokens)
	p.cachedReqID = req.ID
	cachedBlocks := p.idx.MatchLength(p.cachedHashes, globalVirtualInstance)
	nonCachedTokens := len(req.InputTokens) - cachedBlocks*p.blockSize
	return DisaggregationDecision{Disaggregate: nonCachedTokens > p.threshold}
}

// ObserveRouting updates the prefix cache after a routing decision, recording the request's
// block hashes under globalVirtualInstance. Reuses hashes from Decide when the request ID
// matches; otherwise recomputes (e.g., disaggregated path where sub-request ID differs).
func (p *PrefixThresholdDecider) ObserveRouting(req *Request, _ string) {
	if req == nil || len(req.InputTokens) == 0 {
		return
	}
	hashes := p.cachedHashes
	if req.ID != p.cachedReqID || p.cachedHashes == nil {
		hashes = p.idx.ComputeBlockHashes(req.InputTokens)
	}
	p.idx.RecordBlocks(hashes, globalVirtualInstance)
}

// Compile-time interface compliance check.
var _ DisaggregationObserver = (*PrefixThresholdDecider)(nil)
