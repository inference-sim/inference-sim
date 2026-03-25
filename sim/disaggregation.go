package sim

import (
	"fmt"

	"github.com/sirupsen/logrus"
)

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
//
// Stateful implementations may additionally implement DisaggregationObserver to
// learn from routing outcomes (e.g., PrefixThresholdDecider).
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
// Using a single virtual key repurposes the per-instance PrefixCacheIndex structure to
// maintain one aggregate view of cluster-wide prefix state without modifying its interface.
// Collision with real instance IDs is not a risk: instance IDs use a numeric index (e.g.,
// "instance_0"), whereas this sentinel uses the "__" prefix convention.
const globalVirtualInstance = "__global__"

// defaultDisaggLRUCapacity is the number of block hashes tracked in PrefixThresholdDecider's
// router-side prefix cache. 10,000 blocks × 16 tokens/block = 160K tokens per virtual instance.
const defaultDisaggLRUCapacity = 10000

// DisaggregationObserver is an optional interface for stateful DisaggregationDeciders that need
// to learn from routing decisions. ClusterSimulator calls ObserveRouting after each routing
// decision that assigns a request to an instance: after standard routing (RoutingDecisionEvent
// in cluster_event.go) and after prefill routing (PrefillRoutingEvent in pd_events.go).
// It is not called for decode routing, because the decider's prefix knowledge is based on
// input tokens, which are identical between prefill and decode sub-requests.
//
// ObserveRouting is called synchronously within the event loop immediately after each
// routing decision, so the prefix cache is always current at the next Decide() call.
//
// req is guaranteed non-nil. Implementations must treat req as read-only.
// instanceID is the routing target; implementations maintaining per-instance state should
// record it; implementations with global state (like PrefixThresholdDecider) may ignore it.
type DisaggregationObserver interface {
	ObserveRouting(req *Request, instanceID string)
}

// PrefixThresholdDecider disaggregates a request when its non-cached token count exceeds
// the configured threshold. Maintains a router-side prefix cache (globalVirtualInstance)
// to estimate how many input tokens are already cached cluster-wide.
//
// Non-cached token count: len(req.InputTokens) - cachedBlocks * blockSize (always >= 0,
// because ComputeBlockHashes only produces complete-block hashes so cachedBlocks*blockSize
// never exceeds len(InputTokens)). threshold and blockSize are in token counts, not bytes.
// Decision: Disaggregate = (nonCachedTokens > threshold).
//
// The prefix cache is always current at decision time: ObserveRouting is called
// synchronously within the event loop immediately after each routing decision.
//
// Threading: cachedHashes and cachedReqID are a single-use scratchpad — Decide() writes
// them and ObserveRouting() consumes and clears them. This is safe only because the DES
// event loop is single-threaded: no Decide() can interleave between a Decide() call and
// its paired ObserveRouting() call. If routing is rejected after Decide() (no routable
// instances), ObserveRouting() is not called; the stale scratchpad is harmlessly
// overwritten by the next Decide() call.
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
		idx:       NewPrefixCacheIndex(blockSize, defaultDisaggLRUCapacity),
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
// matches; otherwise recomputes. The ID mismatch case is expected in the disaggregated path:
// notifyDisaggregationObserver is called with the prefill sub-request (ID "<parent>_prefill"),
// which differs from the parent request evaluated by Decide.
func (p *PrefixThresholdDecider) ObserveRouting(req *Request, _ string) {
	if req == nil {
		logrus.Errorf("PrefixThresholdDecider.ObserveRouting: req is nil (contract violation); skipping cache update")
		return
	}
	if len(req.InputTokens) == 0 {
		return // no complete blocks to record; consistent with Decide's early-return for empty tokens
	}
	hashes := p.cachedHashes
	if req.ID != p.cachedReqID || p.cachedHashes == nil {
		hashes = p.idx.ComputeBlockHashes(req.InputTokens)
	}
	p.idx.RecordBlocks(hashes, globalVirtualInstance)
	p.cachedHashes = nil
	p.cachedReqID = ""
}

// Compile-time interface compliance check.
var _ DisaggregationObserver = (*PrefixThresholdDecider)(nil)
