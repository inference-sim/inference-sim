package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// capturingDecider records the RequestView and DisaggregationContext it received on
// each Decide call. Used to assert BEH-1 (selected decode endpoint is passed in) and
// BEH-6 (freshness parity: ctx.DecodeCacheQuery uses the same closure precise-prefix-cache
// consumes from cs.cacheQueryFn).
type capturingDecider struct {
	sawInstanceID  string
	sawInputTokens []int
	sawCacheBlocks int
	// reference mirrors the cluster-side cacheQueryFn the stub compares against at
	// decision time, so both counts are captured at the same clock (no skew from
	// cache refreshes happening after Decide returns).
	referenceCacheQuery map[string]func([]int) int
	sawReferenceBlocks  int
	sawCalls            int
	// disaggregate is the static decision returned; kept false so we don't trigger
	// the prefill path in tests that only care about decider plumbing.
	disaggregate bool
}

func (c *capturingDecider) Decide(view sim.RequestView, ctx sim.DisaggregationContext) sim.DisaggregationDecision {
	c.sawCalls++
	c.sawInstanceID = ctx.DecodeInstanceID()
	c.sawInputTokens = view.InputTokens
	c.sawCacheBlocks = ctx.DecodeCacheQuery(view.InputTokens)
	if c.referenceCacheQuery != nil {
		if fn, ok := c.referenceCacheQuery[c.sawInstanceID]; ok && fn != nil {
			c.sawReferenceBlocks = fn(view.InputTokens)
		}
	}
	return sim.DisaggregationDecision{Disaggregate: c.disaggregate}
}

// TestDisaggregationDecider_BEH1_ReceivesSelectedDecodeEndpoint verifies that the
// DisaggregationContext passed into Decide identifies a real decode-pool instance
// (BEH-1). The decide-first order is: decode routing → decider → prefill routing or
// direct inject. This test swaps the default decider out for a capturing stub and
// asserts the instance ID is (a) non-empty and (b) a member of the decode pool.
func TestDisaggregationDecider_BEH1_ReceivesSelectedDecodeEndpoint(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(1)

	cs := NewClusterSimulator(config, requests, nil)
	stub := &capturingDecider{disaggregate: false}
	cs.disaggregationDecider = stub

	mustRun(t, cs)

	if stub.sawCalls == 0 {
		t.Fatal("BEH-1: Decide was never called; check DisaggregationDecisionEvent wiring")
	}
	if stub.sawInstanceID == "" {
		t.Fatal("BEH-1: DisaggregationContext.DecodeInstanceID() returned empty string")
	}
	// Verify the captured instance ID is a decode-pool member.
	foundInDecodePool := false
	for _, inst := range cs.instances {
		if string(inst.ID()) == stub.sawInstanceID {
			if cs.poolMembership[string(inst.ID())] == PoolRoleDecode {
				foundInDecodePool = true
			}
			break
		}
	}
	if !foundInDecodePool {
		t.Errorf("BEH-1: decider saw instance %q, but that instance is not a decode-pool member", stub.sawInstanceID)
	}
}

// TestDisaggregationDecider_BEH6_FreshnessParityWithPPCScorer verifies BEH-6: the
// DisaggregationContext.DecodeCacheQuery closure installed into the decider is the
// SAME closure that precise-prefix-cache consumes from cs.cacheQueryFn — both
// inherit the --cache-signal-delay freshness tier. Enforced by construction at the
// cluster_event.go call site; this test pins that invariant by comparing the count
// the decider observes with the count returned directly by cs.cacheQueryFn for the
// same (instance, tokens) pair.
func TestDisaggregationDecider_BEH6_FreshnessParityWithPPCScorer(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(1)

	cs := NewClusterSimulator(config, requests, nil)
	stub := &capturingDecider{
		disaggregate:        false,
		referenceCacheQuery: cs.cacheQueryFn,
	}
	cs.disaggregationDecider = stub

	mustRun(t, cs)

	if stub.sawCalls == 0 {
		t.Fatal("BEH-6 precondition: Decide was never called")
	}
	if cs.cacheQueryFn == nil {
		t.Fatal("BEH-6 precondition: cs.cacheQueryFn is nil")
	}
	if _, ok := cs.cacheQueryFn[stub.sawInstanceID]; !ok {
		t.Fatalf("BEH-6: no cacheQueryFn entry for selected decode instance %q", stub.sawInstanceID)
	}
	// The count the decider observed (through ctx.DecodeCacheQuery) must equal the
	// count precise-prefix-cache would observe (through cs.cacheQueryFn[id]) at the
	// SAME clock — both values are captured inside Decide().
	if stub.sawReferenceBlocks != stub.sawCacheBlocks {
		t.Errorf("BEH-6: decider saw %d cached blocks, cs.cacheQueryFn returns %d for the same (instance, tokens) at decision time; freshness tier diverged",
			stub.sawCacheBlocks, stub.sawReferenceBlocks)
	}
}

// TestDisaggregationDecider_INV_DECIDER_1_CalledOncePerRequest verifies that Decide
// is invoked exactly once per parent request — not once per sub-request, not twice
// after preemption, etc. Enforced by the Decide-after-decode-routing flow in
// DisaggregationDecisionEvent.Execute.
func TestDisaggregationDecider_INV_DECIDER_1_CalledOncePerRequest(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	const numRequests = 5
	requests := newTestRequests(numRequests)

	cs := NewClusterSimulator(config, requests, nil)
	stub := &capturingDecider{disaggregate: false}
	cs.disaggregationDecider = stub

	mustRun(t, cs)

	if stub.sawCalls != numRequests {
		t.Errorf("INV-DECIDER-1: Decide called %d times for %d requests; want exactly one call per parent request",
			stub.sawCalls, numRequests)
	}
}
