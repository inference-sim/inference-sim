package sim

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests exercise the route-to-holder routing policy (B-2, #1490). Per the
// micro-plan §6, the policy UNIT tests (T3–T8) construct RouteToHolder DIRECTLY
// (`&RouteToHolder{inner: NewRoutingPolicy("weighted", …)}`) rather than through the
// `route-to-holder` factory name, so they stay independent of the §5.1/§5.2
// name-registration ordering constraint. The factory/validation path is covered by
// the validity test (T10, bundle_test.go) and CLI selectability (T11, cmd tests).

// newHolderTestInner builds the inner weighted policy used by RouteToHolder in these
// unit tests. A queue-depth-only profile with nil RNG makes selection a deterministic,
// load-driven positional argmax — no tie-break RNG, no cache dependency — so the tests
// assert on observable routing behavior, not on scorer internals.
func newHolderTestInner() RoutingPolicy {
	return NewRoutingPolicy("weighted", []ScorerConfig{{Name: "queue-depth", Weight: 1.0}}, 16, nil)
}

// T3 / BC-5: empty snapshots ⇒ panic (convention shared with every routing policy).
func TestRouteToHolder_EmptySnapshots_Panics(t *testing.T) {
	policy := &RouteToHolder{inner: newHolderTestInner()}
	assert.Panics(t, func() {
		policy.Route(&Request{Adapter: "sql-lora"}, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 1000})
	}, "RouteToHolder must panic on empty snapshots")
}

// T4 / BC-1 (INV-PS1): with ≥1 holder, the target is a holder even when a non-holder
// scores strictly higher. The holder is deliberately the WORST-loaded instance, so a
// plain weighted policy would pick a non-holder — proving the restriction, not a
// coincidental score win.
func TestRouteToHolder_SelectsHolder_EvenWhenNonHolderScoresHigher(t *testing.T) {
	req := &Request{Adapter: "sql-lora"}
	// "holder" has the highest queue depth ⇒ worst queue-depth score.
	snapshots := []RoutingSnapshot{
		{ID: "good-a", QueueDepth: 0},
		{ID: "good-b", QueueDepth: 1},
		{ID: "holder", QueueDepth: 100, ResidentAdapters: residentSetOf("sql-lora")},
	}

	// Sanity: an unconstrained weighted policy would NOT pick the holder here.
	weighted := newHolderTestInner()
	baseline := weighted.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})
	require.NotEqual(t, "holder", baseline.TargetInstance,
		"precondition: unconstrained weighted must prefer a non-holder for this fixture")

	policy := &RouteToHolder{inner: newHolderTestInner()}
	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})
	assert.Equal(t, "holder", decision.TargetInstance,
		"route-to-holder must select the holder even though it scores lowest (INV-PS1)")
}

// T5 / BC-2 (D1): no instance holds the adapter ⇒ decision is identical to
// unconstrained weighted routing. The oracle is an INDEPENDENTLY-constructed weighted
// policy over the same state, so the test survives any rewrite of RouteToHolder.
func TestRouteToHolder_NoHolder_EqualsUnconstrainedWeighted(t *testing.T) {
	req := &Request{Adapter: "sql-lora"}
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 3, ResidentAdapters: residentSetOf("other")},
		{ID: "b", QueueDepth: 1, ResidentAdapters: residentSetOf("another")},
		{ID: "c", QueueDepth: 7},
	}

	oracle := newHolderTestInner()
	want := oracle.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	policy := &RouteToHolder{inner: newHolderTestInner()}
	got := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	assert.Equal(t, want, got, "no-holder fallback must equal unconstrained weighted routing")
}

// T6 / BC-3: base-model request (empty Adapter) ⇒ no restriction; decision equals
// unconstrained weighted routing, even when adapters happen to be resident.
func TestRouteToHolder_EmptyAdapter_EqualsUnconstrainedWeighted(t *testing.T) {
	req := &Request{Adapter: ""}
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 3, ResidentAdapters: residentSetOf("sql-lora")},
		{ID: "b", QueueDepth: 1, ResidentAdapters: residentSetOf("sql-lora")},
		{ID: "c", QueueDepth: 7},
	}

	oracle := newHolderTestInner()
	want := oracle.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	policy := &RouteToHolder{inner: newHolderTestInner()}
	got := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	assert.Equal(t, want, got, "empty-adapter request must not be restricted (base-model neutrality)")
}

// T7 / BC-4 (INV-6): identical state + request + freshly re-seeded RNG ⇒ byte-identical
// decision (target, reason, scores). Uses a multi-holder tie scenario so the inner
// weighted RNG tie-break actually fires, making the determinism claim non-vacuous.
func TestRouteToHolder_Deterministic(t *testing.T) {
	req := &Request{Adapter: "sql-lora"}
	// Two holders with identical load ⇒ a tie the inner RNG must break.
	snapshots := []RoutingSnapshot{
		{ID: "h1", QueueDepth: 2, ResidentAdapters: residentSetOf("sql-lora")},
		{ID: "h2", QueueDepth: 2, ResidentAdapters: residentSetOf("sql-lora")},
		{ID: "cold", QueueDepth: 0},
	}

	build := func() RoutingPolicy {
		inner := NewRoutingPolicy("weighted", []ScorerConfig{{Name: "queue-depth", Weight: 1.0}}, 16, rand.New(rand.NewSource(42)))
		return &RouteToHolder{inner: inner}
	}

	d1 := build().Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})
	d2 := build().Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	assert.Equal(t, d1, d2, "identical inputs + re-seeded RNG must produce byte-identical decisions")
	// The selected target must still be one of the holders (INV-PS1 under a tie).
	assert.Contains(t, []string{"h1", "h2"}, d1.TargetInstance)
}

// T8 / BC-6 (INV-PS1 property): over ≥100 random configs each with ≥1 holder, the
// selected target is always a holder. Each iteration uses a locally-seeded RNG (no
// math/rand global) so the property run is deterministic and reproducible.
func TestRouteToHolder_Property_AlwaysSelectsHolder(t *testing.T) {
	const iterations = 200
	const adapter = "sql-lora"

	for i := 0; i < iterations; i++ {
		rng := rand.New(rand.NewSource(int64(i)))
		n := 2 + rng.Intn(7) // 2..8 instances

		snapshots := make([]RoutingSnapshot, n)
		holderSet := make(map[string]bool)
		for j := 0; j < n; j++ {
			id := string(rune('a' + j))
			snapshots[j] = RoutingSnapshot{ID: id, QueueDepth: rng.Intn(50)}
		}
		// Guarantee ≥1 holder: force one random index to hold the adapter, then let
		// others independently hold it too.
		forced := rng.Intn(n)
		for j := 0; j < n; j++ {
			if j == forced || rng.Intn(2) == 0 {
				snapshots[j].ResidentAdapters = residentSetOf(adapter)
				holderSet[snapshots[j].ID] = true
			}
		}

		inner := NewRoutingPolicy("weighted", []ScorerConfig{{Name: "queue-depth", Weight: 1.0}}, 16, rand.New(rand.NewSource(int64(1000+i))))
		policy := &RouteToHolder{inner: inner}
		decision := policy.Route(&Request{Adapter: adapter}, &RouterState{Snapshots: snapshots, Clock: 1000})

		require.Truef(t, holderSet[decision.TargetInstance],
			"iter %d: target %q is not a holder (holders=%v)", i, decision.TargetInstance, holderSet)
	}
}
