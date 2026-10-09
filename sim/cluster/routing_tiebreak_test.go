package cluster

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// Tests for --routing-deterministic-tiebreak / DeploymentConfig.RoutingDeterministicTiebreak
// (backlog entry #3). The knob swaps the router's RANDOM equal-score tie-break for the
// positional (first-in-snapshot-order) one already implemented in LeastLoaded.Route and
// WeightedScoring.Route (sim/routing.go, both preceded by "Random tie-breaking when rng is
// non-nil; positional (first) when nil."), by passing a nil *rand.Rand instead of
// rng.ForSubsystem(sim.SubsystemRouter) at the single construction site in
// NewClusterSimulator (sim/cluster/cluster.go).
//
// The brief for this task proposed instrumenting NewClusterSimulator's third argument
// with a counting RNG wrapper to directly count SubsystemRouter draws. That is not
// possible: the third argument is onRequestDone func(*sim.Request, int64) []*sim.Request,
// not an RNG, and even if it were, PartitionedRNG.ForSubsystem returns a concrete
// *rand.Rand (sim/rng.go:74) — there is no interface seam to wrap without changing a
// production signature purely for a test. Instead, this file tests the OBSERVABLE
// consequence of "the router consumes no randomness": with the knob on, routing must be
// seed-invariant, because a positional tie-break never calls into the RNG at all. The
// confound backlog entry #3 removes is RNG-stream MISALIGNMENT — how many draws the
// router consumes depends on how many scoring ties occurred, which an eviction or
// creation policy changes, so two arms of a paired comparison diverge in routing
// decisions that are not the treatment; seed-invariance under the knob is what closes
// that channel by construction.

// tiebreakRequests returns a fresh, content-fixed LoRA-adapter workload: 200 requests
// over a 6-adapter registry with capacity 2 per instance (< adapter count), drawn from a
// Zipfian distribution seeded independently of cfg.Seed (zipfianAdapterRequests uses a
// fixed internal seed of 7), so every call returns objects with IDENTICAL content.
//
// It deliberately returns FRESH *sim.Request objects on every call rather than one slice
// shared across runs: ClusterSimulator mutates Request fields in place during Run (State,
// ProgressIndex, AssignedInstance, FirstTokenTime, ...), so literally reusing one slice's
// pointers across two separate simulator runs would leak the first run's mutations into
// the second run's "identical workload" premise — the two runs would not actually be
// starting from the same requests. Calling this fresh each time gets the same effect
// (workload content held fixed, only cfg.Seed varies) without that aliasing hazard.
func tiebreakRequests() []*sim.Request {
	return zipfianAdapterRequests(200, periodicTestAdapters)
}

// tiebreakConfig builds a 4-instance, LoRA-active deployment with weighted routing scored
// solely by lora-affinity. Ties are common in this fixture: scoreLoRAAffinity
// (sim/routing_scorers.go) scores every instance 1.0 (neutral) whenever the raw residency
// vector is uniform — which is exactly the state of the world for any adapter's first
// requests, before any instance has loaded it — so the router's tie-break branch fires
// routinely, not exotically, and the knob's effect is actually observable.
func tiebreakConfig(seed int64, on bool) DeploymentConfig {
	cfg := loraAffinityTestConfig(4, 2, periodicTestAdapters) // 4 instances: ties are common
	cfg.RoutingPolicy = "weighted"
	cfg.RoutingScorerConfigs = []sim.ScorerConfig{{Name: "lora-affinity", Weight: 1.0}}
	cfg.Seed = seed
	cfg.RoutingDeterministicTiebreak = on
	return cfg
}

// runWithTiebreak builds and runs a cluster for the given seed/knob combination and
// returns its marshaled AggregatedMetrics for byte comparison.
func runWithTiebreak(t *testing.T, seed int64, on bool) []byte {
	t.Helper()
	cfg := tiebreakConfig(seed, on)
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(tiebreakRequests()), nil)
	mustRun(t, cs)
	m, err := json.Marshal(cs.AggregatedMetrics())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return m
}

// TestTiebreak_OffIsByteIdenticalToToday is the inertness law for this knob: default
// false must reproduce current behaviour exactly (no committed result moves), and an
// explicit false must be indistinguishable from the field being left absent.
func TestTiebreak_OffIsByteIdenticalToToday(t *testing.T) {
	a := runWithTiebreak(t, 42, false)
	b := runWithTiebreak(t, 42, false)
	if !bytes.Equal(a, b) {
		t.Fatal("baseline run is not reproducible; fix the fixture before trusting this file")
	}

	// An explicit false and an absent field must be the same run. Built on the SAME
	// LoRA-active, tie-prone shape as tiebreakConfig (4 instances, consistently) rather
	// than a trivial round-robin config: round-robin never touches the router RNG
	// regardless of the knob, so a trivial fixture would pass here even if the knob were
	// wired backwards. (In Go, an untouched bool field and one explicitly set to false
	// are the same zero value — there is no runtime "absent" state; that distinction is
	// real only in the YAML `omitempty` tag. This still guards the construction path.)
	cfgAbsent := loraAffinityTestConfig(4, 2, periodicTestAdapters)
	cfgAbsent.RoutingPolicy = "weighted"
	cfgAbsent.RoutingScorerConfigs = []sim.ScorerConfig{{Name: "lora-affinity", Weight: 1.0}}
	cfgAbsent.Seed = 42
	// RoutingDeterministicTiebreak intentionally left untouched.
	cs := NewClusterSimulator(cfgAbsent, NewSliceRequestSource(tiebreakRequests()), nil)
	mustRun(t, cs)
	m, err := json.Marshal(cs.AggregatedMetrics())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !bytes.Equal(a, m) {
		t.Error("RoutingDeterministicTiebreak=false differs from the field being absent")
	}
}

// TestTiebreak_OnIsReproducible is a basic INV-6 sanity check for the ON state: the same
// seed run twice with the knob on must be byte-identical.
func TestTiebreak_OnIsReproducible(t *testing.T) {
	a := runWithTiebreak(t, 42, true)
	b := runWithTiebreak(t, 42, true)
	if !bytes.Equal(a, b) {
		t.Fatal("knob on, same seed run twice: not reproducible")
	}
}

// TestTiebreak_OnIsSeedInvariant is the substantive, falsifiable test. If the knob truly
// makes the router consume zero draws from the SubsystemRouter partition, then which seed
// governs the OTHER subsystems cannot matter to routing decisions — the workload content
// is held fixed (tiebreakRequests), so cfg.Seed is the ONLY thing that varies between
// these two runs. Two runs differing only in seed must therefore route identically and
// produce byte-identical output.
//
// If this fails, report BLOCKED rather than weakening the assertion: it means some other
// seed-derived partition (not the router) is still reaching the output.
func TestTiebreak_OnIsSeedInvariant(t *testing.T) {
	on42 := runWithTiebreak(t, 42, true)
	on99 := runWithTiebreak(t, 99, true)
	if !bytes.Equal(on42, on99) {
		t.Fatalf("BLOCKED: knob on, seed 42 vs seed 99 differ — a seed-derived partition other "+
			"than the router is reaching routing output:\nseed42=%s\nseed99=%s", on42, on99)
	}
}

// TestTiebreak_OffIsSeedDependent is the required complement to
// TestTiebreak_OnIsSeedInvariant: with the knob OFF, the router's random tie-break
// consumes draws whose count depends on how many ties occurred, so two runs differing
// only in seed must NOT be byte-identical. If they were, this fixture produced no
// scoring ties at all, and the seed-invariance test above would be vacuously green —
// passing because nothing ever exercised the RNG, not because the knob works.
func TestTiebreak_OffIsSeedDependent(t *testing.T) {
	off42 := runWithTiebreak(t, 42, false)
	off99 := runWithTiebreak(t, 99, false)
	if bytes.Equal(off42, off99) {
		t.Fatal("VACUOUS: knob off, seed 42 vs seed 99 are byte-identical — the fixture " +
			"produced no scoring ties, so this file cannot observe the knob's effect; widen " +
			"the fixture (more instances/adapters) rather than trusting a green " +
			"TestTiebreak_OnIsSeedInvariant")
	}
}
