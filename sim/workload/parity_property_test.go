package workload

// Property test for the "eager ≡ lazy" byte-identity claim (issue #1442, A4).
//
// A3 (#1441) added lazy request generation behind --lazy-generation (default
// off). Its load-bearing correctness claim is: for any seed and any
// lazy-supported workload spec, the lazy streaming source (GenerateWorkloadLazy)
// yields the SAME ordered request stream as the eager generator
// (GenerateWorkload) — same length, same per-request fields, same session
// blueprints. stream_test.go pins this for a handful of fixed cases; this file
// puts it under randomized pressure, which is the maintainer's explicit ask
// ("perhaps we can use go property testing to really guarantee this").
//
// The generator (genLazySupportedSpec) samples ONLY from the lazy-supported
// space so the comparison is meaningful: if a draw produced an unsupported
// shape (per-window parameters, Concurrency>0, or multi-session reasoning),
// GenerateWorkloadLazy would return an ErrLazyUnsupported* sentinel and the
// cmd-layer would silently fall back to the eager generator — comparing eager
// to eager and proving nothing. The generator is constructed so it CANNOT emit
// those shapes (no Lifecycle, no Concurrency, reasoning always SingleSession).
//
// This is a companion invariant test to A3's golden/unit tests (BDD/TDD rule
// #4, R7): it asserts a law (eager output == lazy output), not a fixed value.

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"testing"
)

// propertyBaseSeed seeds the top-level draw RNG. It is a FIXED constant (never
// wall-clock) so a CI failure is reproducible: rerun with the printed draw
// index and this base seed regenerates the exact offending spec. Changing this
// value reshuffles the sampled specs — do so only intentionally.
const propertyBaseSeed int64 = 0x1442A4

// defaultPropertyDraws is the number of random (seed, spec) draws per test
// invocation. The issue asks for ≥100 in CI with an env-gated extended mode
// for nightly runs.
const defaultPropertyDraws = 100

// propertyDraws returns the draw count, honoring BLIS_PROPERTY_DRAWS for
// extended (e.g. nightly 10×) runs. A malformed or non-positive value falls
// back to the default rather than silently running zero draws (which would
// make the test vacuously pass — R20 degenerate-input guard).
func propertyDraws(t *testing.T) int {
	t.Helper()
	v := os.Getenv("BLIS_PROPERTY_DRAWS")
	if v == "" {
		return defaultPropertyDraws
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		t.Logf("ignoring invalid BLIS_PROPERTY_DRAWS=%q; using default %d", v, defaultPropertyDraws)
		return defaultPropertyDraws
	}
	return n
}

// pick returns a uniformly random element of opts.
func pick[T any](rng *rand.Rand, opts []T) T {
	return opts[rng.Intn(len(opts))]
}

// genLazySupportedSpec samples a WorkloadSpec from a constrained-but-
// representative space that is ALWAYS lazy-supported. It returns the spec plus
// a short human-readable label describing the sampled shape (surfaced in
// failure messages so a diverging draw is identifiable at a glance) and the
// (horizon, maxRequests) pair to drive generation with.
//
// Sampled dimensions (per the issue's suggested space):
//   - clients: 1–4 explicit + optional cohort (1–6 members) → up to ~10 clients
//   - arrival process: poisson / gamma / weibull / constant
//   - num_requests (maxRequests cap): 10–120 (kept small so 100 draws finish
//     in well under a minute; occasionally a tight cap to exercise the
//     blueprint pre-pass truncation path)
//   - input/output dists: gaussian / exponential with bounded params
//   - optional shared prefix (PrefixGroup + PrefixLength)
//   - optional single-session reasoning (MaxRounds 1–8, accumulate/none)
//
// NEVER sampled (would trip the A3 fallback gates and make the comparison
// vacuous): Lifecycle/per-window parameters, Concurrency>0, SingleSession=false.
func genLazySupportedSpec(rng *rand.Rand) (spec *WorkloadSpec, label string, horizon, maxRequests int64) {
	seed := rng.Int63()

	// Decide overall shape once so eager and lazy specs are built identically.
	numClients := 1 + rng.Intn(4) // 1–4 explicit clients
	useCohort := rng.Intn(2) == 0
	usePrefix := rng.Intn(2) == 0
	useReasoning := rng.Intn(3) == 0 // ~1/3 of draws are reasoning
	accumulate := rng.Intn(2) == 0
	process := pick(rng, []string{"poisson", "gamma", "weibull", "constant"})

	// A shared prefix group string; per-client PrefixLength when usePrefix.
	prefixGroup := ""
	prefixLen := 0
	if usePrefix {
		prefixGroup = "grp"
		prefixLen = 32 + rng.Intn(96) // 32–127 tokens
	}

	var reasoning *ReasoningSpec
	maxRounds := 0
	if useReasoning {
		maxRounds = 1 + rng.Intn(8) // 1–8 rounds
		growth := "none"
		if accumulate {
			growth = "accumulate"
		}
		reasoning = &ReasoningSpec{
			MultiTurn: &MultiTurnSpec{
				MaxRounds:     maxRounds,
				ContextGrowth: growth,
				ThinkTimeUs:   50_000 + int64(rng.Intn(100_000)),
				SingleSession: true, // lazy-supported reasoning only
			},
		}
	}

	// Bounded token distributions. Gaussian needs mean/std_dev/min/max;
	// exponential needs mean.
	inputDist := DistSpec{Type: "gaussian", Params: map[string]float64{
		"mean": float64(40 + rng.Intn(80)), "std_dev": float64(5 + rng.Intn(15)),
		"min": 8, "max": 400,
	}}
	outputDist := DistSpec{Type: "exponential", Params: map[string]float64{
		"mean": float64(20 + rng.Intn(40)),
	}}

	// Build explicit clients. RateFraction is split evenly across explicit
	// clients + cohort so aggregate normalization is well-defined.
	totalUnits := numClients
	cohortMembers := 0
	if useCohort {
		cohortMembers = 1 + rng.Intn(6) // 1–6 members
		totalUnits += cohortMembers
	}
	frac := 1.0 / float64(totalUnits)

	clients := make([]ClientSpec, 0, numClients)
	for i := 0; i < numClients; i++ {
		clients = append(clients, ClientSpec{
			ID:           fmt.Sprintf("c%d", i),
			TenantID:     fmt.Sprintf("t%d", i),
			SLOClass:     "batch",
			RateFraction: frac,
			Arrival:      ArrivalSpec{Process: process},
			InputDist:    inputDist,
			OutputDist:   outputDist,
			PrefixGroup:  prefixGroup,
			PrefixLength: prefixLen,
			Reasoning:    reasoning,
		})
	}

	var cohorts []CohortSpec
	if useCohort {
		cohorts = []CohortSpec{{
			ID:           "co",
			Population:   cohortMembers,
			TenantID:     "tco",
			SLOClass:     "batch",
			RateFraction: frac * float64(cohortMembers),
			Arrival:      ArrivalSpec{Process: process},
			InputDist:    inputDist,
			OutputDist:   outputDist,
			PrefixGroup:  prefixGroup,
			PrefixLength: prefixLen,
			Reasoning:    reasoning,
		}}
	}

	spec = &WorkloadSpec{
		Version:       "2",
		Seed:          seed,
		Category:      "language",
		AggregateRate: 2.0 + float64(rng.Intn(18)), // 2–19 req/s
		Clients:       clients,
		Cohorts:       cohorts,
	}

	// Horizon generous enough that the maxRequests cap is what bounds
	// generation (not horizon exhaustion) — mirrors the existing stream_test
	// cases. Occasionally use a tight cap to exercise blueprint pre-pass
	// truncation for reasoning specs (a known subtle path in stream.go).
	maxRequests = int64(10 + rng.Intn(110)) // 10–119
	if useReasoning && rng.Intn(4) == 0 {
		maxRequests = int64(2 + rng.Intn(8)) // tight cap: 2–9
	}
	horizon = 60_000_000 // 60s — ample for the sampled rates

	label = fmt.Sprintf("clients=%d cohort=%d process=%s prefix=%v reasoning=%v(rounds=%d,accum=%v) maxReq=%d",
		numClients, cohortMembers, process, usePrefix, useReasoning, maxRounds, accumulate, maxRequests)
	return spec, label, horizon, maxRequests
}

// buildSpec reconstructs a spec deterministically from a per-draw seed so that
// eager and lazy each receive their OWN independent instance. GenerateWorkload
// and GenerateWorkloadLazy both run validateAndExpandSpec (which can mutate
// spec.Clients for some spec kinds); giving each mode a freshly-built spec
// removes any chance of cross-contamination between the two runs.
func buildSpec(drawSeed int64) (*WorkloadSpec, string, int64, int64) {
	return genLazySupportedSpec(rand.New(rand.NewSource(drawSeed)))
}

// TestProperty_EagerEqualsLazy_RequestStreams is the core BC-1/BC-2 property:
// across ≥100 random (seed, spec) draws, the eager and lazy generators produce
// byte-identical request streams (and identical session blueprints for
// reasoning specs). Reuses the field-by-field comparator from stream_test.go.
//
// On divergence the failure names the base seed, the draw index, and the spec
// label — enough to reconstruct the exact case as a regression unit test
// (BC-2). Determinism of the test itself (INV-6): the draw RNG is seeded from a
// fixed constant, so the same draw index always yields the same spec.
func TestProperty_EagerEqualsLazy_RequestStreams(t *testing.T) {
	draws := propertyDraws(t)
	drawRNG := rand.New(rand.NewSource(propertyBaseSeed))

	// Coverage guards: the suite must not pass vacuously. Track that we
	// actually exercised non-empty streams and at least one reasoning draw
	// (the trickiest path). If neither is ever hit the generator or caps
	// regressed and the test is not testing what it claims.
	var sawNonEmpty, sawReasoning, sawBlueprints bool

	for i := 0; i < draws; i++ {
		drawSeed := drawRNG.Int63()

		eagerSpec, label, horizon, maxReq := buildSpec(drawSeed)
		lazySpec, _, _, _ := buildSpec(drawSeed)

		isReasoning := eagerSpec.Clients[0].Reasoning != nil

		wl, err := GenerateWorkload(eagerSpec, horizon, maxReq)
		if err != nil {
			t.Fatalf("draw %d [base=%#x] eager GenerateWorkload failed: %v\n  spec: %s",
				i, propertyBaseSeed, err, label)
		}
		src, lazySessions, _, err := GenerateWorkloadLazy(lazySpec, horizon, maxReq)
		if err != nil {
			t.Fatalf("draw %d [base=%#x] lazy GenerateWorkloadLazy failed: %v\n  spec: %s\n  "+
				"(a lazy sentinel error here means the generator emitted an unsupported "+
				"shape — the generator must only produce lazy-supported specs)",
				i, propertyBaseSeed, err, label)
		}
		lazyReqs := drainLazy(t, src)

		// Wrap the shared comparator so a failure carries the draw context.
		// assertRequestStreamsEqual calls t.Fatalf itself; we front-load a
		// Logf so the draw index/label precede its field-level message.
		if len(wl.Requests) != len(lazyReqs) {
			t.Fatalf("draw %d [base=%#x, seed=%d]: stream length eager=%d lazy=%d\n  spec: %s",
				i, propertyBaseSeed, eagerSpec.Seed, len(wl.Requests), len(lazyReqs), label)
		}
		if len(lazyReqs) > 0 {
			sawNonEmpty = true
		}
		// Field-by-field equality (reuses stream_test.go helper). On mismatch
		// this Fatalf's with the offending field; the preceding context is in
		// the t.Log below so failures are self-describing.
		t.Logf("draw %d [base=%#x, seed=%d]: comparing %d requests — spec: %s",
			i, propertyBaseSeed, eagerSpec.Seed, len(lazyReqs), label)
		assertRequestStreamsEqual(t, wl.Requests, lazyReqs)

		if isReasoning {
			sawReasoning = true
			if len(wl.Sessions) != len(lazySessions) {
				t.Fatalf("draw %d [base=%#x, seed=%d]: blueprint count eager=%d lazy=%d\n  spec: %s",
					i, propertyBaseSeed, eagerSpec.Seed, len(wl.Sessions), len(lazySessions), label)
			}
			if len(lazySessions) > 0 {
				sawBlueprints = true
			}
			for j := range wl.Sessions {
				if wl.Sessions[j].SessionID != lazySessions[j].SessionID {
					t.Fatalf("draw %d [base=%#x, seed=%d]: blueprint %d SessionID eager=%q lazy=%q\n  spec: %s",
						i, propertyBaseSeed, eagerSpec.Seed, j,
						wl.Sessions[j].SessionID, lazySessions[j].SessionID, label)
				}
				if wl.Sessions[j].ClientID != lazySessions[j].ClientID {
					t.Fatalf("draw %d [base=%#x, seed=%d]: blueprint %d ClientID eager=%q lazy=%q\n  spec: %s",
						i, propertyBaseSeed, eagerSpec.Seed, j,
						wl.Sessions[j].ClientID, lazySessions[j].ClientID, label)
				}
				// A blueprint's RNG seeds the follow-up round stream. If the
				// two blueprints were seeded identically, one draw from each
				// must match — this catches a shifted blueprintRNG (the
				// tight-cap pre-pass bug class from PR #1453). The draw is
				// destructive but both RNGs are discarded after.
				if e, l := wl.Sessions[j].RNG.Int63(), lazySessions[j].RNG.Int63(); e != l {
					t.Fatalf("draw %d [base=%#x, seed=%d]: blueprint %d (session %q) RNG diverged eager=%d lazy=%d — blueprintRNG seeds shifted\n  spec: %s",
						i, propertyBaseSeed, eagerSpec.Seed, j, wl.Sessions[j].SessionID, e, l, label)
				}
			}
		}
	}

	if !sawNonEmpty {
		t.Fatalf("no draw produced a non-empty request stream across %d draws — "+
			"the spec generator or maxRequests cap regressed; the property is vacuous", draws)
	}
	if !sawReasoning {
		t.Fatalf("no reasoning draw occurred across %d draws — the trickiest lazy path "+
			"(single-session blueprint pre-pass) is unexercised; adjust the generator's reasoning probability", draws)
	}
	if !sawBlueprints {
		t.Fatalf("reasoning draws occurred but none produced session blueprints across %d draws — "+
			"blueprint parity is unexercised; adjust caps/horizon so round-0 requests survive", draws)
	}
}
