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
// As of #1460 the lazy generator supports EVERY workload class, so
// genLazySupportedSpec now spans the full space: single-shot, single- and
// multi-session reasoning (#1458), concurrency clients (#1459), and time-varying
// per-window workloads (#1460). GenerateWorkloadLazy returns no ErrLazyUnsupported*
// sentinel for any of them, so every draw is a genuine eager-vs-lazy comparison
// (not eager-vs-eager). Coverage guards below assert each hard shape was actually
// exercised so the property cannot pass vacuously.
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
//   - optional reasoning (MaxRounds 1–8, accumulate/none), single- OR
//     multi-session (SingleSession true/false — both lazy-supported as of #1458)
//
// Also dispatches (each ~1/4) to genConcurrencySpec (#1459) and genTimeVaryingSpec
// (#1460), which build their own self-contained specs. As of #1460 EVERY shape is
// lazy-supported — there is no unsupported class to avoid.
func genLazySupportedSpec(rng *rand.Rand) (spec *WorkloadSpec, label string, horizon, maxRequests int64) {
	seed := rng.Int63()

	// ~1/4 of draws are a CONCURRENCY shape (#1459). Concurrency clients cannot
	// mix with reasoning/multi-turn (spec.Validate hard-errors that) and carry no
	// RateFraction, so a concurrency draw builds its own spec — no cohorts, no
	// reasoning, no prefix-group entanglement — and returns early. This keeps the
	// draw byte-identical between eager and lazy while exercising the individual-
	// seed-heap-entry merge, including the ≥2-concurrency-client interleave case.
	if rng.Intn(4) == 0 {
		return genConcurrencySpec(rng, seed)
	}

	// ~1/4 of the remaining draws are a TIME-VARYING shape (#1460): clients with
	// per-window trace_rate/input_distribution overrides, including single-shot,
	// single-session, and multi-session reasoning variants. Built as its own spec
	// (own RateFraction/lifecycle layout) and returned early.
	if rng.Intn(4) == 0 {
		return genTimeVaryingSpec(rng, seed)
	}

	// Decide overall shape once so eager and lazy specs are built identically.
	numClients := 1 + rng.Intn(4) // 1–4 explicit clients
	useCohort := rng.Intn(2) == 0
	usePrefix := rng.Intn(2) == 0
	useReasoning := rng.Intn(3) == 0 // ~1/3 of draws are reasoning
	accumulate := rng.Intn(2) == 0
	// SingleSession true/false are both lazy-supported (#1458). Sample both so
	// the property covers the multi-session per-client merge, not just the
	// single-session path.
	singleSession := rng.Intn(2) == 0
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
				SingleSession: singleSession, // both single- and multi-session are lazy-supported (#1458)
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

	label = fmt.Sprintf("clients=%d cohort=%d process=%s prefix=%v reasoning=%v(rounds=%d,accum=%v,single=%v) maxReq=%d",
		numClients, cohortMembers, process, usePrefix, useReasoning, maxRounds, accumulate, singleSession, maxRequests)
	return spec, label, horizon, maxRequests
}

// genConcurrencySpec samples a lazy-supported CONCURRENCY spec (#1459): 1–3
// concurrency clients (distinct Concurrency and ThinkTimeUs so their staggered
// seed arrivals interleave — the CRITICAL-1 case) plus 0–2 open-loop single-shot
// clients. Never emits reasoning (mutually exclusive with concurrency per
// spec.Validate) or cohorts (cohorts are always rate-based and would force an
// AggregateRate requirement, hiding the all-concurrency path). The proportional
// rate split covers only the open-loop clients; concurrency clients carry no
// RateFraction. When there are zero open-loop clients the spec is all-concurrency
// and AggregateRate is 0 (not required — spec.Validate allows it).
func genConcurrencySpec(rng *rand.Rand, seed int64) (spec *WorkloadSpec, label string, horizon, maxRequests int64) {
	numConc := 1 + rng.Intn(3)    // 1–3 concurrency pools
	numOpen := rng.Intn(3)        // 0–2 open-loop clients
	usePrefix := rng.Intn(2) == 0 // shared prefix group across both kinds
	process := pick(rng, []string{"poisson", "gamma", "weibull", "constant"})

	prefixGroup := ""
	prefixLen := 0
	if usePrefix {
		prefixGroup = "grp"
		prefixLen = 32 + rng.Intn(96)
	}
	inputDist := DistSpec{Type: "gaussian", Params: map[string]float64{
		"mean": float64(40 + rng.Intn(80)), "std_dev": float64(5 + rng.Intn(15)),
		"min": 8, "max": 400,
	}}
	outputDist := DistSpec{Type: "exponential", Params: map[string]float64{
		"mean": float64(20 + rng.Intn(40)),
	}}

	clients := make([]ClientSpec, 0, numConc+numOpen)
	// Concurrency clients first (matches allClients order = spec.Clients order).
	for i := 0; i < numConc; i++ {
		clients = append(clients, ClientSpec{
			ID:           fmt.Sprintf("conc%d", i),
			TenantID:     fmt.Sprintf("tconc%d", i),
			SLOClass:     "batch",
			Concurrency:  1 + rng.Intn(8),                   // 1–8 users; distinct across pools with high probability
			ThinkTimeUs:  int64(50_000 + rng.Intn(250_000)), // distinct stagger cadence
			InputDist:    inputDist,
			OutputDist:   outputDist,
			PrefixGroup:  prefixGroup,
			PrefixLength: prefixLen,
		})
	}
	// Open-loop clients share the aggregate rate equally.
	aggRate := 0.0
	if numOpen > 0 {
		aggRate = 2.0 + float64(rng.Intn(18))
		frac := 1.0 / float64(numOpen)
		for i := 0; i < numOpen; i++ {
			clients = append(clients, ClientSpec{
				ID:           fmt.Sprintf("ol%d", i),
				TenantID:     fmt.Sprintf("tol%d", i),
				SLOClass:     "batch",
				RateFraction: frac,
				Arrival:      ArrivalSpec{Process: process},
				InputDist:    inputDist,
				OutputDist:   outputDist,
				PrefixGroup:  prefixGroup,
				PrefixLength: prefixLen,
			})
		}
	}

	spec = &WorkloadSpec{
		Version:       "2",
		Seed:          seed,
		Category:      "language",
		AggregateRate: aggRate, // 0 when all-concurrency (allowed)
		Clients:       clients,
	}
	maxRequests = int64(10 + rng.Intn(110)) // 10–119
	horizon = 60_000_000
	label = fmt.Sprintf("CONCURRENCY conc=%d open=%d process=%s prefix=%v maxReq=%d",
		numConc, numOpen, process, usePrefix, maxRequests)
	return spec, label, horizon, maxRequests
}

// genTimeVaryingSpec samples a lazy-supported TIME-VARYING spec (#1460): a single
// rate-based client whose lifecycle carries 1–4 windows, EACH with a per-window
// trace_rate override (which is what trips hasPerWindowParameters), some windows
// additionally overriding input/output distributions. Absolute-rate mode
// (aggregate_rate: 0) so per-window trace_rate is used verbatim. Windows are
// contiguous and non-overlapping here (the OutOfOrder/overlap cases are covered by
// dedicated unit tests); this generator's job is randomized coverage of per-window
// sampler switching + reasoning-under-TV. ~1/3 of TV draws are reasoning, split
// between single- and multi-session so the D5 per-batch stable sort is exercised.
func genTimeVaryingSpec(rng *rand.Rand, seed int64) (spec *WorkloadSpec, label string, horizon, maxRequests int64) {
	numWindows := 1 + rng.Intn(4) // 1–4 windows
	usePrefix := rng.Intn(2) == 0
	useReasoning := rng.Intn(3) == 0
	accumulate := rng.Intn(2) == 0
	singleSession := rng.Intn(2) == 0
	process := pick(rng, []string{"poisson", "gamma", "weibull", "constant"})

	prefixGroup, prefixLen := "", 0
	if usePrefix {
		prefixGroup = "grp"
		prefixLen = 32 + rng.Intn(96)
	}
	baseInput := DistSpec{Type: "gaussian", Params: map[string]float64{
		"mean": float64(40 + rng.Intn(80)), "std_dev": float64(5 + rng.Intn(15)), "min": 8, "max": 400,
	}}
	baseOutput := DistSpec{Type: "exponential", Params: map[string]float64{"mean": float64(20 + rng.Intn(40))}}

	var reasoning *ReasoningSpec
	maxRounds := 0
	if useReasoning {
		maxRounds = 1 + rng.Intn(6)
		growth := "none"
		if accumulate {
			growth = "accumulate"
		}
		reasoning = &ReasoningSpec{MultiTurn: &MultiTurnSpec{
			MaxRounds: maxRounds, ContextGrowth: growth,
			ThinkTimeUs: 50_000 + int64(rng.Intn(100_000)), SingleSession: singleSession,
		}}
	}

	// Contiguous windows of ~1s each, each with a distinct per-window trace_rate;
	// some windows also override the input distribution (exercises sampler swap).
	// ~1/3 of draws use OVERLAPPING windows (a half-duration stride) so windows are
	// co-active in arrival time — the regime where the emit-safety gate's build-ahead
	// path does load-bearing work (a live window's head withheld until a co-active
	// window is built). The rest are contiguous.
	const windowDur = 1_000_000
	overlap := rng.Intn(3) == 0
	stride := int64(windowDur)
	if overlap {
		stride = windowDur / 2
	}
	windows := make([]ActiveWindow, numWindows)
	for i := 0; i < numWindows; i++ {
		rate := 2.0 + float64(rng.Intn(18)) // 2–19 req/s
		w := ActiveWindow{
			StartUs:   int64(i) * stride,
			EndUs:     int64(i)*stride + windowDur,
			TraceRate: &rate,
		}
		if rng.Intn(2) == 0 {
			wIn := DistSpec{Type: "gaussian", Params: map[string]float64{
				"mean": float64(60 + rng.Intn(120)), "std_dev": float64(8 + rng.Intn(20)), "min": 8, "max": 600,
			}}
			w.InputDist = &wIn
		}
		windows[i] = w
	}

	// Rate mode: ~half absolute (aggregate_rate=0, per-window trace_rate used
	// verbatim), ~half PROPORTIONAL (aggregate_rate>0, per-window trace_rate is a
	// weight fed to computeProportionalRate's co-active summation). In proportional
	// mode ~half of draws add a WINDOWLESS always-on client — legal only in
	// proportional mode (absolute mode requires every rate-based client to carry
	// per-window trace_rate) — which the TV path warns-and-skips (generating
	// nothing) but which still contributes its RateFraction to the co-active
	// denominator. This exercises both the proportional co-active path and the
	// windowless-skip RNG-parity (windowless clients get NO clientSeed draw).
	proportional := rng.Intn(2) == 0
	aggregateRate := 0.0
	addWindowless := false
	if proportional {
		aggregateRate = float64(50 + rng.Intn(150)) // 50–199 req/s
		addWindowless = rng.Intn(2) == 0
	}

	clients := []ClientSpec{{
		ID: "tv", TenantID: "t0", SLOClass: "batch", RateFraction: 1.0,
		Arrival:      ArrivalSpec{Process: process},
		InputDist:    baseInput,
		OutputDist:   baseOutput,
		PrefixGroup:  prefixGroup,
		PrefixLength: prefixLen,
		Reasoning:    reasoning,
		Lifecycle:    &LifecycleSpec{Windows: windows},
	}}
	if addWindowless {
		clients = append(clients, ClientSpec{
			ID: "always-on", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: process},
			InputDist:  baseInput,
			OutputDist: baseOutput,
			// No Lifecycle: an always-on client. In the TV path it is warned-and-
			// skipped (generates nothing) but contributes RateFraction to the
			// co-active denominator in computeProportionalRate.
		})
	}

	spec = &WorkloadSpec{
		Version: "2", Seed: seed, Category: "language",
		AggregateRate: aggregateRate,
		Clients:       clients,
	}
	// Horizon covers all windows (last window ends at (n-1)*stride + windowDur).
	horizon = int64(numWindows-1)*stride + windowDur + 500_000
	maxRequests = int64(10 + rng.Intn(80))
	if useReasoning && rng.Intn(4) == 0 {
		maxRequests = int64(2 + rng.Intn(8))
	}
	label = fmt.Sprintf("TIME-VARYING windows=%d overlap=%v proportional=%v windowless=%v process=%s prefix=%v reasoning=%v(rounds=%d,accum=%v,single=%v) maxReq=%d",
		numWindows, overlap, proportional, addWindowless, process, usePrefix, useReasoning, maxRounds, accumulate, singleSession, maxRequests)
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
	// actually exercised non-empty streams, at least one reasoning draw
	// (the trickiest path), at least one MULTI-session reasoning draw (#1458,
	// the per-client live-session merge), at least one CONCURRENCY draw (#1459)
	// and at least one MULTI-concurrency-client draw (the individual-seed-heap-
	// entry interleave). If any is never hit the generator or caps regressed and
	// the test is not testing what it claims.
	var sawNonEmpty, sawReasoning, sawBlueprints, sawMultiSession bool
	var sawConcurrency, sawMultiConcurrency bool
	var sawTimeVarying, sawTimeVaryingMultiSession bool
	var sawTimeVaryingProportional, sawTimeVaryingOverlap, sawTimeVaryingWindowless bool

	for i := 0; i < draws; i++ {
		drawSeed := drawRNG.Int63()

		eagerSpec, label, horizon, maxReq := buildSpec(drawSeed)
		lazySpec, _, _, _ := buildSpec(drawSeed)

		isReasoning := eagerSpec.Clients[0].Reasoning != nil
		isMultiSession := isReasoning && eagerSpec.Clients[0].Reasoning.MultiTurn != nil &&
			!eagerSpec.Clients[0].Reasoning.MultiTurn.SingleSession
		if isMultiSession {
			sawMultiSession = true
		}
		// Time-varying draws (#1460): per-window parameter overrides. Track the
		// multi-session-under-TV combination separately — it is the D5 per-batch
		// stable-sort path, which must not go unexercised (else the guard is vacuous).
		// Also track proportional-mode, overlapping-window, and windowless-client
		// sub-shapes so those distinct code paths (computeProportionalRate co-active
		// summation, the build-ahead emit gate, and the windowless warn+skip) can't
		// silently go uncovered.
		isTimeVarying := hasPerWindowParameters(eagerSpec.Clients)
		if isTimeVarying {
			sawTimeVarying = true
			if isMultiSession {
				sawTimeVaryingMultiSession = true
			}
			if eagerSpec.AggregateRate > 0 {
				sawTimeVaryingProportional = true
			}
			for ci := range eagerSpec.Clients {
				c := &eagerSpec.Clients[ci]
				if c.Lifecycle == nil || len(c.Lifecycle.Windows) == 0 {
					sawTimeVaryingWindowless = true
					continue
				}
				// Overlap: any two windows of this client whose [start,end) intersect.
				for a := range c.Lifecycle.Windows {
					for b := a + 1; b < len(c.Lifecycle.Windows); b++ {
						wa, wb := c.Lifecycle.Windows[a], c.Lifecycle.Windows[b]
						if wa.StartUs < wb.EndUs && wb.StartUs < wa.EndUs {
							sawTimeVaryingOverlap = true
						}
					}
				}
			}
		}
		// Count concurrency clients in this draw (spec.Clients-only field).
		nConc := 0
		for ci := range eagerSpec.Clients {
			if eagerSpec.Clients[ci].Concurrency > 0 {
				nConc++
			}
		}
		if nConc > 0 {
			sawConcurrency = true
		}
		if nConc >= 2 {
			sawMultiConcurrency = true
		}

		wl, err := GenerateWorkload(eagerSpec, horizon, maxReq)
		if err != nil {
			t.Fatalf("draw %d [base=%#x] eager GenerateWorkload failed: %v\n  spec: %s",
				i, propertyBaseSeed, err, label)
		}
		src, lazySessions, lazyBudget, err := GenerateWorkloadLazy(lazySpec, horizon, maxReq)
		if err != nil {
			t.Fatalf("draw %d [base=%#x] lazy GenerateWorkloadLazy failed: %v\n  spec: %s\n  "+
				"(as of #1460 there are no unsupported-shape sentinels — an error here is a "+
				"real generation bug, since eager accepted the same spec above)",
				i, propertyBaseSeed, err, label)
		}
		// FollowUpBudget must match eager wherever it is observable — i.e. for
		// concurrency specs (BC-3, this PR's contract) and for any spec that
		// produced session blueprints (cmd only reads FollowUpBudget when
		// len(Sessions) > 0). Pure open-loop specs with no sessions are excluded:
		// eager's sessionless early return leaves the field at its Go zero value
		// (0) while lazy returns -1 — a pre-existing, inert difference (the value
		// is never read without sessions) that predates and is out of scope for
		// #1459.
		if nConc > 0 || len(wl.Sessions) > 0 {
			if wl.FollowUpBudget != lazyBudget {
				t.Fatalf("draw %d [base=%#x, seed=%d]: FollowUpBudget eager=%d lazy=%d\n  spec: %s",
					i, propertyBaseSeed, eagerSpec.Seed, wl.FollowUpBudget, lazyBudget, label)
			}
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
	if !sawMultiSession {
		t.Fatalf("no multi-session (SingleSession=false) reasoning draw occurred across %d draws — "+
			"the per-client live-session merge (#1458) is unexercised; adjust the generator's SingleSession probability", draws)
	}
	if !sawBlueprints {
		t.Fatalf("reasoning draws occurred but none produced session blueprints across %d draws — "+
			"blueprint parity is unexercised; adjust caps/horizon so round-0 requests survive", draws)
	}
	if !sawConcurrency {
		t.Fatalf("no concurrency (Concurrency>0) draw occurred across %d draws — "+
			"the lazy concurrency seed path (#1459) is unexercised; adjust genConcurrencySpec probability", draws)
	}
	if !sawMultiConcurrency {
		t.Fatalf("no draw had ≥2 concurrency clients across %d draws — the individual-seed-heap-entry "+
			"interleave (CRITICAL-1, #1459) is unexercised; adjust genConcurrencySpec's numConc distribution", draws)
	}
	if !sawTimeVarying {
		t.Fatalf("no time-varying (per-window parameter) draw occurred across %d draws — "+
			"the lazy time-varying path (#1460) is unexercised; adjust genTimeVaryingSpec probability", draws)
	}
	if !sawTimeVaryingMultiSession {
		t.Fatalf("no time-varying + multi-session-reasoning draw occurred across %d draws — the "+
			"per-batch stable-sort (D5, #1460) that makes non-monotonic multi-session windows "+
			"byte-identical is unexercised; adjust genTimeVaryingSpec's reasoning/SingleSession probability", draws)
	}
	if !sawTimeVaryingProportional {
		t.Fatalf("no PROPORTIONAL-mode time-varying draw (aggregate_rate>0) occurred across %d draws — "+
			"computeProportionalRate's co-active summation is unexercised; adjust genTimeVaryingSpec's proportional probability", draws)
	}
	if !sawTimeVaryingOverlap {
		t.Fatalf("no OVERLAPPING-window time-varying draw occurred across %d draws — the emit-gate "+
			"build-ahead path (#1460) is unexercised in the property loop; adjust genTimeVaryingSpec's overlap probability", draws)
	}
	if !sawTimeVaryingWindowless {
		t.Fatalf("no time-varying draw with a WINDOWLESS always-on client occurred across %d draws — "+
			"the warn+skip RNG-parity path (#1460) is unexercised; adjust genTimeVaryingSpec's windowless probability", draws)
	}
}
