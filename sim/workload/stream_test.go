package workload

import (
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// drainLazy collects every request a lazyRequestSource emits, then verifies
// exhaustion stickiness on a follow-up Next() call (BC-3 contract semantics).
func drainLazy(t *testing.T, src *lazyRequestSource) []*sim.Request {
	t.Helper()
	var out []*sim.Request
	for {
		r, ok := src.Next()
		if !ok {
			break
		}
		out = append(out, r)
	}
	// Exhaustion must be sticky.
	if r, ok := src.Next(); ok || r != nil {
		t.Fatalf("Next after exhaustion returned (%v, %v); want (nil, false)", r, ok)
	}
	return out
}

// assertRequestStreamsEqual compares two request sequences field-by-field.
// Used by all eager ≡ lazy byte-identity tests (BC-3 / BC-4).
func assertRequestStreamsEqual(t *testing.T, eager, lazy []*sim.Request) {
	t.Helper()
	if len(eager) != len(lazy) {
		t.Fatalf("length mismatch: eager=%d lazy=%d", len(eager), len(lazy))
	}
	for i := range eager {
		e, l := eager[i], lazy[i]
		if e.ID != l.ID {
			t.Fatalf("request %d: ID eager=%q lazy=%q", i, e.ID, l.ID)
		}
		if e.ArrivalTime != l.ArrivalTime {
			t.Fatalf("request %d (%s): ArrivalTime eager=%d lazy=%d", i, e.ID, e.ArrivalTime, l.ArrivalTime)
		}
		if len(e.InputTokens) != len(l.InputTokens) {
			t.Fatalf("request %d (%s): InputTokens length eager=%d lazy=%d", i, e.ID, len(e.InputTokens), len(l.InputTokens))
		}
		for j := range e.InputTokens {
			if e.InputTokens[j] != l.InputTokens[j] {
				t.Fatalf("request %d (%s): InputTokens[%d] eager=%d lazy=%d", i, e.ID, j, e.InputTokens[j], l.InputTokens[j])
			}
		}
		if len(e.OutputTokens) != len(l.OutputTokens) {
			t.Fatalf("request %d (%s): OutputTokens length eager=%d lazy=%d", i, e.ID, len(e.OutputTokens), len(l.OutputTokens))
		}
		for j := range e.OutputTokens {
			if e.OutputTokens[j] != l.OutputTokens[j] {
				t.Fatalf("request %d (%s): OutputTokens[%d] eager=%d lazy=%d", i, e.ID, j, e.OutputTokens[j], l.OutputTokens[j])
			}
		}
		if e.MaxOutputLen != l.MaxOutputLen {
			t.Fatalf("request %d (%s): MaxOutputLen eager=%d lazy=%d", i, e.ID, e.MaxOutputLen, l.MaxOutputLen)
		}
		if e.TenantID != l.TenantID {
			t.Fatalf("request %d: TenantID eager=%q lazy=%q", i, e.TenantID, l.TenantID)
		}
		if e.SLOClass != l.SLOClass {
			t.Fatalf("request %d: SLOClass eager=%q lazy=%q", i, e.SLOClass, l.SLOClass)
		}
		if e.Model != l.Model {
			t.Fatalf("request %d: Model eager=%q lazy=%q", i, e.Model, l.Model)
		}
		if e.ClientID != l.ClientID {
			t.Fatalf("request %d: ClientID eager=%q lazy=%q", i, e.ClientID, l.ClientID)
		}
		if e.PrefixGroup != l.PrefixGroup {
			t.Fatalf("request %d: PrefixGroup eager=%q lazy=%q", i, e.PrefixGroup, l.PrefixGroup)
		}
		if e.PrefixLength != l.PrefixLength {
			t.Fatalf("request %d: PrefixLength eager=%d lazy=%d", i, e.PrefixLength, l.PrefixLength)
		}
		if e.Deadline != l.Deadline {
			t.Fatalf("request %d: Deadline eager=%d lazy=%d", i, e.Deadline, l.Deadline)
		}
		if e.SLOTargetUs != l.SLOTargetUs {
			t.Fatalf("request %d: SLOTargetUs eager=%d lazy=%d", i, e.SLOTargetUs, l.SLOTargetUs)
		}
		if e.SessionID != l.SessionID {
			t.Fatalf("request %d: SessionID eager=%q lazy=%q", i, e.SessionID, l.SessionID)
		}
		if e.RoundIndex != l.RoundIndex {
			t.Fatalf("request %d: RoundIndex eager=%d lazy=%d", i, e.RoundIndex, l.RoundIndex)
		}
		// ReasonRatio is drawn from the same RNG state in both paths; compare it
		// so a future reordering of reasoning-round RNG draws is caught here too.
		if e.ReasonRatio != l.ReasonRatio {
			t.Fatalf("request %d (%s): ReasonRatio eager=%v lazy=%v", i, e.ID, e.ReasonRatio, l.ReasonRatio)
		}
		if e.Streaming != l.Streaming {
			t.Fatalf("request %d: Streaming eager=%v lazy=%v", i, e.Streaming, l.Streaming)
		}
		// Multimodal token counts (populated by produceNextSingleShot's
		// multimodal branch; mirrored from the eager path).
		if e.TextTokenCount != l.TextTokenCount {
			t.Fatalf("request %d: TextTokenCount eager=%d lazy=%d", i, e.TextTokenCount, l.TextTokenCount)
		}
		if e.ImageTokenCount != l.ImageTokenCount {
			t.Fatalf("request %d: ImageTokenCount eager=%d lazy=%d", i, e.ImageTokenCount, l.ImageTokenCount)
		}
		if e.AudioTokenCount != l.AudioTokenCount {
			t.Fatalf("request %d: AudioTokenCount eager=%d lazy=%d", i, e.AudioTokenCount, l.AudioTokenCount)
		}
		if e.VideoTokenCount != l.VideoTokenCount {
			t.Fatalf("request %d: VideoTokenCount eager=%d lazy=%d", i, e.VideoTokenCount, l.VideoTokenCount)
		}
	}
}

// singleClientChatbotSpec builds a single-client open-loop spec — the
// simplest case that exercises every step of the streaming path
// (samplers, prefix, deadline) without reasoning complications.
func singleClientChatbotSpec(seed int64) *WorkloadSpec {
	return &WorkloadSpec{
		Version: "1", Seed: seed, Category: "language", AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", TenantID: "t1", SLOClass: "batch",
			RateFraction: 1.0,
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
}

// twoClientConstantArrivalSpec returns a spec with two clients producing
// requests at identical (constant) arrival cadence — used to deliberately
// generate arrival-time ties so the heap tie-breaker (clientIdx) is
// exercised.
func twoClientConstantArrivalSpec(seed int64) *WorkloadSpec {
	mk := func(id string) ClientSpec {
		return ClientSpec{
			ID: id, TenantID: id + "-tenant", SLOClass: "batch",
			RateFraction: 0.5,
			Arrival:      ArrivalSpec{Process: "constant"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 80, "std_dev": 10, "min": 10, "max": 500}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 40}},
		}
	}
	return &WorkloadSpec{
		Version: "1", Seed: seed, Category: "language", AggregateRate: 10.0,
		Clients: []ClientSpec{mk("c1"), mk("c2")},
	}
}

// reasoningSingleSessionSpec returns a spec with two closed-loop multi-turn
// reasoning clients in SingleSession=true mode — the inference-perf shape
// and the issue's reproducer shape. Each client represents one persistent
// session cycling through rounds.
func reasoningSingleSessionSpec(seed int64) *WorkloadSpec {
	mk := func(id string) ClientSpec {
		return ClientSpec{
			ID: id, TenantID: id + "-t", SLOClass: "batch",
			RateFraction: 0.5,
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 60, "std_dev": 10, "min": 10, "max": 200}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 30}},
			Reasoning: &ReasoningSpec{
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     3,
					ContextGrowth: "accumulate",
					ThinkTimeUs:   100_000,
					SingleSession: true,
				},
			},
		}
	}
	return &WorkloadSpec{
		Version: "1", Seed: seed, Category: "language", AggregateRate: 4.0,
		Clients: []ClientSpec{mk("r1"), mk("r2")},
	}
}

// reasoningMultiSessionSpec builds a single multi-turn reasoning client with
// SingleSession=false (the default): a traffic source that spawns many
// independent, overlapping sessions over the horizon. Lazy supports this via
// the per-client live-session merge (#1458); the multi-session parity tests
// use it as the eager-vs-lazy oracle case.
func reasoningMultiSessionSpec(seed int64) *WorkloadSpec {
	return &WorkloadSpec{
		Version: "1", Seed: seed, Category: "language", AggregateRate: 4.0,
		Clients: []ClientSpec{{
			ID: "r1", TenantID: "rt1", SLOClass: "batch",
			RateFraction: 1.0,
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 60, "std_dev": 10, "min": 10, "max": 200}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 30}},
			Reasoning: &ReasoningSpec{
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     3,
					ContextGrowth: "accumulate",
					ThinkTimeUs:   100_000,
					// SingleSession: false (default)
				},
			},
		}},
	}
}

// TestLazyRequestSource_SingleClient_NonReasoning_MatchesEager pins BC-3
// for the simplest non-reasoning case: a single open-loop client. The eager
// generator's slice and the lazy source's drain must be byte-identical.
func TestLazyRequestSource_SingleClient_NonReasoning_MatchesEager(t *testing.T) {
	spec := singleClientChatbotSpec(42)
	specCopy := *spec
	specCopy.Clients = append([]ClientSpec{}, spec.Clients...)

	eager, err := GenerateRequests(spec, 10_000_000, 50)
	if err != nil {
		t.Fatalf("eager: %v", err)
	}
	src, _, _, err := GenerateWorkloadLazy(&specCopy, 10_000_000, 50)
	if err != nil {
		t.Fatalf("lazy: %v", err)
	}
	lazy := drainLazy(t, src)
	assertRequestStreamsEqual(t, eager, lazy)
}

// TestLazyRequestSource_TieBreakerByClientIndex pins BC-7: when two clients
// produce identical-arrival-time requests, the lower-index client wins
// the heap pop — matching sort.SliceStable's behavior in eager mode.
func TestLazyRequestSource_TieBreakerByClientIndex(t *testing.T) {
	spec := twoClientConstantArrivalSpec(7)
	specCopy := *spec
	specCopy.Clients = append([]ClientSpec{}, spec.Clients...)

	eager, err := GenerateRequests(spec, 5_000_000, 10)
	if err != nil {
		t.Fatalf("eager: %v", err)
	}
	src, _, _, err := GenerateWorkloadLazy(&specCopy, 5_000_000, 10)
	if err != nil {
		t.Fatalf("lazy: %v", err)
	}
	lazy := drainLazy(t, src)
	assertRequestStreamsEqual(t, eager, lazy)
}

// TestLazyRequestSource_IDsSequentialInArrivalOrder pins BC-7's ID format:
// IDs are "request_<i>" in heap-pop (arrival) order.
func TestLazyRequestSource_IDsSequentialInArrivalOrder(t *testing.T) {
	spec := singleClientChatbotSpec(99)
	src, _, _, err := GenerateWorkloadLazy(spec, 5_000_000, 20)
	if err != nil {
		t.Fatalf("lazy: %v", err)
	}
	lazy := drainLazy(t, src)
	for i, r := range lazy {
		want := fmt.Sprintf("request_%d", i)
		if r.ID != want {
			t.Fatalf("request %d: ID=%q want=%q", i, r.ID, want)
		}
		if i > 0 && r.ArrivalTime < lazy[i-1].ArrivalTime {
			t.Fatalf("request %d: ArrivalTime=%d < prev=%d (out of arrival order)",
				i, r.ArrivalTime, lazy[i-1].ArrivalTime)
		}
	}
}

// TestGenerateWorkloadLazy_StopsAtMaxRequests pins BC-6: lazy mode counts
// emitted requests as it goes and stops at maxRequests (no per-client
// 2*maxRequests safety cap applied — that exists only in eager mode).
// The assertion is exact (== 7) so a regression that produces zero
// requests (e.g., the horizon being misinterpreted) would also fail.
func TestGenerateWorkloadLazy_StopsAtMaxRequests(t *testing.T) {
	// 60s horizon × ~10 req/s mean rate → ~600 candidate IATs; with
	// maxRequests=7 the source MUST stop exactly at 7. If the test ever
	// emits fewer, the horizon is wrong, the cap is being applied at the
	// wrong layer, or stream.go's Next() loop terminates early.
	spec := singleClientChatbotSpec(13)
	src, _, _, err := GenerateWorkloadLazy(spec, 60_000_000, 7)
	if err != nil {
		t.Fatalf("lazy: %v", err)
	}
	lazy := drainLazy(t, src)
	if int64(len(lazy)) != 7 {
		t.Fatalf("emitted %d requests; want exactly 7 (maxRequests cap should bind and horizon should not exhaust the source first)", len(lazy))
	}
}

// TestLazyRequestSource_TimeVarying_SingleWindow_MatchesEager pins BC-1/BC-4 for
// the simplest time-varying shape: one client, one lifecycle window carrying a
// per-window trace_rate in absolute-rate mode (aggregate_rate: 0). Before #1460
// this spec fell back to eager (ErrLazyUnsupportedTimeVarying); now the lazy
// streaming source must produce a byte-identical request stream.
func TestLazyRequestSource_TimeVarying_SingleWindow_MatchesEager(t *testing.T) {
	traceRate := 8.0
	mk := func() *WorkloadSpec {
		return &WorkloadSpec{
			Version: "2", Seed: 4242, Category: "language", AggregateRate: 0, // absolute rate mode
			Clients: []ClientSpec{{
				ID: "tv", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 96, "std_dev": 20, "min": 16, "max": 400}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 48}},
				Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{
					{StartUs: 0, EndUs: 5_000_000, TraceRate: &traceRate},
				}},
			}},
		}
	}
	eager, err := GenerateRequests(mk(), 6_000_000, 40)
	if err != nil {
		t.Fatalf("eager GenerateRequests: %v", err)
	}
	src, _, _, err := GenerateWorkloadLazy(mk(), 6_000_000, 40)
	if err != nil {
		t.Fatalf("lazy GenerateWorkloadLazy: %v", err)
	}
	lazyReqs := drainLazy(t, src)
	assertRequestStreamsEqual(t, eager, lazyReqs)
	if len(lazyReqs) == 0 {
		t.Fatal("time-varying single-window test ineffective: empty stream")
	}
}

// assertTimeVaryingParity generates spec eagerly and lazily and asserts the two
// request streams (and, for reasoning specs, the session blueprints) are
// byte-identical. mk must return a freshly-built spec on each call so the two
// generators cannot cross-contaminate (both run validateAndExpandSpec in place).
func assertTimeVaryingParity(t *testing.T, mk func() *WorkloadSpec, horizon, maxReq int64, wantNonEmpty bool) {
	t.Helper()
	eagerWl, err := GenerateWorkload(mk(), horizon, maxReq)
	if err != nil {
		t.Fatalf("eager GenerateWorkload: %v", err)
	}
	src, lazySessions, _, err := GenerateWorkloadLazy(mk(), horizon, maxReq)
	if err != nil {
		t.Fatalf("lazy GenerateWorkloadLazy: %v", err)
	}
	lazyReqs := drainLazy(t, src)
	assertRequestStreamsEqual(t, eagerWl.Requests, lazyReqs)
	if wantNonEmpty && len(lazyReqs) == 0 {
		t.Fatal("time-varying parity test ineffective: empty stream")
	}
	if len(eagerWl.Sessions) != len(lazySessions) {
		t.Fatalf("blueprint count eager=%d lazy=%d", len(eagerWl.Sessions), len(lazySessions))
	}
	for j := range eagerWl.Sessions {
		if eagerWl.Sessions[j].SessionID != lazySessions[j].SessionID {
			t.Fatalf("blueprint %d SessionID eager=%q lazy=%q", j, eagerWl.Sessions[j].SessionID, lazySessions[j].SessionID)
		}
		if eagerWl.Sessions[j].ClientID != lazySessions[j].ClientID {
			t.Fatalf("blueprint %d ClientID eager=%q lazy=%q", j, eagerWl.Sessions[j].ClientID, lazySessions[j].ClientID)
		}
		// Identically-seeded blueprint RNGs must produce the same first draw.
		if e, l := eagerWl.Sessions[j].RNG.Int63(), lazySessions[j].RNG.Int63(); e != l {
			t.Fatalf("blueprint %d (session %q) RNG diverged eager=%d lazy=%d", j, eagerWl.Sessions[j].SessionID, e, l)
		}
	}
}

// TestLazyRequestSource_TimeVarying_MultiWindow_MatchesEager pins BC-1/BC-2 for a
// client with multiple sequential windows carrying distinct per-window trace_rate
// AND a per-window input_distribution override, with the horizon clipping the last
// window. Exercises window-boundary handling, the EndUs clamp, and per-window
// sampler switching.
func TestLazyRequestSource_TimeVarying_MultiWindow_MatchesEager(t *testing.T) {
	r1, r2, r3 := 4.0, 12.0, 6.0
	windowInput := DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 200, "std_dev": 30, "min": 32, "max": 600}}
	mk := func() *WorkloadSpec {
		return &WorkloadSpec{
			Version: "2", Seed: 77, Category: "language", AggregateRate: 0,
			Clients: []ClientSpec{{
				ID: "diurnal", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 96, "std_dev": 20, "min": 16, "max": 400}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 48}},
				Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{
					{StartUs: 0, EndUs: 2_000_000, TraceRate: &r1},
					{StartUs: 2_000_000, EndUs: 4_000_000, TraceRate: &r2, InputDist: &windowInput},
					{StartUs: 4_000_000, EndUs: 8_000_000, TraceRate: &r3},
				}},
			}},
		}
	}
	// Horizon clips the middle of the last window (4s..8s → clamped to 5s).
	assertTimeVaryingParity(t, mk, 5_000_000, 60, true)
}

// TestLazyRequestSource_TimeVarying_OutOfOrderWindows_MatchesEager pins BC-2: the
// suffix-min emit gate must reproduce eager's global stable sort even when a
// client's lifecycle windows are listed OUT OF spec order (validation only checks
// end > start, not ordering). A later-in-spec window starting EARLIER than an
// earlier-in-spec window means the lazy source must build ahead before emitting.
func TestLazyRequestSource_TimeVarying_OutOfOrderWindows_MatchesEager(t *testing.T) {
	rA, rB, rC := 8.0, 5.0, 10.0
	mk := func() *WorkloadSpec {
		return &WorkloadSpec{
			Version: "2", Seed: 99, Category: "language", AggregateRate: 0,
			Clients: []ClientSpec{{
				ID: "ooo", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 80, "std_dev": 16, "min": 16, "max": 300}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 40}},
				// Spec order is [start=2s, start=4s, start=0s] — deliberately not sorted.
				Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{
					{StartUs: 2_000_000, EndUs: 3_000_000, TraceRate: &rA},
					{StartUs: 4_000_000, EndUs: 5_000_000, TraceRate: &rB},
					{StartUs: 0, EndUs: 1_000_000, TraceRate: &rC},
				}},
			}},
		}
	}
	assertTimeVaryingParity(t, mk, 6_000_000, 60, true)
}

// TestLazyRequestSource_TimeVarying_WithConcurrency_MatchesEager pins BC-1/BC-3:
// a time-varying rate-based lifecycle client coexisting with a concurrency client
// (no reasoning — spec.Validate forbids concurrency+multi-turn). The lazy TV path
// must still append concurrency seeds through the shared helper so blueprints and
// FollowUpBudget match eager.
func TestLazyRequestSource_TimeVarying_WithConcurrency_MatchesEager(t *testing.T) {
	tvRate := 6.0
	mk := func() *WorkloadSpec {
		return &WorkloadSpec{
			Version: "2", Seed: 1234, Category: "language", AggregateRate: 0, // absolute mode
			Clients: []ClientSpec{
				{
					ID: "tv", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
					Arrival:    ArrivalSpec{Process: "poisson"},
					InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 90, "std_dev": 20, "min": 16, "max": 400}},
					OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 48}},
					Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{
						{StartUs: 0, EndUs: 3_000_000, TraceRate: &tvRate},
					}},
				},
				{
					ID: "pool", TenantID: "t2", SLOClass: "batch",
					Concurrency: 4, ThinkTimeUs: 200_000,
					InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 90, "std_dev": 20, "min": 16, "max": 400}},
					OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 64}},
				},
			},
		}
	}
	// Also assert FollowUpBudget parity (observable because concurrency present).
	eagerWl, err := GenerateWorkload(mk(), 4_000_000, 40)
	if err != nil {
		t.Fatalf("eager GenerateWorkload: %v", err)
	}
	_, _, lazyBudget, err := GenerateWorkloadLazy(mk(), 4_000_000, 40)
	if err != nil {
		t.Fatalf("lazy GenerateWorkloadLazy: %v", err)
	}
	if eagerWl.FollowUpBudget != lazyBudget {
		t.Fatalf("FollowUpBudget eager=%d lazy=%d", eagerWl.FollowUpBudget, lazyBudget)
	}
	assertTimeVaryingParity(t, mk, 4_000_000, 40, true)
}

// TestLazyRequestSource_TimeVarying_Reasoning_MatchesEager pins BC-1/BC-3 for
// time-varying reasoning clients, both single-session and (crucially) MULTI-session
// (single_session:false). Multi-session windows are NOT arrival-monotonic in build
// order — the per-batch stable sort (D5) is what makes them byte-identical to
// eager's global sort. Sub-case (ii) is the D5 regression guard.
func TestLazyRequestSource_TimeVarying_Reasoning_MatchesEager(t *testing.T) {
	traceRate := 5.0
	mkReasoning := func(singleSession bool) func() *WorkloadSpec {
		return func() *WorkloadSpec {
			return &WorkloadSpec{
				Version: "2", Seed: 555, Category: "language", AggregateRate: 0,
				Clients: []ClientSpec{{
					ID: "r", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
					PrefixGroup: "sys", PrefixLength: 48,
					Arrival:    ArrivalSpec{Process: "poisson"},
					InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 80, "std_dev": 16, "min": 16, "max": 300}},
					OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 48}},
					Reasoning: &ReasoningSpec{MultiTurn: &MultiTurnSpec{
						MaxRounds: 3, ContextGrowth: "accumulate", ThinkTimeUs: 60_000, SingleSession: singleSession,
					}},
					Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{
						{StartUs: 0, EndUs: 4_000_000, TraceRate: &traceRate},
					}},
				}},
			}
		}
	}
	t.Run("single-session", func(t *testing.T) {
		assertTimeVaryingParity(t, mkReasoning(true), 5_000_000, 40, true)
	})
	t.Run("multi-session", func(t *testing.T) {
		assertTimeVaryingParity(t, mkReasoning(false), 5_000_000, 40, true)
	})
}

// TestLazyRequestSource_TimeVarying_InvalidWindowDist_Rejected pins BC-6: an
// invalid per-window input/output distribution override is rejected up front by
// spec.Validate (#1460), so BOTH GenerateWorkload (eager) and GenerateWorkloadLazy
// return the same construction-time error — regardless of maxRequests. This closes
// the parity gap where a late malformed window under a binding cap would abort the
// eager path (which builds every window) but never be built by the lazy path.
//
// Two failure kinds are exercised: an unknown distribution TYPE and a valid type
// with MISSING REQUIRED PARAMS (the subtler case — validateDistSpec accepts it,
// but generateRequestsForWindow's NewLengthSampler would reject it, so Validate
// must too). The bad window is placed SECOND with a tight maxRequests so that,
// absent the validation, the lazy source would fill the cap from the first window
// and never build the bad one — the exact silent-completion scenario prevented.
func TestLazyRequestSource_TimeVarying_InvalidWindowDist_Rejected(t *testing.T) {
	goodRate := 5.0
	cases := []struct {
		name    string
		badDist DistSpec
	}{
		{"unknown-type", DistSpec{Type: "not_a_real_distribution", Params: map[string]float64{}}},
		// Valid type, but gaussian requires mean/std_dev/min/max — none supplied.
		{"valid-type-missing-params", DistSpec{Type: "gaussian", Params: map[string]float64{}}},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			bad := tc.badDist
			mk := func() *WorkloadSpec {
				return &WorkloadSpec{
					Version: "2", Seed: 7, Category: "language", AggregateRate: 0,
					Clients: []ClientSpec{{
						ID: "tv", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
						Arrival:    ArrivalSpec{Process: "poisson"},
						InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 96, "std_dev": 20, "min": 16, "max": 400}},
						OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 48}},
						Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{
							// First window valid; the overridden output dist in the second is invalid.
							{StartUs: 0, EndUs: 2_000_000, TraceRate: &goodRate},
							{StartUs: 2_000_000, EndUs: 4_000_000, TraceRate: &goodRate, OutputDist: &bad},
						}},
					}},
				}
			}
			// Tight cap so an unvalidated bad late window would never be built by lazy.
			const horizon, maxReq = int64(5_000_000), int64(3)
			_, eagerErr := GenerateWorkload(mk(), horizon, maxReq)
			if eagerErr == nil {
				t.Fatal("eager: want a validation error for the invalid per-window output_distribution")
			}
			src, _, _, lazyErr := GenerateWorkloadLazy(mk(), horizon, maxReq)
			if lazyErr == nil {
				t.Fatalf("lazy: want the SAME validation error as eager (%v), got nil source=%v", eagerErr, src)
			}
			// Both must reject at construction; neither returns a usable source (parity).
			if src != nil {
				t.Errorf("lazy: source must be nil on a rejected spec, got %v", src)
			}
		})
	}
}

// (Concurrency clients are no longer a lazy fallback case as of #1459 — the
// former TestGenerateWorkloadLazy_FallsBackOnConcurrencyClient was replaced by
// the positive parity table TestGenerateWorkloadLazy_ConcurrencyClient_MatchesEager.)

// TestGenerateWorkloadLazy_EmptyHorizon_ImmediatelyExhausted pins R20: a
// zero/negative horizon must return an empty, immediately-exhausted source
// rather than nil.
func TestGenerateWorkloadLazy_EmptyHorizon_ImmediatelyExhausted(t *testing.T) {
	spec := singleClientChatbotSpec(1)
	src, sessions, budget, err := GenerateWorkloadLazy(spec, 0, 10)
	if err != nil {
		t.Fatalf("zero-horizon: unexpected err %v", err)
	}
	if src == nil {
		t.Fatal("expected non-nil source even for zero horizon")
	}
	if r, ok := src.Next(); ok || r != nil {
		t.Fatalf("zero-horizon Next() = (%v, %v); want (nil, false)", r, ok)
	}
	if len(sessions) != 0 {
		t.Errorf("zero-horizon should return 0 sessions, got %d", len(sessions))
	}
	if budget != -1 {
		t.Errorf("zero-horizon followUpBudget = %d; want -1", budget)
	}
}

// TestLazyRequestSource_NegativeMaxRequests_ReturnsError pins R3-style
// validation: negative maxRequests is rejected with an error rather than
// silently treated as "unlimited".
func TestLazyRequestSource_NegativeMaxRequests_ReturnsError(t *testing.T) {
	spec := singleClientChatbotSpec(1)
	_, _, _, err := GenerateWorkloadLazy(spec, 1_000_000, -1)
	if err == nil {
		t.Fatal("expected error for negative maxRequests")
	}
}

// TestLazyRequestSource_SameSeed_TwoRunsIdentical pins BC-5 (INV-6
// determinism): running the lazy source twice with the same seed
// produces byte-identical request sequences.
func TestLazyRequestSource_SameSeed_TwoRunsIdentical(t *testing.T) {
	spec1 := singleClientChatbotSpec(2026)
	spec2 := singleClientChatbotSpec(2026)
	a, _, _, err := GenerateWorkloadLazy(spec1, 5_000_000, 25)
	if err != nil {
		t.Fatalf("lazy a: %v", err)
	}
	b, _, _, err := GenerateWorkloadLazy(spec2, 5_000_000, 25)
	if err != nil {
		t.Fatalf("lazy b: %v", err)
	}
	first := drainLazy(t, a)
	second := drainLazy(t, b)
	assertRequestStreamsEqual(t, first, second)
}

// TestLazyRequestSource_PrefixGroup_MatchesEager pins BC-3 across a
// non-trivial code path: shared prefix tokens prepended to each
// request's InputTokens. The prefix slice contents must be identical
// in both modes (drawn from workloadRNG in identical order).
func TestLazyRequestSource_PrefixGroup_MatchesEager(t *testing.T) {
	mk := func() *WorkloadSpec {
		return &WorkloadSpec{
			Version: "1", Seed: 314, Category: "language", AggregateRate: 5.0,
			Clients: []ClientSpec{{
				ID: "c1", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
				PrefixGroup:  "g1",
				PrefixLength: 200,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 10, "max": 200}},
				OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 30}},
			}},
		}
	}
	eager, err := GenerateRequests(mk(), 5_000_000, 20)
	if err != nil {
		t.Fatalf("eager: %v", err)
	}
	src, _, _, err := GenerateWorkloadLazy(mk(), 5_000_000, 20)
	if err != nil {
		t.Fatalf("lazy: %v", err)
	}
	lazy := drainLazy(t, src)
	assertRequestStreamsEqual(t, eager, lazy)
	if len(lazy) == 0 || lazy[0].PrefixLength != 200 {
		t.Fatalf("expected PrefixLength=200, got %d", lazy[0].PrefixLength)
	}
}

// TestLazyRequestSource_Reasoning_SingleSession_MatchesEager pins BC-3 for
// SingleSession=true reasoning (the inference-perf shape and #1441's
// reproducer). The streaming source must produce the same session IDs in
// the same order (only round-0 emitted) and consume RNG draws identically.
func TestLazyRequestSource_Reasoning_SingleSession_MatchesEager(t *testing.T) {
	mk := func() *WorkloadSpec {
		return reasoningSingleSessionSpec(2027)
	}
	// GenerateWorkload is the eager closed-loop baseline (filters round-0).
	wlEager, err := GenerateWorkload(mk(), 3_000_000, 20)
	if err != nil {
		t.Fatalf("eager GenerateWorkload: %v", err)
	}
	src, lazySessions, _, err := GenerateWorkloadLazy(mk(), 3_000_000, 20)
	if err != nil {
		t.Fatalf("lazy: %v", err)
	}
	lazyReqs := drainLazy(t, src)
	assertRequestStreamsEqual(t, wlEager.Requests, lazyReqs)

	// Blueprints must match in count, order, and SessionID.
	if len(wlEager.Sessions) != len(lazySessions) {
		t.Fatalf("session count: eager=%d lazy=%d", len(wlEager.Sessions), len(lazySessions))
	}
	for i := range wlEager.Sessions {
		if wlEager.Sessions[i].SessionID != lazySessions[i].SessionID {
			t.Fatalf("session %d: SessionID eager=%q lazy=%q",
				i, wlEager.Sessions[i].SessionID, lazySessions[i].SessionID)
		}
		if wlEager.Sessions[i].ClientID != lazySessions[i].ClientID {
			t.Fatalf("session %d: ClientID eager=%q lazy=%q",
				i, wlEager.Sessions[i].ClientID, lazySessions[i].ClientID)
		}
	}
}

// TestLazyRequestSource_SamplerError_SurfacedViaErr pins the fix for
// PR #1453 review round 3 (IMPORTANT): a sampler failure inside the
// streaming source MUST be surfaced via lazyRequestSource.Err() so
// cmd/root.go can Fatalf after cluster.Run — matching eager mode's
// abort-on-invalid-spec behavior. Regressions that swallow the error
// (or return nil after logging) would silently reduce traffic and
// yield misleading capacity numbers.
//
// spec.Validate() does NOT validate MultimodalSpec's sub-distribution
// types, so an unknown TextDist type passes construction but fails
// deterministically inside GenerateMultimodalTokens. Drain the source
// once; assert Err() returns a wrapped error that names the client
// and includes the "unknown distribution type" phrase.
func TestLazyRequestSource_SamplerError_SurfacedViaErr(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 5.0,
		Clients: []ClientSpec{{
			ID: "mm-bad", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 10, "max": 200}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 30}},
			Multimodal: &MultimodalSpec{
				// TextDist.Type is not a valid distribution; NewLengthSampler
				// will error. spec.Validate() does NOT catch this because
				// Multimodal sub-fields are not validated in advance.
				TextDist: DistSpec{Type: "not-a-real-dist", Params: map[string]float64{"mean": 10}},
			},
		}},
	}
	src, _, _, err := GenerateWorkloadLazy(spec, 5_000_000, 5)
	if err != nil {
		t.Fatalf("construction: %v", err)
	}
	// Drain — first Next() should trigger GenerateMultimodalTokens and
	// record the error on the state, which will then exhaust.
	for {
		if _, ok := src.Next(); !ok {
			break
		}
	}
	got := src.Err()
	if got == nil {
		t.Fatal("Err() returned nil after sampler failure; must surface the error so cmd can Fatalf")
	}
	// Sanity-check the error mentions the offending client + the underlying cause.
	msg := got.Error()
	if !strings.Contains(msg, "mm-bad") {
		t.Errorf("Err() = %q; expected mention of client ID 'mm-bad'", msg)
	}
	if !strings.Contains(msg, "unknown distribution type") {
		t.Errorf("Err() = %q; expected underlying 'unknown distribution type' wrapped in message", msg)
	}
}

// TestLazyRequestSource_Multimodal_MatchesEager pins BC-3 for the
// multimodal branch of produceNextSingleShot: TextTokenCount,
// ImageTokenCount, AudioTokenCount, and VideoTokenCount must be
// populated identically to the eager generator (which calls
// GenerateMultimodalTokens with the same RNG state).
func TestLazyRequestSource_Multimodal_MatchesEager(t *testing.T) {
	mk := func() *WorkloadSpec {
		return &WorkloadSpec{
			Version: "1", Seed: 99, Category: "language", AggregateRate: 5.0,
			Clients: []ClientSpec{{
				ID: "mm1", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 10, "max": 200}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 30}},
				Multimodal: &MultimodalSpec{
					TextDist:       DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 40, "std_dev": 5, "min": 10, "max": 100}},
					ImageDist:      DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 20, "std_dev": 4, "min": 5, "max": 50}},
					ImageCountDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 2, "std_dev": 1, "min": 1, "max": 3}},
					AudioDist:      DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 15, "std_dev": 3, "min": 5, "max": 40}},
					AudioCountDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 1, "std_dev": 1, "min": 0, "max": 2}},
					VideoDist:      DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 25, "std_dev": 5, "min": 10, "max": 50}},
					VideoCountDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 0, "std_dev": 1, "min": 0, "max": 1}},
				},
			}},
		}
	}
	eager, err := GenerateRequests(mk(), 5_000_000, 12)
	if err != nil {
		t.Fatalf("eager: %v", err)
	}
	src, _, _, err := GenerateWorkloadLazy(mk(), 5_000_000, 12)
	if err != nil {
		t.Fatalf("lazy: %v", err)
	}
	lazy := drainLazy(t, src)
	// At least one request must have a non-zero multimodal token count for
	// the assertion-helper to actually exercise those fields.
	var hasMM bool
	for _, r := range lazy {
		if r.TextTokenCount+r.ImageTokenCount+r.AudioTokenCount+r.VideoTokenCount > 0 {
			hasMM = true
			break
		}
	}
	if !hasMM {
		t.Fatal("multimodal spec produced zero counts across all token kinds — test would not exercise the new comparisons")
	}
	assertRequestStreamsEqual(t, eager, lazy)
}

// TestLazyRequestSource_PoppedNotYielded_StopsAtPoppedCap pins the
// stopping rule of lazyRequestSource.Next: the cap applies to `popped`
// (total heap pops, including suppressed intermediate closed-loop
// rounds), not to `yielded`. Regressions that change the stop condition
// to `l.yielded >= l.maxRequests` would yield extra round-0 requests
// past the eager truncation point and break parity.
//
// Construction: 2 closed-loop SingleSession=true reasoning clients.
// Each client's one session has rounds 0, 1, 2 — all popped, only
// round-0 yielded. With maxRequests = 4, eager truncates to first 4 in
// arrival order (a mix of round-0 and intermediate rounds). Lazy must
// stop at popped = 4 and yield strictly fewer than 4 requests to the
// cluster.
func TestLazyRequestSource_PoppedNotYielded_StopsAtPoppedCap(t *testing.T) {
	mk := func() *WorkloadSpec {
		mkClient := func(id string) ClientSpec {
			return ClientSpec{
				ID: id, TenantID: id + "-t", SLOClass: "batch",
				RateFraction: 0.5,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 60, "std_dev": 10, "min": 10, "max": 200}},
				OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 30}},
				Reasoning: &ReasoningSpec{
					MultiTurn: &MultiTurnSpec{
						MaxRounds:     3,
						ContextGrowth: "accumulate",
						ThinkTimeUs:   100_000,
						SingleSession: true, // need lazy-supported variant
					},
				},
			}
		}
		return &WorkloadSpec{
			Version: "1", Seed: 2027, Category: "language", AggregateRate: 4.0,
			Clients: []ClientSpec{mkClient("r1"), mkClient("r2")},
		}
	}
	const tightCap = int64(4)
	wlEager, err := GenerateWorkload(mk(), 3_000_000, tightCap)
	if err != nil {
		t.Fatalf("eager: %v", err)
	}
	src, _, _, err := GenerateWorkloadLazy(mk(), 3_000_000, tightCap)
	if err != nil {
		t.Fatalf("lazy: %v", err)
	}
	lazyReqs := drainLazy(t, src)
	// Lazy and eager must yield the same set of round-0 requests to the
	// cluster, including the SAME truncation effect (fewer than maxRequests
	// if intermediate rounds consumed some of the cap).
	assertRequestStreamsEqual(t, wlEager.Requests, lazyReqs)
	// Sanity: the test must actually exercise the popped > yielded case.
	// If lazy yields >= maxRequests round-0s, the cap was applied at the
	// wrong layer.
	if int64(len(lazyReqs)) >= tightCap {
		t.Fatalf("yielded %d requests under cap=%d — expected fewer (intermediate rounds should consume part of the cap)", len(lazyReqs), tightCap)
	}
}

// TestLazyRequestSource_Reasoning_TightMaxRequests_BlueprintParity pins
// the fix for PR #1453 self-review (IMPORTANT issue #1): when
// maxRequests is small enough to drop a later session's round-0, the
// blueprint pre-pass must produce the SAME set of SessionBlueprints
// as the eager path — same count, same SessionIDs, same order, same
// per-session RNG seeds.
//
// Eager constructs blueprints by scanning the truncated request slice.
// Lazy's blueprint pre-pass must mirror that exactly, including the
// truncation effect. A pre-pass that enumerates all sessions across
// the horizon would consume extra blueprintRNG.Int63() draws,
// shifting all subsequent blueprint RNG seeds and breaking byte-
// identity (INV-6) + run/replay parity (INV-13).
//
// Construction: 8 single-session reasoning clients × MaxRounds=2 → 16
// round-emissions total. maxRequests=10 → eager truncates to first 10
// in arrival order. Most clients' round-0 survives but some don't.
// Lazy must match the surviving subset exactly.
func TestLazyRequestSource_Reasoning_TightMaxRequests_BlueprintParity(t *testing.T) {
	mk := func() *WorkloadSpec {
		mkClient := func(id string) ClientSpec {
			return ClientSpec{
				ID: id, TenantID: id + "-t", SLOClass: "batch",
				RateFraction: 0.125, // 8 clients × 0.125 = 1.0
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 60, "std_dev": 10, "min": 10, "max": 200}},
				OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 30}},
				Reasoning: &ReasoningSpec{
					MultiTurn: &MultiTurnSpec{
						MaxRounds:     2,
						ContextGrowth: "accumulate",
						ThinkTimeUs:   100_000,
						SingleSession: true,
					},
				},
			}
		}
		return &WorkloadSpec{
			Version: "1", Seed: 2031, Category: "language", AggregateRate: 4.0,
			Clients: []ClientSpec{
				mkClient("r0"), mkClient("r1"), mkClient("r2"), mkClient("r3"),
				mkClient("r4"), mkClient("r5"), mkClient("r6"), mkClient("r7"),
			},
		}
	}
	const tightCap = int64(10)
	wlEager, err := GenerateWorkload(mk(), 5_000_000, tightCap)
	if err != nil {
		t.Fatalf("eager: %v", err)
	}
	src, lazySessions, _, err := GenerateWorkloadLazy(mk(), 5_000_000, tightCap)
	if err != nil {
		t.Fatalf("lazy: %v", err)
	}
	lazyReqs := drainLazy(t, src)
	assertRequestStreamsEqual(t, wlEager.Requests, lazyReqs)

	// The point of this test: blueprints must match in count AND order
	// AND RNG-seed-derived state. If lazy enumerated all sessions across
	// the horizon (ignoring the cap), it would have MORE blueprints than
	// eager — and each surviving blueprint's RNG seed would be shifted.
	if len(wlEager.Sessions) != len(lazySessions) {
		t.Fatalf("session count under tight cap: eager=%d lazy=%d (the pre-pass must respect maxRequests)",
			len(wlEager.Sessions), len(lazySessions))
	}
	for i := range wlEager.Sessions {
		if wlEager.Sessions[i].SessionID != lazySessions[i].SessionID {
			t.Fatalf("blueprint %d: SessionID eager=%q lazy=%q",
				i, wlEager.Sessions[i].SessionID, lazySessions[i].SessionID)
		}
		if wlEager.Sessions[i].ClientID != lazySessions[i].ClientID {
			t.Fatalf("blueprint %d: ClientID eager=%q lazy=%q",
				i, wlEager.Sessions[i].ClientID, lazySessions[i].ClientID)
		}
		// RNG byte-identity: draw one Int63 from each and compare. If
		// blueprint RNG seeds matched, these MUST match too.
		eDraw := wlEager.Sessions[i].RNG.Int63()
		lDraw := lazySessions[i].RNG.Int63()
		if eDraw != lDraw {
			t.Fatalf("blueprint %d (session %q): RNG draw diverged eager=%d lazy=%d — blueprintRNG seeds shifted",
				i, wlEager.Sessions[i].SessionID, eDraw, lDraw)
		}
	}
	// Sanity: confirm at least one client lost its session to the cap.
	// Otherwise the test would pass even without the fix.
	if len(wlEager.Sessions) >= 8 {
		t.Fatalf("test setup ineffective: expected fewer than 8 sessions to survive cap=%d, got %d — adjust the setup so the cap actually binds",
			tightCap, len(wlEager.Sessions))
	}
}

// TestLazyRequestSource_Reasoning_MultiSession_MatchesEager pins BC-1 for
// SingleSession=false reasoning (issue #1458): a traffic source that spawns
// many independent, overlapping sessions over the horizon. The lazy source
// runs a per-client internal merge over its live sessions so its emitted
// sub-stream is arrival-monotonic (BC-2), yet byte-identical to the eager
// generator's globally-sorted stream. Covers closed-loop (default: only
// round-0 emitted + blueprints) here; the open-loop variant is below.
func TestLazyRequestSource_Reasoning_MultiSession_MatchesEager(t *testing.T) {
	// A single multi-session client with a modest rate over a long horizon
	// produces many overlapping sessions (each MaxRounds rounds). This is the
	// exact shape that the old dead per-client cursor got wrong.
	mk := func() *WorkloadSpec {
		return reasoningMultiSessionSpec(2027)
	}
	wlEager, err := GenerateWorkload(mk(), 3_000_000, 40)
	if err != nil {
		t.Fatalf("eager GenerateWorkload: %v", err)
	}
	src, lazySessions, _, err := GenerateWorkloadLazy(mk(), 3_000_000, 40)
	if err != nil {
		t.Fatalf("lazy GenerateWorkloadLazy: %v", err)
	}
	lazyReqs := drainLazy(t, src)

	// Sanity: the spec must actually produce multiple overlapping sessions,
	// else the test degenerates to the single-session case and proves nothing.
	sessionIDs := make(map[string]bool)
	for _, r := range lazyReqs {
		if r.SessionID != "" {
			sessionIDs[r.SessionID] = true
		}
	}
	if len(sessionIDs) < 2 {
		t.Fatalf("multi-session test ineffective: expected >= 2 distinct sessions, got %d", len(sessionIDs))
	}

	assertRequestStreamsEqual(t, wlEager.Requests, lazyReqs)

	// Blueprints (closed-loop default) must match count, order, SessionID,
	// ClientID, and first RNG draw.
	if len(wlEager.Sessions) != len(lazySessions) {
		t.Fatalf("session count: eager=%d lazy=%d", len(wlEager.Sessions), len(lazySessions))
	}
	for i := range wlEager.Sessions {
		if wlEager.Sessions[i].SessionID != lazySessions[i].SessionID {
			t.Fatalf("session %d: SessionID eager=%q lazy=%q",
				i, wlEager.Sessions[i].SessionID, lazySessions[i].SessionID)
		}
		if wlEager.Sessions[i].ClientID != lazySessions[i].ClientID {
			t.Fatalf("session %d: ClientID eager=%q lazy=%q",
				i, wlEager.Sessions[i].ClientID, lazySessions[i].ClientID)
		}
		if e, l := wlEager.Sessions[i].RNG.Int63(), lazySessions[i].RNG.Int63(); e != l {
			t.Fatalf("session %d (%q): RNG draw diverged eager=%d lazy=%d",
				i, wlEager.Sessions[i].SessionID, e, l)
		}
	}
}

// TestLazyRequestSource_Reasoning_MultiSession_OpenLoop_MatchesEager pins
// BC-1 for open-loop (closed_loop=false) multi-session reasoning: ALL rounds
// of every overlapping session are emitted (no round-0 filter, no blueprints),
// so the interleaving across sessions is exercised most directly.
func TestLazyRequestSource_Reasoning_MultiSession_OpenLoop_MatchesEager(t *testing.T) {
	openLoop := false
	mk := func() *WorkloadSpec {
		s := reasoningMultiSessionSpec(3001)
		s.Clients[0].ClosedLoop = &openLoop
		return s
	}
	eager, err := GenerateRequests(mk(), 3_000_000, 40)
	if err != nil {
		t.Fatalf("eager GenerateRequests: %v", err)
	}
	src, _, _, err := GenerateWorkloadLazy(mk(), 3_000_000, 40)
	if err != nil {
		t.Fatalf("lazy GenerateWorkloadLazy: %v", err)
	}
	lazyReqs := drainLazy(t, src)

	// Sanity: open-loop must emit non-round-0 rounds (otherwise the
	// interleaving path is unexercised).
	var sawIntermediate bool
	for _, r := range lazyReqs {
		if r.RoundIndex > 0 {
			sawIntermediate = true
			break
		}
	}
	if !sawIntermediate {
		t.Fatal("open-loop multi-session test ineffective: no round > 0 emitted")
	}
	assertRequestStreamsEqual(t, eager, lazyReqs)
}

// TestLazyRequestSource_MultiSession_PerClientMonotonic pins BC-2: a single
// multi-session client feeds the global heap ONE entry at a time, and the
// heap-merge is only correct if that per-client sub-stream is non-decreasing
// in arrival time. Overlapping sessions make this non-trivial (session N+1's
// round 0 can precede session N's later rounds). Drain a single-client spec
// and assert monotonicity directly.
func TestLazyRequestSource_MultiSession_PerClientMonotonic(t *testing.T) {
	openLoop := false // open-loop so all rounds (not just round-0) are emitted
	spec := reasoningMultiSessionSpec(4099)
	spec.Clients[0].ClosedLoop = &openLoop
	src, _, _, err := GenerateWorkloadLazy(spec, 3_000_000, 60)
	if err != nil {
		t.Fatalf("lazy: %v", err)
	}
	reqs := drainLazy(t, src)
	if len(reqs) < 4 {
		t.Fatalf("expected several requests to exercise ordering, got %d", len(reqs))
	}
	var prev int64 = -1
	for i, r := range reqs {
		if r.ArrivalTime < prev {
			t.Fatalf("request %d (%s round %d): ArrivalTime %d < previous %d — per-client stream not arrival-monotonic",
				i, r.SessionID, r.RoundIndex, r.ArrivalTime, prev)
		}
		prev = r.ArrivalTime
	}
}

// TestLazyRequestSource_MultiSession_MultiClientTightCap_MatchesEager pins
// BC-1/BC-4 for the hardest multi-session shape: MULTIPLE overlapping
// multi-session clients (explicit + cohort-expanded) under a tight maxRequests
// cap. This stresses (a) cross-client interleaving through the global heap,
// (b) the per-client build cap (perClientCap parity), and (c) closed-loop
// round-0 blueprint enumeration under truncation — all at once.
func TestLazyRequestSource_MultiSession_MultiClientTightCap_MatchesEager(t *testing.T) {
	mkClient := func(id string, frac float64) ClientSpec {
		return ClientSpec{
			ID: id, TenantID: id + "-t", SLOClass: "batch",
			RateFraction: frac,
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 40, "std_dev": 8, "min": 10, "max": 120}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 20}},
			PrefixGroup:  "shared",
			PrefixLength: 48,
			Reasoning: &ReasoningSpec{
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     4,
					ContextGrowth: "accumulate",
					ThinkTimeUs:   80_000,
					// SingleSession: false (multi-session)
				},
			},
		}
	}
	mk := func() *WorkloadSpec {
		return &WorkloadSpec{
			Version: "1", Seed: 5150, Category: "language", AggregateRate: 6.0,
			Clients: []ClientSpec{mkClient("a", 0.3), mkClient("b", 0.3)},
			Cohorts: []CohortSpec{{
				ID: "co", Population: 3, TenantID: "co-t", SLOClass: "batch",
				RateFraction: 0.4,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 40, "std_dev": 8, "min": 10, "max": 120}},
				OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 20}},
				PrefixGroup:  "shared",
				PrefixLength: 48,
				Reasoning: &ReasoningSpec{
					MultiTurn: &MultiTurnSpec{
						MaxRounds:     4,
						ContextGrowth: "accumulate",
						ThinkTimeUs:   80_000,
					},
				},
			}},
		}
	}
	const tightCap = int64(12)
	wlEager, err := GenerateWorkload(mk(), 4_000_000, tightCap)
	if err != nil {
		t.Fatalf("eager: %v", err)
	}
	src, lazySessions, _, err := GenerateWorkloadLazy(mk(), 4_000_000, tightCap)
	if err != nil {
		t.Fatalf("lazy: %v", err)
	}
	lazyReqs := drainLazy(t, src)
	assertRequestStreamsEqual(t, wlEager.Requests, lazyReqs)

	// Blueprints must match under truncation (count, order, SessionID, ClientID).
	if len(wlEager.Sessions) != len(lazySessions) {
		t.Fatalf("session count under tight cap: eager=%d lazy=%d", len(wlEager.Sessions), len(lazySessions))
	}
	for i := range wlEager.Sessions {
		if wlEager.Sessions[i].SessionID != lazySessions[i].SessionID {
			t.Fatalf("blueprint %d: SessionID eager=%q lazy=%q", i, wlEager.Sessions[i].SessionID, lazySessions[i].SessionID)
		}
		if wlEager.Sessions[i].ClientID != lazySessions[i].ClientID {
			t.Fatalf("blueprint %d: ClientID eager=%q lazy=%q", i, wlEager.Sessions[i].ClientID, lazySessions[i].ClientID)
		}
	}
	// Sanity: multiple clients' sessions must survive so cross-client
	// interleaving is actually exercised.
	clientIDs := make(map[string]bool)
	for _, r := range lazyReqs {
		clientIDs[r.ClientID] = true
	}
	if len(clientIDs) < 2 {
		t.Fatalf("multi-client test ineffective: expected >= 2 distinct client IDs in surviving stream, got %d", len(clientIDs))
	}
}

// TestLazyRequestSource_MultiSession_Lifecycle_MatchesEager pins BC-1/BC-4 for
// multi-session reasoning under a PLAIN lifecycle (two active windows with a
// gap, no per-window parameter overrides — so this stays on the non-time-varying
// path and is lazy-supported). This is the only committed test exercising the
// new lifecycle branches in buildNextSession (skip-past-window, break-past-last-
// window) and the emit-path per-round lifecycle filter for multi-session.
// Open-loop so every round flows through the per-round window filter.
func TestLazyRequestSource_MultiSession_Lifecycle_MatchesEager(t *testing.T) {
	openLoop := false
	mk := func() *WorkloadSpec {
		s := reasoningMultiSessionSpec(6270)
		s.Clients[0].ClosedLoop = &openLoop
		// Two windows with a mid gap: [0,1s) active, [1s,2s) inactive, [2s,4s)
		// active. Sessions starting in the gap are skipped (build branch);
		// rounds that drift into the gap are suppressed (emit-path filter).
		s.Clients[0].Lifecycle = &LifecycleSpec{
			Windows: []ActiveWindow{
				{StartUs: 0, EndUs: 1_000_000},
				{StartUs: 2_000_000, EndUs: 4_000_000},
			},
		}
		return s
	}
	// Horizon beyond the last window so the break-past-last-window branch fires.
	eager, err := GenerateRequests(mk(), 5_000_000, 80)
	if err != nil {
		t.Fatalf("eager GenerateRequests: %v", err)
	}
	src, _, _, err := GenerateWorkloadLazy(mk(), 5_000_000, 80)
	if err != nil {
		t.Fatalf("lazy GenerateWorkloadLazy: %v", err)
	}
	lazyReqs := drainLazy(t, src)
	assertRequestStreamsEqual(t, eager, lazyReqs)

	// Sanity: the lifecycle must actually bind — every emitted round must fall
	// inside an active window, and the stream must be non-empty.
	if len(lazyReqs) == 0 {
		t.Fatal("lifecycle multi-session test ineffective: empty stream")
	}
	inWindow := func(t int64) bool {
		return (t >= 0 && t < 1_000_000) || (t >= 2_000_000 && t < 4_000_000)
	}
	for i, r := range lazyReqs {
		if !inWindow(r.ArrivalTime) {
			t.Fatalf("request %d (%s round %d): ArrivalTime %d outside all active windows",
				i, r.SessionID, r.RoundIndex, r.ArrivalTime)
		}
	}
}

// TestExpandClientsAndCohorts_Idempotent_InferencePerf pins the contract
// that lets cmd/root.go safely call ExpandClientsAndCohorts before the
// generator runs (so applyTimeoutToSpec sees the expanded clients for
// inference-perf specs under --lazy-generation, fixing the bug flagged
// in PR #1453 self-review) — and have the generator's internal
// validateAndExpandSpec call NOT re-expand. A second call MUST be a
// no-op: spec.Clients must not be re-expanded, AggregateRate must not
// be recomputed.
func TestExpandClientsAndCohorts_Idempotent_InferencePerf(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2", Seed: 11, Category: "language",
		InferencePerf: &InferencePerfSpec{
			Stages: []StageSpec{{Rate: 4.0, Duration: 5}},
			SharedPrefix: &SharedPrefixSpec{
				NumUniqueSystemPrompts:  2,
				NumUsersPerSystemPrompt: 2,
				SystemPromptLen:         16,
				QuestionLen:             32,
				OutputLen:               16,
			},
		},
	}
	if err := ExpandClientsAndCohorts(spec); err != nil {
		t.Fatalf("first call: %v", err)
	}
	firstLen := len(spec.Clients)
	firstAgg := spec.AggregateRate
	if firstLen == 0 {
		t.Fatal("first call did not expand inference-perf into clients")
	}
	if err := ExpandClientsAndCohorts(spec); err != nil {
		t.Fatalf("second call: %v", err)
	}
	if got := len(spec.Clients); got != firstLen {
		t.Errorf("second call re-expanded clients: %d → %d (must be idempotent)", firstLen, got)
	}
	if got := spec.AggregateRate; got != firstAgg {
		t.Errorf("second call mutated AggregateRate: %v → %v (must be idempotent)", firstAgg, got)
	}
}

// TestGenerateWorkloadLazy_InferencePerf_TimeoutAppliedAfterPreExpand
// pins the cmd/root.go fix for the PR-review regression: when an
// inference-perf spec is expanded by ExpandClientsAndCohorts BEFORE
// applyTimeoutToSpec runs, the Timeout pointer is set on every
// expanded client, so produceNextSingleShot's computeDeadline call
// returns ArrivalTime + the user-requested timeout — not the
// 300 s session-client default that would arise from a nil Timeout.
//
// We can't reach applyTimeoutToSpec from sim/workload (cyclic), so we
// emulate it: pre-expand, then set every client's Timeout pointer to
// a known non-default value, then build the lazy source and assert
// the first request's Deadline matches our value.
func TestGenerateWorkloadLazy_InferencePerf_TimeoutAppliedAfterPreExpand(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2", Seed: 11, Category: "language",
		InferencePerf: &InferencePerfSpec{
			Stages: []StageSpec{{Rate: 4.0, Duration: 5}},
			SharedPrefix: &SharedPrefixSpec{
				NumUniqueSystemPrompts:  2,
				NumUsersPerSystemPrompt: 2,
				SystemPromptLen:         16,
				QuestionLen:             32,
				OutputLen:               16,
			},
		},
	}
	// Step 1: expand inference-perf so spec.Clients is populated.
	if err := ExpandClientsAndCohorts(spec); err != nil {
		t.Fatalf("expand: %v", err)
	}
	if len(spec.Clients) == 0 {
		t.Fatal("inference-perf expansion produced 0 clients")
	}
	// Step 2: apply a non-default timeout (120 s = 120_000_000 µs) to every
	// expanded client — mirroring what cmd/root.go's applyTimeoutToSpec does.
	const wantTimeoutUs = int64(120_000_000)
	t120 := wantTimeoutUs
	for i := range spec.Clients {
		spec.Clients[i].Timeout = &t120
	}
	// Step 3: build the lazy source (single-session inference-perf is supported).
	src, _, _, err := GenerateWorkloadLazy(spec, 10_000_000, 1)
	if err != nil {
		t.Fatalf("lazy: %v", err)
	}
	r, ok := src.Next()
	if !ok || r == nil {
		t.Fatal("expected at least one request from inference-perf spec")
	}
	// The Deadline should be ArrivalTime + 120 s, NOT ArrivalTime + 300 s
	// (the DefaultTimeoutUs fallback that would fire if Timeout were nil).
	wantDeadline := r.ArrivalTime + wantTimeoutUs
	if r.Deadline != wantDeadline {
		t.Fatalf("Deadline=%d, want %d (arrival=%d + timeout=%d). "+
			"A 300_000_000 timeout-µs value would indicate the inference-perf "+
			"clients were built with nil Timeout — the bug fixed in PR #1453.",
			r.Deadline, wantDeadline, r.ArrivalTime, wantTimeoutUs)
	}
}

// assertBlueprintsEqual compares two SessionBlueprint slices field-by-field for
// the fields the concurrency path controls, then compares one RNG draw from each
// (blueprint RNGs seed the follow-up round stream; a shifted seed diverges here).
// The RNG draw is destructive but both blueprints are discarded after.
func assertBlueprintsEqual(t *testing.T, eager, lazy []SessionBlueprint) {
	t.Helper()
	if len(eager) != len(lazy) {
		t.Fatalf("blueprint count: eager=%d lazy=%d", len(eager), len(lazy))
	}
	for i := range eager {
		e, l := &eager[i], &lazy[i]
		if e.SessionID != l.SessionID {
			t.Fatalf("blueprint %d: SessionID eager=%q lazy=%q", i, e.SessionID, l.SessionID)
		}
		if e.ClientID != l.ClientID {
			t.Fatalf("blueprint %d: ClientID eager=%q lazy=%q", i, e.ClientID, l.ClientID)
		}
		if e.UnlimitedRounds != l.UnlimitedRounds {
			t.Fatalf("blueprint %d (%s): UnlimitedRounds eager=%v lazy=%v", i, e.SessionID, e.UnlimitedRounds, l.UnlimitedRounds)
		}
		if e.ThinkTimeUs != l.ThinkTimeUs {
			t.Fatalf("blueprint %d (%s): ThinkTimeUs eager=%d lazy=%d", i, e.SessionID, e.ThinkTimeUs, l.ThinkTimeUs)
		}
		if e.SLOClass != l.SLOClass {
			t.Fatalf("blueprint %d (%s): SLOClass eager=%q lazy=%q", i, e.SessionID, e.SLOClass, l.SLOClass)
		}
		if e.Model != l.Model {
			t.Fatalf("blueprint %d (%s): Model eager=%q lazy=%q", i, e.SessionID, e.Model, l.Model)
		}
		if e.RNG == nil || l.RNG == nil {
			t.Fatalf("blueprint %d (%s): nil RNG eager=%v lazy=%v", i, e.SessionID, e.RNG == nil, l.RNG == nil)
		}
		if ed, ld := e.RNG.Int63(), l.RNG.Int63(); ed != ld {
			t.Fatalf("blueprint %d (%s): RNG draw diverged eager=%d lazy=%d — bpSeed shifted",
				i, e.SessionID, ed, ld)
		}
	}
}

// TestGenerateWorkloadLazy_ConcurrencyClient_MatchesEager pins BC-1/BC-2/BC-3/
// BC-4 for concurrency clients (#1459, AC2): the lazy streaming source must
// produce byte-identical request streams + session blueprints + FollowUpBudget
// to the eager GenerateWorkload across every concurrency shape. Before #1459
// every sub-case returned ErrLazyUnsupportedConcurrency (RED).
//
// The five sub-cases are chosen to exercise the load-bearing correctness edges
// the plan review surfaced:
//   - mixed: one open-loop + one concurrency client (the common case).
//   - multi-concurrency: TWO concurrency clients with DISTINCT Concurrency and
//     ThinkTimeUs so their staggered seed arrivals interleave — the case a naive
//     single-FIFO seed emitter mis-orders (CRITICAL-1). This is the regression
//     test for the individual-seed-heap-entry design.
//   - all-concurrency: zero open-loop clients (keptOpen==0, empty preps).
//   - binding-cap: open-loop + seeds exceed maxRequests; asserts eager cap
//     semantics + FollowUpBudget==0 (BC-4).
//   - horizon-zero: horizon==0 with a concurrency client — eager still emits
//     seeds (horizon-independent), so lazy must too (CRITICAL-2).
func TestGenerateWorkloadLazy_ConcurrencyClient_MatchesEager(t *testing.T) {
	openLoop := func(id string, frac float64) ClientSpec {
		return ClientSpec{
			ID: id, TenantID: id + "-t", SLOClass: "batch",
			RateFraction: frac,
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 60, "std_dev": 10, "min": 8, "max": 300}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 40}},
		}
	}
	conc := func(id string, users int, thinkUs int64) ClientSpec {
		return ClientSpec{
			ID: id, TenantID: id + "-t", SLOClass: "batch",
			Concurrency: users,
			ThinkTimeUs: thinkUs,
			InputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 8, "min": 8, "max": 200}},
			OutputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 30}},
		}
	}

	cases := []struct {
		name    string
		clients []ClientSpec
		aggRate float64 // AggregateRate; 0 for all-concurrency specs
		horizon int64
		maxReqs int64
	}{
		{
			// Rate 4/s over 3s → ~10 open-loop kept, well under the cap, so the
			// 5 concurrency seeds also survive (no-displacement invariant holds).
			name:    "mixed",
			clients: []ClientSpec{openLoop("ol", 1.0), conc("pool", 5, 200_000)},
			aggRate: 4.0, horizon: 3_000_000, maxReqs: 60,
		},
		{
			// Two concurrency pools with distinct stagger → interleaved seed
			// arrivals. A single-FIFO seed emitter would mis-order these.
			name:    "multi-concurrency",
			clients: []ClientSpec{conc("poolA", 3, 300_000), conc("poolB", 2, 100_000)},
			aggRate: 0, horizon: 30_000_000, maxReqs: 40,
		},
		{
			name:    "all-concurrency",
			clients: []ClientSpec{conc("only", 6, 150_000)},
			aggRate: 0, horizon: 30_000_000, maxReqs: 40,
		},
		{
			// Concurrency users (20) exceed the cap (10): eager generates only 10
			// seeds and FollowUpBudget==0 (seeds consume the whole budget). Pins
			// the seed cap parity (BC-4) — lazy must stop at the same 10 seeds.
			name:    "binding-cap",
			clients: []ClientSpec{conc("pool", 20, 120_000)},
			aggRate: 0, horizon: 30_000_000, maxReqs: 10,
		},
		{
			// horizon==0: eager's concurrency seed loop is horizon-independent.
			name:    "horizon-zero",
			clients: []ClientSpec{conc("pool", 4, 100_000)},
			aggRate: 0, horizon: 0, maxReqs: 10,
		},
		{
			// Combined binding cap: open-loop is PARTIALLY kept AND seeds are
			// PARTIALLY truncated in the same run (0 < alreadyKept(10) <
			// maxRequests(15) < alreadyKept+users(20) → 10 open-loop + 5 seeds).
			// Exercises the no-displacement invariant at the boundary where
			// keptOpen>0 feeds the seed cap — the case the all-concurrency
			// binding-cap sub-case (keptOpen==0) does not reach.
			name:    "combined-cap",
			clients: []ClientSpec{openLoop("ol", 1.0), conc("pool", 10, 120_000)},
			aggRate: 4.0, horizon: 3_000_000, maxReqs: 15,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			mk := func() *WorkloadSpec {
				return &WorkloadSpec{
					Version: "2", Seed: 4242, Category: "language",
					AggregateRate: tc.aggRate,
					Clients:       append([]ClientSpec{}, tc.clients...),
				}
			}
			wlEager, err := GenerateWorkload(mk(), tc.horizon, tc.maxReqs)
			if err != nil {
				t.Fatalf("eager GenerateWorkload: %v", err)
			}
			src, lazySessions, lazyBudget, err := GenerateWorkloadLazy(mk(), tc.horizon, tc.maxReqs)
			if err != nil {
				t.Fatalf("lazy GenerateWorkloadLazy: %v (want nil — concurrency is supported as of #1459)", err)
			}
			lazyReqs := drainLazy(t, src)

			assertRequestStreamsEqual(t, wlEager.Requests, lazyReqs)
			assertBlueprintsEqual(t, wlEager.Sessions, lazySessions)
			if wlEager.FollowUpBudget != lazyBudget {
				t.Fatalf("FollowUpBudget: eager=%d lazy=%d", wlEager.FollowUpBudget, lazyBudget)
			}

			// Guard against a vacuous pass: every sub-case must emit at least one
			// concurrency seed (SessionID prefixed "concurrency_").
			sawSeed := false
			for _, r := range lazyReqs {
				if strings.HasPrefix(r.SessionID, "concurrency_") {
					sawSeed = true
					break
				}
			}
			if !sawSeed {
				t.Fatalf("no concurrency seed emitted in sub-case %q — test is vacuous", tc.name)
			}
			// binding-cap: seeds should consume the remaining budget → FollowUpBudget==0.
			if tc.name == "binding-cap" && lazyBudget != 0 {
				t.Fatalf("binding-cap: FollowUpBudget=%d, want 0 (seeds consume remainder)", lazyBudget)
			}
		})
	}
}
