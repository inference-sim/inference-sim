package workload

import (
	"errors"
	"fmt"
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

// reasoningMultiSessionSpec is for the fallback test (SingleSession=false
// is NOT in this PR's lazy scope — caller falls back to GenerateWorkload).
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

// TestGenerateWorkloadLazy_FallsBackOnPerWindowParameters pins BC-8's
// time-varying fallback signal: when any client has a per-window
// parameter override, the factory returns ErrLazyUnsupportedTimeVarying
// (caller falls back to GenerateWorkload).
func TestGenerateWorkloadLazy_FallsBackOnPerWindowParameters(t *testing.T) {
	traceRate := 2.0
	spec := &WorkloadSpec{
		Version: "1", Seed: 1, Category: "language", AggregateRate: 4.0,
		Clients: []ClientSpec{{
			ID: "c1", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 10, "max": 500}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
			Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{
				{StartUs: 0, EndUs: 5_000_000, TraceRate: &traceRate},
			}},
		}},
	}
	_, _, _, err := GenerateWorkloadLazy(spec, 10_000_000, 10)
	if !errors.Is(err, ErrLazyUnsupportedTimeVarying) {
		t.Fatalf("want ErrLazyUnsupportedTimeVarying, got %v", err)
	}
}

// TestGenerateWorkloadLazy_FallsBackOnConcurrencyClient pins BC-8's
// concurrency fallback: any client with Concurrency > 0 forces the
// caller to fall back to GenerateWorkload.
func TestGenerateWorkloadLazy_FallsBackOnConcurrencyClient(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "1", Seed: 1, Category: "language",
		Clients: []ClientSpec{{
			ID: "c1", TenantID: "t1", SLOClass: "batch",
			Concurrency: 4,
			ThinkTimeUs: 100_000,
			InputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 10, "max": 500}},
			OutputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	_, _, _, err := GenerateWorkloadLazy(spec, 10_000_000, 10)
	if !errors.Is(err, ErrLazyUnsupportedConcurrency) {
		t.Fatalf("want ErrLazyUnsupportedConcurrency, got %v", err)
	}
}

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

// TestGenerateWorkloadLazy_FallsBackOnMultiSession pins the
// SingleSession=false fallback: any reasoning client without
// SingleSession=true forces the caller back to GenerateWorkload.
func TestGenerateWorkloadLazy_FallsBackOnMultiSession(t *testing.T) {
	_, _, _, err := GenerateWorkloadLazy(reasoningMultiSessionSpec(7), 5_000_000, 20)
	if !errors.Is(err, ErrLazyUnsupportedMultiSession) {
		t.Fatalf("want ErrLazyUnsupportedMultiSession, got %v", err)
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
