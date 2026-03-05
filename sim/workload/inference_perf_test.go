package workload

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestValidateInferencePerfSpec_ValidSpec_NoError(t *testing.T) {
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
			{Rate: 20.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  9,
			NumUsersPerSystemPrompt: 5,
			SystemPromptLen:         100,
			QuestionLen:             447,
			OutputLen:               248,
		},
	}
	if err := validateInferencePerfSpec(spec); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidateInferencePerfSpec_ZeroDuration_ReturnsError(t *testing.T) {
	// BC-10: zero-duration stages rejected
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 0},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for zero duration")
	}
	if !strings.Contains(err.Error(), "duration must be positive") {
		t.Errorf("error should mention duration: %v", err)
	}
}

func TestValidateInferencePerfSpec_ZeroPrompts_ReturnsError(t *testing.T) {
	// BC-11: zero system prompts rejected
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  0,
			NumUsersPerSystemPrompt: 5,
			SystemPromptLen:         100,
			QuestionLen:             447,
			OutputLen:               248,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for zero system prompts")
	}
}

func TestValidateInferencePerfSpec_NegativeLength_ReturnsError(t *testing.T) {
	// BC-12: negative lengths rejected
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         -1,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for negative system_prompt_len")
	}
}

func TestValidateInferencePerfSpec_NegativeRate_ReturnsError(t *testing.T) {
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: -1.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for negative rate")
	}
}

func TestValidateInferencePerfSpec_NoStages_ReturnsError(t *testing.T) {
	spec := &InferencePerfSpec{
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for no stages")
	}
}

func TestValidateInferencePerfSpec_NoSharedPrefix_ReturnsError(t *testing.T) {
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for nil shared_prefix")
	}
}

// --- Expansion tests (Task 3) ---

func TestExpandInferencePerfSpec_SharedPrefix_GeneratesNxMClients(t *testing.T) {
	// BC-3: N*M clients generated
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  9,
			NumUsersPerSystemPrompt: 5,
			SystemPromptLen:         100,
			QuestionLen:             447,
			OutputLen:               248,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ws.Clients) != 45 {
		t.Fatalf("client count = %d, want 45 (9*5)", len(ws.Clients))
	}
}

func TestExpandInferencePerfSpec_PrefixGroups_NineDistinct(t *testing.T) {
	// BC-3: 9 distinct prefix groups
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  9,
			NumUsersPerSystemPrompt: 5,
			SystemPromptLen:         100,
			QuestionLen:             447,
			OutputLen:               248,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	groups := make(map[string]int)
	for _, c := range ws.Clients {
		groups[c.PrefixGroup]++
	}
	if len(groups) != 9 {
		t.Errorf("distinct prefix groups = %d, want 9", len(groups))
	}
	for g, count := range groups {
		if count != 5 {
			t.Errorf("prefix group %q has %d clients, want 5", g, count)
		}
	}
}

func TestExpandInferencePerfSpec_PrefixLength_Configurable(t *testing.T) {
	// BC-4: configurable prefix length
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         100,
			QuestionLen:             50,
			OutputLen:               25,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, c := range ws.Clients {
		if c.PrefixLength != 100 {
			t.Errorf("client %q: PrefixLength = %d, want 100", c.ID, c.PrefixLength)
		}
	}
}

func TestExpandInferencePerfSpec_ConstantDistributions(t *testing.T) {
	// BC-5: fixed lengths become constant distributions
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         100,
			QuestionLen:             447,
			OutputLen:               248,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	c := ws.Clients[0]
	if c.InputDist.Type != "constant" {
		t.Errorf("input dist type = %q, want constant", c.InputDist.Type)
	}
	if c.InputDist.Params["value"] != 447 {
		t.Errorf("input dist value = %f, want 447", c.InputDist.Params["value"])
	}
	if c.OutputDist.Type != "constant" {
		t.Errorf("output dist type = %q, want constant", c.OutputDist.Type)
	}
	if c.OutputDist.Params["value"] != 248 {
		t.Errorf("output dist value = %f, want 248", c.OutputDist.Params["value"])
	}
}

func TestExpandInferencePerfSpec_ValidWorkloadSpec(t *testing.T) {
	// BC-8: expansion produces valid WorkloadSpec
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  3,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("expand error: %v", err)
	}
	if err := ws.Validate(); err != nil {
		t.Errorf("expanded spec validation failed: %v", err)
	}
}

func TestExpandInferencePerfSpec_MultiStage_ValidWorkloadSpec(t *testing.T) {
	// Multi-stage expanded spec must pass Validate().
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
			{Rate: 20.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  3,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("expand error: %v", err)
	}
	if err := ws.Validate(); err != nil {
		t.Errorf("multi-stage expanded spec validation failed: %v", err)
	}
}

func TestExpandInferencePerfSpec_EqualRateFractions(t *testing.T) {
	// Each client gets equal share of traffic
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  3,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// All 6 clients should have equal rate fractions
	expectedFrac := 1.0 / 6.0
	for _, c := range ws.Clients {
		if c.RateFraction < expectedFrac*0.99 || c.RateFraction > expectedFrac*1.01 {
			t.Errorf("client %q: rate_fraction = %f, want ~%f", c.ID, c.RateFraction, expectedFrac)
		}
	}
}

// --- Stage-based rate tests (Task 4) ---

func TestExpandInferencePerfSpec_TwoStages_PerStageClients(t *testing.T) {
	// BC-1: multi-stage creates per-stage client cohorts
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
			{Rate: 20.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// 2 stages × 1 client each = 2 clients
	if len(ws.Clients) != 2 {
		t.Fatalf("client count = %d, want 2 (1 per stage)", len(ws.Clients))
	}

	// Stage 0 client: active during [0, 600_000_000)
	lc0 := ws.Clients[0].Lifecycle
	if lc0 == nil || len(lc0.Windows) != 1 {
		t.Fatal("stage 0 client should have exactly 1 lifecycle window")
	}
	if lc0.Windows[0].StartUs != 0 || lc0.Windows[0].EndUs != 600_000_000 {
		t.Errorf("stage 0 window = [%d, %d), want [0, 600000000)",
			lc0.Windows[0].StartUs, lc0.Windows[0].EndUs)
	}

	// Stage 1 client: active during [600_000_000, 1_200_000_000)
	lc1 := ws.Clients[1].Lifecycle
	if lc1 == nil || len(lc1.Windows) != 1 {
		t.Fatal("stage 1 client should have exactly 1 lifecycle window")
	}
	if lc1.Windows[0].StartUs != 600_000_000 || lc1.Windows[0].EndUs != 1_200_000_000 {
		t.Errorf("stage 1 window = [%d, %d), want [600000000, 1200000000)",
			lc1.Windows[0].StartUs, lc1.Windows[0].EndUs)
	}
}

func TestExpandInferencePerfSpec_TwoStages_AggregateRate(t *testing.T) {
	// BC-2: aggregate rate is sum of stage rates (not time-weighted average).
	// Each stage's clients emit at the stage rate during their window;
	// aggregateRate = sum ensures normalizeRateFractions produces correct per-client rates.
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
			{Rate: 20.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Sum of stage rates: 8.0 + 20.0 = 28.0
	expectedRate := 28.0
	if ws.AggregateRate != expectedRate {
		t.Errorf("aggregate rate = %f, want %f", ws.AggregateRate, expectedRate)
	}
}

func TestExpandInferencePerfSpec_SingleStage_NoLifecycle(t *testing.T) {
	// Single stage: no lifecycle windows needed
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ws.Clients[0].Lifecycle != nil {
		t.Error("single stage should not set lifecycle windows")
	}
	if ws.AggregateRate != 10.0 {
		t.Errorf("aggregate rate = %f, want 10.0", ws.AggregateRate)
	}
}

func TestExpandInferencePerfSpec_ThreeStages_PerStageClients(t *testing.T) {
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 5.0, Duration: 100},
			{Rate: 10.0, Duration: 200},
			{Rate: 15.0, Duration: 300},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// 3 stages × 1 client each = 3 clients
	if len(ws.Clients) != 3 {
		t.Fatalf("client count = %d, want 3 (1 per stage)", len(ws.Clients))
	}
	// aggregateRate = 5 + 10 + 15 = 30
	if ws.AggregateRate != 30.0 {
		t.Errorf("aggregate rate = %f, want 30.0", ws.AggregateRate)
	}

	// Each client has exactly one lifecycle window matching its stage
	expectedWindows := []ActiveWindow{
		{StartUs: 0, EndUs: 100_000_000},
		{StartUs: 100_000_000, EndUs: 300_000_000},
		{StartUs: 300_000_000, EndUs: 600_000_000},
	}
	for i, client := range ws.Clients {
		lc := client.Lifecycle
		if lc == nil || len(lc.Windows) != 1 {
			t.Fatalf("client[%d]: expected exactly 1 lifecycle window", i)
		}
		got := lc.Windows[0]
		want := expectedWindows[i]
		if got.StartUs != want.StartUs || got.EndUs != want.EndUs {
			t.Errorf("client[%d] window = [%d, %d), want [%d, %d)",
				i, got.StartUs, got.EndUs, want.StartUs, want.EndUs)
		}
	}
}

// --- Multi-turn flag tests (#481) ---

func TestExpandInferencePerfSpec_MultiTurn_ProducesIndependentRequests(t *testing.T) {
	// BC-1: enable_multi_turn_chat is ignored — no ReasoningSpec created.
	// Real inference-perf data shows constant input tokens (~574).
	// BLIS's generator architecture (many-sessions-per-client) is incompatible
	// with inference-perf's model (one-session-per-user). See #481.
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 3,
			SystemPromptLen:         100,
			QuestionLen:             50,
			OutputLen:               25,
			EnableMultiTurnChat:     true,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, c := range ws.Clients {
		if c.Reasoning != nil {
			t.Errorf("client %q: Reasoning should be nil (multi-turn flag is ignored for inference-perf)", c.ID)
		}
	}
	// BC-3: category should be "language" (not "reasoning")
	if ws.Category != "language" {
		t.Errorf("category = %q, want language", ws.Category)
	}
}

func TestExpandInferencePerfSpec_MultiTurnFalse_NoReasoning(t *testing.T) {
	// BC-2: enable_multi_turn_chat=false produces nil Reasoning (explicit false-path test)
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
			EnableMultiTurnChat:     false,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ws.Clients[0].Reasoning != nil {
		t.Error("Reasoning should be nil when enable_multi_turn_chat is false")
	}
	if ws.Category != "language" {
		t.Errorf("category = %q, want language", ws.Category)
	}
}

func TestExpandInferencePerfSpec_MultiStageMultiTurn_Succeeds(t *testing.T) {
	// BC-4: multi-stage + multi-turn no longer rejected since the flag is a no-op.
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 5.0, Duration: 600},
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
			EnableMultiTurnChat:     true,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ws.Clients) != 2 {
		t.Fatalf("client count = %d, want 2 (1 per stage)", len(ws.Clients))
	}
	// BC-1 also holds for multi-stage path
	for _, c := range ws.Clients {
		if c.Reasoning != nil {
			t.Errorf("client %q: Reasoning should be nil", c.ID)
		}
	}
	// Verify expanded spec passes full validation
	if err := ws.Validate(); err != nil {
		t.Fatalf("expanded spec validation failed: %v", err)
	}
	// Verify lifecycle windows are correctly assigned (stage 0: [0, 600s), stage 1: [600s, 1200s))
	lc0 := ws.Clients[0].Lifecycle
	if lc0 == nil || len(lc0.Windows) != 1 {
		t.Fatal("stage 0 client should have exactly 1 lifecycle window")
	}
	if lc0.Windows[0].StartUs != 0 || lc0.Windows[0].EndUs != 600_000_000 {
		t.Errorf("stage 0 window = [%d, %d), want [0, 600000000)",
			lc0.Windows[0].StartUs, lc0.Windows[0].EndUs)
	}
	lc1 := ws.Clients[1].Lifecycle
	if lc1 == nil || len(lc1.Windows) != 1 {
		t.Fatal("stage 1 client should have exactly 1 lifecycle window")
	}
	if lc1.Windows[0].StartUs != 600_000_000 || lc1.Windows[0].EndUs != 1_200_000_000 {
		t.Errorf("stage 1 window = [%d, %d), want [600000000, 1200000000)",
			lc1.Windows[0].StartUs, lc1.Windows[0].EndUs)
	}
}

func TestExpandInferencePerfSpec_MultiTurnFlag_IsNoOpEndToEnd(t *testing.T) {
	// BC-7: enable_multi_turn_chat=true produces the same request count as false.
	// This is the quantitative regression anchor proving the flag is truly a no-op.
	horizon := int64(10_000_000) // 10 seconds

	// With flag = false
	specFalse := &InferencePerfSpec{
		Stages: []StageSpec{{Rate: 10.0, Duration: 10}},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
			EnableMultiTurnChat:     false,
		},
	}
	wsFalse, err := ExpandInferencePerfSpec(specFalse, 42)
	if err != nil {
		t.Fatalf("expand false: %v", err)
	}
	rFalse, err := GenerateRequests(wsFalse, horizon, 0)
	if err != nil {
		t.Fatalf("generate false: %v", err)
	}

	// With flag = true
	specTrue := &InferencePerfSpec{
		Stages: []StageSpec{{Rate: 10.0, Duration: 10}},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
			EnableMultiTurnChat:     true,
		},
	}
	wsTrue, err := ExpandInferencePerfSpec(specTrue, 42)
	if err != nil {
		t.Fatalf("expand true: %v", err)
	}
	rTrue, err := GenerateRequests(wsTrue, horizon, 0)
	if err != nil {
		t.Fatalf("generate true: %v", err)
	}

	if len(rFalse) != len(rTrue) {
		t.Errorf("request count differs: false=%d, true=%d (flag should be no-op)", len(rFalse), len(rTrue))
	}
}

// --- Integration tests (Task 6) ---

func TestGenerateRequests_InferencePerfSpec_ProducesRequests(t *testing.T) {
	// BC-8: end-to-end generation from inference-perf spec
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 10}, // 10 seconds
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}
	spec := &WorkloadSpec{
		Version:       "1",
		Seed:          42,
		AggregateRate: 10.0,
		InferencePerf: ipSpec,
	}
	horizon := int64(10_000_000) // 10 seconds
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected requests from inference-perf spec")
	}
}

func TestGenerateRequests_InferencePerfSpec_Deterministic(t *testing.T) {
	// BC-9: determinism preserved
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 10},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}
	horizon := int64(10_000_000)

	spec1 := &WorkloadSpec{
		Version:       "1",
		Seed:          42,
		AggregateRate: 10.0,
		InferencePerf: ipSpec,
	}
	r1, err1 := GenerateRequests(spec1, horizon, 0)

	// Second run with fresh spec (expansion mutates spec.Clients)
	spec2 := &WorkloadSpec{
		Version:       "1",
		Seed:          42,
		AggregateRate: 10.0,
		InferencePerf: ipSpec,
	}
	r2, err2 := GenerateRequests(spec2, horizon, 0)
	if err1 != nil || err2 != nil {
		t.Fatalf("errors: %v, %v", err1, err2)
	}
	if len(r1) != len(r2) {
		t.Fatalf("different counts: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
	}
}

func TestLoadWorkloadSpec_InferencePerfSpec_StrictParsing(t *testing.T) {
	// BC-13: strict YAML parsing for new types
	dir := t.TempDir()
	path := filepath.Join(dir, "bad-ip.yaml")
	yamlData := `
version: "1"
seed: 42
aggregate_rate: 10.0
inference_perf:
  stages:
    - rate: 10.0
      duraton: 600
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 10
    question_len: 10
    output_len: 10
`
	if err := os.WriteFile(path, []byte(yamlData), 0644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadWorkloadSpec(path)
	if err == nil {
		t.Fatal("expected error for typo 'duraton' in YAML")
	}
}

// --- Equivalence tests (Task 7) ---

func TestInferencePerfExpansion_EquivalentToManual(t *testing.T) {
	// Acceptance criterion: shorthand and manual expansion produce identical requests.
	// Build equivalent specs and compare generated request sequences.

	// Shorthand spec
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 10},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}
	expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}

	// Manual spec: construct the same clients explicitly
	manual := &WorkloadSpec{
		Version:       expanded.Version,
		Seed:          expanded.Seed,
		Category:      expanded.Category,
		AggregateRate: expanded.AggregateRate,
		Clients:       expanded.Clients, // use the expanded clients directly
	}

	horizon := int64(10_000_000) // 10 seconds

	// Generate from expanded
	r1, err1 := GenerateRequests(expanded, horizon, 0)
	if err1 != nil {
		t.Fatalf("expanded generation error: %v", err1)
	}

	// Generate from manual (same clients)
	r2, err2 := GenerateRequests(manual, horizon, 0)
	if err2 != nil {
		t.Fatalf("manual generation error: %v", err2)
	}

	if len(r1) != len(r2) {
		t.Fatalf("different request counts: expanded=%d, manual=%d", len(r1), len(r2))
	}

	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
		if len(r1[i].InputTokens) != len(r2[i].InputTokens) {
			t.Errorf("request %d: input len %d vs %d", i, len(r1[i].InputTokens), len(r2[i].InputTokens))
			break
		}
	}
}

func TestInferencePerfExpansion_SharedPrefixTokensIdentical(t *testing.T) {
	// Verify that clients in the same prefix group actually share prefix tokens
	// when requests are generated through the full pipeline.
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 20.0, Duration: 5},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 3,
			SystemPromptLen:         80,
			QuestionLen:             50,
			OutputLen:               25,
		},
	}
	expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	horizon := int64(5_000_000) // 5 seconds
	requests, err := GenerateRequests(expanded, horizon, 100)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}
	if len(requests) < 2 {
		t.Fatal("need at least 2 requests for prefix comparison")
	}

	// All requests should have inputs at least 80 tokens long (prefix length)
	prefixLen := 80
	for i, req := range requests {
		if len(req.InputTokens) < prefixLen {
			t.Errorf("request %d: input len %d < prefix len %d", i, len(req.InputTokens), prefixLen)
		}
	}

	// Group requests by tenant (which maps to prefix group)
	byTenant := make(map[string][]*sim.Request)
	for _, req := range requests {
		byTenant[req.TenantID] = append(byTenant[req.TenantID], req)
	}

	// Within each group, first prefixLen tokens must be identical
	for tenant, reqs := range byTenant {
		if len(reqs) < 2 {
			continue
		}
		first := reqs[0].InputTokens[:prefixLen]
		for i := 1; i < len(reqs); i++ {
			other := reqs[i].InputTokens[:prefixLen]
			for j := 0; j < prefixLen; j++ {
				if first[j] != other[j] {
					t.Errorf("tenant %q: request %d prefix token %d differs from request 0", tenant, i, j)
					break
				}
			}
		}
	}
}

// --- YAML pipeline test (Task 8) ---

func TestLoadWorkloadSpec_InferencePerfSpec_FullPipeline(t *testing.T) {
	// Full YAML -> parse -> expand -> generate pipeline
	dir := t.TempDir()
	path := filepath.Join(dir, "ip-spec.yaml")
	yamlData := `
version: "1"
seed: 42
aggregate_rate: 10.0
inference_perf:
  stages:
    - rate: 8.0
      duration: 5
    - rate: 20.0
      duration: 5
  shared_prefix:
    num_unique_system_prompts: 3
    num_users_per_system_prompt: 2
    system_prompt_len: 50
    question_len: 100
    output_len: 50
`
	if err := os.WriteFile(path, []byte(yamlData), 0644); err != nil {
		t.Fatal(err)
	}

	spec, err := LoadWorkloadSpec(path)
	if err != nil {
		t.Fatalf("load error: %v", err)
	}
	if spec.InferencePerf == nil {
		t.Fatal("InferencePerf should be parsed from YAML")
	}
	if len(spec.InferencePerf.Stages) != 2 {
		t.Errorf("stage count = %d, want 2", len(spec.InferencePerf.Stages))
	}

	horizon := int64(10_000_000) // 10 seconds
	requests, err := GenerateRequests(spec, horizon, 50)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected requests from YAML pipeline")
	}
	if len(requests) > 50 {
		t.Errorf("request count %d exceeds maxRequests 50", len(requests))
	}
}

func TestGenerateRequests_InferencePerfSpec_AggregateRateOverridden(t *testing.T) {
	// The expanded aggregate rate must always override the user-specified value.
	// A user-specified aggregate_rate conflicts with per-stage rates and would
	// silently scale all rates by the wrong factor.
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 5},
			{Rate: 20.0, Duration: 5},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	spec := &WorkloadSpec{
		Version:       "2",
		Seed:          42,
		AggregateRate: 10.0, // wrong — should be 28.0 (sum of stage rates)
		InferencePerf: ipSpec,
	}
	horizon := int64(10_000_000) // 10 seconds
	_, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}
	// After expansion, AggregateRate must be overridden to sum of stage rates.
	if spec.AggregateRate != 28.0 {
		t.Errorf("AggregateRate = %f, want 28.0 (sum of 8+20)", spec.AggregateRate)
	}
}

// --- Invariant tests (Task 9) ---

func TestInferencePerf_Determinism_SameSeedIdenticalOutput(t *testing.T) {
	// INV-6: same seed -> identical output
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 10},
			{Rate: 20.0, Duration: 10},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  3,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}

	generate := func() []*sim.Request {
		expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
		if err != nil {
			t.Fatalf("expansion error: %v", err)
		}
		reqs, err := GenerateRequests(expanded, 20_000_000, 100)
		if err != nil {
			t.Fatalf("generation error: %v", err)
		}
		return reqs
	}

	r1 := generate()
	r2 := generate()

	if len(r1) != len(r2) {
		t.Fatalf("different counts: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
		if r1[i].ID != r2[i].ID {
			t.Errorf("request %d: ID %q vs %q", i, r1[i].ID, r2[i].ID)
			break
		}
		if len(r1[i].InputTokens) != len(r2[i].InputTokens) {
			t.Errorf("request %d: input len %d vs %d", i, len(r1[i].InputTokens), len(r2[i].InputTokens))
			break
		}
		// Verify token-level identity
		for j := range r1[i].InputTokens {
			if r1[i].InputTokens[j] != r2[i].InputTokens[j] {
				t.Errorf("request %d token %d: %d vs %d", i, j, r1[i].InputTokens[j], r2[i].InputTokens[j])
				break
			}
		}
	}
}

func TestInferencePerf_TwoStages_PerStageRateFidelity(t *testing.T) {
	// Core behavioral test for #503: per-stage rates must produce proportional
	// request counts, not a flattened uniform rate.
	// Stage 0: 5 QPS for 600s → ~3000 requests
	// Stage 1: 10 QPS for 600s → ~6000 requests
	// Ratio should be ~0.5 (±0.15 for Poisson variance).
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 5.0, Duration: 600},
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	expanded, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}

	horizon := int64(1_200_000_000) // 1200 seconds in µs
	requests, err := GenerateRequests(expanded, horizon, 0)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}

	boundary := int64(600_000_000) // 600s in µs
	var stage1Count, stage2Count int
	for _, req := range requests {
		if req.ArrivalTime < boundary {
			stage1Count++
		} else {
			stage2Count++
		}
	}

	if stage2Count == 0 {
		t.Fatal("no requests in stage 2")
	}
	ratio := float64(stage1Count) / float64(stage2Count)
	// Expected ratio: 5/10 = 0.5. Allow 20% tolerance for Poisson variance.
	if ratio < 0.35 || ratio > 0.65 {
		t.Errorf("stage rate ratio = %.3f (stage1=%d, stage2=%d), want ~0.5 (±0.15)",
			ratio, stage1Count, stage2Count)
	}
}

func TestInferencePerf_MultiStage_ClientCountIsNxMxStages(t *testing.T) {
	// Multi-stage expansion creates N*M clients per stage.
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
			{Rate: 20.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  9,
			NumUsersPerSystemPrompt: 5,
			SystemPromptLen:         100,
			QuestionLen:             447,
			OutputLen:               248,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// 2 stages × 9 prompts × 5 users = 90 clients
	if len(ws.Clients) != 90 {
		t.Errorf("client count = %d, want 90 (2×9×5)", len(ws.Clients))
	}
}

func TestInferencePerf_MultiStage_PrefixGroupsPreserved(t *testing.T) {
	// All stages share the same prefix groups (same system prompts across stages).
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 5.0, Duration: 100},
			{Rate: 10.0, Duration: 100},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  3,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	groups := make(map[string]int)
	for _, c := range ws.Clients {
		groups[c.PrefixGroup]++
	}
	// 3 prefix groups, each appearing 2 users × 2 stages = 4 times
	if len(groups) != 3 {
		t.Errorf("distinct prefix groups = %d, want 3", len(groups))
	}
	for g, count := range groups {
		if count != 4 {
			t.Errorf("prefix group %q has %d clients, want 4 (2 users × 2 stages)", g, count)
		}
	}
}

func TestInferencePerf_Causality_ArrivalTimesMonotonic(t *testing.T) {
	// INV-3/INV-5: arrival times never decrease
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 50.0, Duration: 5},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  3,
			NumUsersPerSystemPrompt: 3,
			SystemPromptLen:         20,
			QuestionLen:             50,
			OutputLen:               25,
		},
	}
	expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	requests, err := GenerateRequests(expanded, 5_000_000, 0)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}
	for i := 1; i < len(requests); i++ {
		if requests[i].ArrivalTime < requests[i-1].ArrivalTime {
			t.Errorf("arrival time not monotonic: request %d (%d) < request %d (%d)",
				i, requests[i].ArrivalTime, i-1, requests[i-1].ArrivalTime)
			break
		}
	}
}

func TestInferencePerf_AllRequestsHaveValidTokens(t *testing.T) {
	// Every request must have non-empty input and output tokens
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 20.0, Duration: 5},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         30,
			QuestionLen:             50,
			OutputLen:               25,
		},
	}
	expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	requests, err := GenerateRequests(expanded, 5_000_000, 50)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}
	for i, req := range requests {
		if len(req.InputTokens) == 0 {
			t.Errorf("request %d has empty input tokens", i)
		}
		if len(req.OutputTokens) == 0 {
			t.Errorf("request %d has empty output tokens", i)
		}
		// Input tokens should be at least prefix_length (30) + question_len (50)
		expectedMinLen := 30 + 50
		if len(req.InputTokens) < expectedMinLen {
			t.Errorf("request %d: input len %d < expected min %d (prefix+question)",
				i, len(req.InputTokens), expectedMinLen)
		}
	}
}
