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

func TestExpandInferencePerfSpec_TwoStages_LifecycleWindows(t *testing.T) {
	// BC-1: stage-to-lifecycle expansion
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
	if len(ws.Clients) != 1 {
		t.Fatalf("client count = %d, want 1", len(ws.Clients))
	}
	lc := ws.Clients[0].Lifecycle
	if lc == nil {
		t.Fatal("lifecycle should be set for multi-stage spec")
	}
	if len(lc.Windows) != 2 {
		t.Fatalf("window count = %d, want 2", len(lc.Windows))
	}
	// Window 1: [0, 600_000_000)
	if lc.Windows[0].StartUs != 0 {
		t.Errorf("window[0].StartUs = %d, want 0", lc.Windows[0].StartUs)
	}
	if lc.Windows[0].EndUs != 600_000_000 {
		t.Errorf("window[0].EndUs = %d, want 600000000", lc.Windows[0].EndUs)
	}
	// Window 2: [600_000_000, 1_200_000_000)
	if lc.Windows[1].StartUs != 600_000_000 {
		t.Errorf("window[1].StartUs = %d, want 600000000", lc.Windows[1].StartUs)
	}
	if lc.Windows[1].EndUs != 1_200_000_000 {
		t.Errorf("window[1].EndUs = %d, want 1200000000", lc.Windows[1].EndUs)
	}
}

func TestExpandInferencePerfSpec_TwoStages_AggregateRate(t *testing.T) {
	// BC-2: aggregate rate is time-weighted average
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
	// Time-weighted average: (8*600 + 20*600) / 1200 = 14.0
	expectedRate := 14.0
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

func TestExpandInferencePerfSpec_ThreeStages_CumulativeWindows(t *testing.T) {
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
	lc := ws.Clients[0].Lifecycle
	if lc == nil || len(lc.Windows) != 3 {
		t.Fatalf("expected 3 lifecycle windows")
	}
	// Window 1: [0, 100_000_000)
	if lc.Windows[0].EndUs != 100_000_000 {
		t.Errorf("window[0].EndUs = %d, want 100000000", lc.Windows[0].EndUs)
	}
	// Window 2: [100_000_000, 300_000_000)
	if lc.Windows[1].StartUs != 100_000_000 || lc.Windows[1].EndUs != 300_000_000 {
		t.Errorf("window[1] = [%d, %d), want [100000000, 300000000)", lc.Windows[1].StartUs, lc.Windows[1].EndUs)
	}
	// Window 3: [300_000_000, 600_000_000)
	if lc.Windows[2].StartUs != 300_000_000 || lc.Windows[2].EndUs != 600_000_000 {
		t.Errorf("window[2] = [%d, %d), want [300000000, 600000000)", lc.Windows[2].StartUs, lc.Windows[2].EndUs)
	}
}

// --- Multi-turn mapping tests (Task 5) ---

func TestExpandInferencePerfSpec_MultiTurn_MapsToReasoning(t *testing.T) {
	// BC-7: multi-turn flag maps to reasoning spec
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
		if c.Reasoning == nil {
			t.Fatalf("client %q: Reasoning should be set when multi-turn enabled", c.ID)
		}
		mt := c.Reasoning.MultiTurn
		if mt == nil {
			t.Fatalf("client %q: MultiTurn should be set", c.ID)
		}
		if mt.MaxRounds != 5 {
			t.Errorf("client %q: MaxRounds = %d, want 5", c.ID, mt.MaxRounds)
		}
		if mt.ContextGrowth != "accumulate" {
			t.Errorf("client %q: ContextGrowth = %q, want accumulate", c.ID, mt.ContextGrowth)
		}
		if mt.ThinkTimeUs != 500000 {
			t.Errorf("client %q: ThinkTimeUs = %d, want 500000", c.ID, mt.ThinkTimeUs)
		}
	}
}

func TestExpandInferencePerfSpec_NoMultiTurn_NoReasoning(t *testing.T) {
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
	for _, c := range ws.Clients {
		if c.Reasoning != nil {
			t.Errorf("client %q: Reasoning should be nil when multi-turn disabled", c.ID)
		}
	}
}

func TestExpandInferencePerfSpec_MultiTurn_CategoryIsReasoning(t *testing.T) {
	// When multi-turn is enabled, category should be "reasoning"
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
			EnableMultiTurnChat:     true,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ws.Category != "reasoning" {
		t.Errorf("category = %q, want reasoning when multi-turn enabled", ws.Category)
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

// suppress unused imports
var _ = sim.StateQueued
var _ = filepath.Join
var _ = os.WriteFile
