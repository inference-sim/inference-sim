package workload

import (
	"fmt"
	"math"
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

func TestValidateInferencePerfSpec_MultiStageMultiTurn_NoError(t *testing.T) {
	// BC-4: multi-stage + multi-turn now works (lifecycle bug fixed in generator.go).
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
	if err := validateInferencePerfSpec(spec); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidateInferencePerfSpec_SingleStageMultiTurn_NoError(t *testing.T) {
	// Single-stage + multi-turn is fine (no lifecycle windows needed).
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
	if err := validateInferencePerfSpec(spec); err != nil {
		t.Errorf("unexpected error: %v", err)
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

// --- Multi-turn mapping tests ---

func TestExpandInferencePerfSpec_MultiTurn_MapsToReasoning(t *testing.T) {
	// enable_multi_turn_chat=true maps to ReasoningSpec with computed parameters.
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
		// rate=10, duration=600, sessions=2*3=6
		// MaxRounds = ceil(10*600/6) = 1000
		// ThinkTimeUs = floor(6/10 * 1e6) = 600_000
		if mt.MaxRounds != 1000 {
			t.Errorf("client %q: MaxRounds = %d, want 1000", c.ID, mt.MaxRounds)
		}
		if mt.ContextGrowth != "" {
			t.Errorf("client %q: ContextGrowth = %q, want empty (fixed-length per H30)", c.ID, mt.ContextGrowth)
		}
		if mt.ThinkTimeUs != 600_000 {
			t.Errorf("client %q: ThinkTimeUs = %d, want 600000", c.ID, mt.ThinkTimeUs)
		}
		if !mt.SingleSession {
			t.Errorf("client %q: SingleSession = false, want true", c.ID)
		}
	}
	if ws.Category != "reasoning" {
		t.Errorf("category = %q, want reasoning when multi-turn enabled", ws.Category)
	}
}

func TestExpandInferencePerfSpec_MultiTurnFalse_NoReasoning(t *testing.T) {
	// enable_multi_turn_chat=false produces nil Reasoning.
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
	// BC-4: multi-stage + multi-turn works with per-stage computed parameters.
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
	for _, c := range ws.Clients {
		if c.Reasoning == nil {
			t.Errorf("client %q: Reasoning should be set", c.ID)
		}
	}
	if err := ws.Validate(); err != nil {
		t.Fatalf("expanded spec validation failed: %v", err)
	}
	// Verify lifecycle windows
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

func TestExpandInferencePerfSpec_MultiTurn_ComputedParameters(t *testing.T) {
	// BC-3: MaxRounds, ThinkTimeUs, SingleSession computed from stage parameters.
	// 1 stage, rate=10, duration=600, 2 prompts × 5 users = 10 sessions
	// MaxRounds = ceil(10*600/10) = 600
	// ThinkTimeUs = floor(10/10 * 1e6) = 1_000_000
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 5,
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
		mt := c.Reasoning.MultiTurn
		if mt.MaxRounds != 600 {
			t.Errorf("client %q: MaxRounds = %d, want 600", c.ID, mt.MaxRounds)
		}
		if mt.ThinkTimeUs != 1_000_000 {
			t.Errorf("client %q: ThinkTimeUs = %d, want 1000000", c.ID, mt.ThinkTimeUs)
		}
		if !mt.SingleSession {
			t.Errorf("client %q: SingleSession = false, want true", c.ID)
		}
		if mt.ContextGrowth != "" {
			t.Errorf("client %q: ContextGrowth = %q, want empty (fixed-length per H30)", c.ID, mt.ContextGrowth)
		}
	}
}

func TestExpandInferencePerfSpec_MultiStageMultiTurn_PerStageParameters(t *testing.T) {
	// BC-4: Each stage gets its own computed MaxRounds and ThinkTimeUs.
	// 2 stages (rate=5/dur=600, rate=20/dur=300), 1 prompt × 1 user = 1 session
	// Stage 0: MaxRounds = ceil(5*600/1) = 3000, ThinkTimeUs = floor(1/5 * 1e6) = 200_000
	// Stage 1: MaxRounds = ceil(20*300/1) = 6000, ThinkTimeUs = floor(1/20 * 1e6) = 50_000
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 5.0, Duration: 600},
			{Rate: 20.0, Duration: 300},
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
	// 2 stages × 1 client each = 2 clients
	if len(ws.Clients) != 2 {
		t.Fatalf("client count = %d, want 2", len(ws.Clients))
	}

	// Stage 0 client
	mt0 := ws.Clients[0].Reasoning.MultiTurn
	if mt0.MaxRounds != 3000 {
		t.Errorf("stage 0: MaxRounds = %d, want 3000", mt0.MaxRounds)
	}
	if mt0.ThinkTimeUs != 200_000 {
		t.Errorf("stage 0: ThinkTimeUs = %d, want 200000", mt0.ThinkTimeUs)
	}
	if !mt0.SingleSession {
		t.Errorf("stage 0: SingleSession = false, want true")
	}

	// Stage 1 client
	mt1 := ws.Clients[1].Reasoning.MultiTurn
	if mt1.MaxRounds != 6000 {
		t.Errorf("stage 1: MaxRounds = %d, want 6000", mt1.MaxRounds)
	}
	if mt1.ThinkTimeUs != 50_000 {
		t.Errorf("stage 1: ThinkTimeUs = %d, want 50000", mt1.ThinkTimeUs)
	}
	if !mt1.SingleSession {
		t.Errorf("stage 1: SingleSession = false, want true")
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
	// Acceptance criterion: two expansions with same seed produce identical results.
	// CustomSamplerFactory ensures fresh sampler instances for each GenerateRequests call.

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
	horizon := int64(10_000_000) // 10 seconds

	// First expansion and generation
	expanded1, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	r1, err1 := GenerateRequests(expanded1, horizon, 0)
	if err1 != nil {
		t.Fatalf("generation error: %v", err1)
	}

	// Second expansion with same seed (fresh samplers)
	expanded2, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	r2, err2 := GenerateRequests(expanded2, horizon, 0)
	if err2 != nil {
		t.Fatalf("generation error: %v", err2)
	}

	// Verify identical output
	if len(r1) != len(r2) {
		t.Fatalf("different request counts: %d vs %d", len(r1), len(r2))
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

func TestGenerateRequests_InferencePerfSpec_MultiTurnMultiStage_Integration(t *testing.T) {
	// End-to-end integration: 2-stage multi-turn inference-perf spec generates
	// requests in both stage windows with approximately 2x ratio.
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 5.0, Duration: 10},  // stage 0: 5 QPS for 10s
			{Rate: 10.0, Duration: 10}, // stage 1: 10 QPS for 10s
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
	expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expand error: %v", err)
	}

	horizon := int64(20_000_000) // 20 seconds in µs
	requests, err := GenerateRequests(expanded, horizon, 0)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected requests from multi-stage multi-turn spec")
	}

	// Count requests per stage window
	boundary := int64(10_000_000) // 10s in µs
	var stage0Count, stage1Count int
	for _, req := range requests {
		if req.ArrivalTime < boundary {
			stage0Count++
		} else {
			stage1Count++
		}
	}

	if stage0Count == 0 {
		t.Error("no requests in stage 0 window")
	}
	if stage1Count == 0 {
		t.Error("no requests in stage 1 window")
	}
	// Stage 1 has 2x rate, so should have roughly 2x requests.
	// Tolerance accounts for SingleSession staggering and Poisson start time.
	if stage1Count > 0 && stage0Count > 0 {
		ratio := float64(stage0Count) / float64(stage1Count)
		if ratio < 0.3 || ratio > 0.7 {
			t.Errorf("stage ratio = %.3f (s0=%d, s1=%d), want ~0.5 (±0.2)",
				ratio, stage0Count, stage1Count)
		}
	}
}

// --- NormalizedExponentialSampler integration tests (Task 3) ---

func TestExpandInferencePerfSpec_SingleStage_NormalizedExponential(t *testing.T) {
	// BC-2: Single-stage expansion uses normalized exponential and produces
	// exactly ceil(rate*duration/numClients) requests per client, summing to
	// the stage duration.
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 60}, // 10 req/s for 60s = 600 total requests
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  3,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	// 3*2 = 6 clients, each should get ceil(600/6) = 100 requests
	// Horizon = 2× duration: ensures sampler exhaustion (not horizon) stops generation.
	horizon := int64(120_000_000) // 120 seconds in µs (2× stage duration)
	requests, err := GenerateRequests(expanded, horizon, 0)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}
	// Total requests should be exactly 600
	if len(requests) != 600 {
		t.Errorf("request count = %d, want 600", len(requests))
	}
	// Verify all arrivals are within the stage duration (not the inflated horizon)
	stageDurationUs := int64(60_000_000)
	for i, req := range requests {
		if req.ArrivalTime < 0 || req.ArrivalTime >= stageDurationUs {
			t.Errorf("request %d: arrival time %d µs outside [0, %d) µs",
				i, req.ArrivalTime, stageDurationUs)
		}
	}
}

func TestExpandInferencePerfSpec_SingleStage_ExactCountWithLargeHorizon(t *testing.T) {
	// Verify that sampler (not horizon) limits request count.
	// With horizon >> durationUs, all N intervals should be consumed.
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 60}, // 10 req/s for 60s = 600 total requests
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  3,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}

	// Set horizon to 2x the stage duration to ensure sampler exhaustion, not horizon, limits requests
	horizon := int64(120_000_000) // 120 seconds (2x the 60s stage duration)
	requests, err := GenerateRequests(expanded, horizon, 0)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}

	// Should get exactly 600 requests (sampler-limited, not horizon-limited)
	if len(requests) != 600 {
		t.Errorf("request count = %d, want 600 (sampler should limit, not horizon)", len(requests))
	}

	// All requests should arrive well before the 120s horizon
	for i, req := range requests {
		if req.ArrivalTime >= int64(60_000_000) {
			t.Errorf("request %d: arrival %d µs >= stage duration 60s; sampler should limit to stage duration",
				i, req.ArrivalTime)
		}
	}
}

func TestExpandInferencePerfSpec_SingleStage_NormalizedDeterministic(t *testing.T) {
	// BC-3: Same seed produces byte-identical expansion.
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 10},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	horizon := int64(10_000_000) // 10 seconds in µs

	// First run
	expanded1, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	r1, err := GenerateRequests(expanded1, horizon, 0)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}

	// Second run with same seed
	expanded2, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	r2, err := GenerateRequests(expanded2, horizon, 0)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}

	// Verify identical output
	if len(r1) != len(r2) {
		t.Fatalf("different request counts: %d vs %d", len(r1), len(r2))
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
	}
}

func TestExpandInferencePerfSpec_MultiStage_KeepsPoisson(t *testing.T) {
	// BC-4: Multi-stage expansion still uses Poisson arrival (for now).
	// Verify that clients have Arrival field set (not CustomSamplerFactory).
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 10},
			{Rate: 20.0, Duration: 10},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	// 2 stages × 1 client each = 2 clients
	if len(expanded.Clients) != 2 {
		t.Fatalf("client count = %d, want 2", len(expanded.Clients))
	}
	for i, client := range expanded.Clients {
		if client.Arrival.Process != "poisson" {
			t.Errorf("client %d: arrival process = %q, want poisson", i, client.Arrival.Process)
		}
		if client.CustomSamplerFactory != nil {
			t.Errorf("client %d: CustomSamplerFactory should be nil (using Poisson)", i)
		}
	}
}

func TestExpandInferencePerfSpec_SingleStage_ConservationInvariant(t *testing.T) {
	// Invariant test: sum(per_client_requests) == total expected requests
	// Tests the conservation law that requestsPerClient allocation sums correctly.
	cases := []struct {
		rate       float64
		duration   int64
		numPrompts int
		numUsers   int
	}{
		{rate: 10.0, duration: 60, numPrompts: 3, numUsers: 2},  // 600 total / 6 clients = 100 each
		{rate: 10.0, duration: 61, numPrompts: 2, numUsers: 2},  // 610 total / 4 clients = ceil(152.5)=153 each
		{rate: 5.0, duration: 100, numPrompts: 1, numUsers: 7},  // 500 total / 7 clients
		{rate: 100.0, duration: 10, numPrompts: 10, numUsers: 3}, // 1000 total / 30 clients
	}

	for _, tc := range cases {
		t.Run(fmt.Sprintf("rate=%.0f_dur=%d_clients=%dx%d", tc.rate, tc.duration, tc.numPrompts, tc.numUsers), func(t *testing.T) {
			ipSpec := &InferencePerfSpec{
				Stages: []StageSpec{
					{Rate: tc.rate, Duration: tc.duration},
				},
				SharedPrefix: &SharedPrefixSpec{
					NumUniqueSystemPrompts:  tc.numPrompts,
					NumUsersPerSystemPrompt: tc.numUsers,
					SystemPromptLen:         10,
					QuestionLen:             10,
					OutputLen:               10,
				},
			}
			expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
			if err != nil {
				t.Fatalf("expansion error: %v", err)
			}

			// Horizon = 2× duration: ensures sampler exhaustion (not horizon) stops generation.
			// With horizon == duration, sampler and horizon guard race, creating test fragility.
			horizon := tc.duration * 2_000_000
			requests, err := GenerateRequests(expanded, horizon, 0)
			if err != nil {
				t.Fatalf("generation error: %v", err)
			}

			// Count requests per client ID
			perClientCounts := make(map[string]int)
			for _, req := range requests {
				perClientCounts[req.ClientID]++
			}

			// Expected: each client gets ceil(rate * duration / numClients) requests
			numClients := tc.numPrompts * tc.numUsers
			expectedPerClient := int(math.Ceil(tc.rate * float64(tc.duration) / float64(numClients)))

			// Verify each client got exactly expectedPerClient requests
			for clientID, count := range perClientCounts {
				if count != expectedPerClient {
					t.Errorf("client %s: got %d requests, want %d", clientID, count, expectedPerClient)
				}
			}

			// Verify sum equals numClients * expectedPerClient (conservation)
			expectedTotal := numClients * expectedPerClient
			if len(requests) != expectedTotal {
				t.Errorf("total requests = %d, want %d (conservation: %d clients * %d each)",
					len(requests), expectedTotal, numClients, expectedPerClient)
			}
		})
	}
}

// TestExpandInferencePerfSpec_WorkloadReusability verifies that factory pattern
// enables calling GenerateRequests multiple times on the same WorkloadSpec.
// This is the primary architectural validation for CustomSamplerFactory.
func TestExpandInferencePerfSpec_WorkloadReusability(t *testing.T) {
	// GIVEN a WorkloadSpec with NormalizedExponentialSampler (via factory)
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 30}, // 300 requests total
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 3, // 6 clients × 50 requests/client = 300 total
			SystemPromptLen:         10,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}
	expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}

	horizon := int64(60_000_000) // 60s (2× duration)

	// WHEN GenerateRequests is called twice with the same WorkloadSpec
	requests1, err := GenerateRequests(expanded, horizon, 0)
	if err != nil {
		t.Fatalf("first generation error: %v", err)
	}

	requests2, err := GenerateRequests(expanded, horizon, 0)
	if err != nil {
		t.Fatalf("second generation error: %v", err)
	}

	// THEN both runs produce identical request counts
	if len(requests1) != len(requests2) {
		t.Fatalf("request count mismatch: run1=%d, run2=%d", len(requests1), len(requests2))
	}

	// AND identical per-client distributions
	perClient1 := make(map[string]int)
	perClient2 := make(map[string]int)
	for _, req := range requests1 {
		perClient1[req.ClientID]++
	}
	for _, req := range requests2 {
		perClient2[req.ClientID]++
	}

	if len(perClient1) != len(perClient2) {
		t.Fatalf("client count mismatch: run1=%d clients, run2=%d clients",
			len(perClient1), len(perClient2))
	}

	for clientID, count1 := range perClient1 {
		count2, ok := perClient2[clientID]
		if !ok {
			t.Errorf("client %s missing in run2", clientID)
			continue
		}
		if count1 != count2 {
			t.Errorf("client %s: run1=%d requests, run2=%d requests", clientID, count1, count2)
		}
	}

	// AND identical arrival times (deterministic from seed)
	for i := range requests1 {
		if requests1[i].ArrivalTime != requests2[i].ArrivalTime {
			t.Errorf("request %d: arrival time mismatch: run1=%d, run2=%d",
				i, requests1[i].ArrivalTime, requests2[i].ArrivalTime)
			if i >= 5 {
				t.Logf("... (stopping after 5 mismatches)")
				break
			}
		}
	}
}

// TestInferencePerfClients_SLOClass_IsStandard asserts BC-1: no client from
// ExpandInferencePerfSpec uses SLOClass "batch" or "background".
// Regression guard for issue #965 (commit 8bc7a48c deferred-queue interaction).
func TestInferencePerfClients_SLOClass_IsStandard(t *testing.T) {
	spec := &InferencePerfSpec{
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 3,
			SystemPromptLen:         100,
			QuestionLen:             200,
			OutputLen:               50,
			EnableMultiTurnChat:     false,
		},
		Stages: []StageSpec{
			{Rate: 5.0, Duration: 60},
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("ExpandInferencePerfSpec: %v", err)
	}
	if len(ws.Clients) == 0 {
		t.Fatal("expected at least one client")
	}
	for _, c := range ws.Clients {
		if c.SLOClass == "batch" || c.SLOClass == "background" {
			t.Errorf("client %q has SLOClass %q; inference_perf must use \"standard\" (regression: issue #965)",
				c.ID, c.SLOClass)
		}
	}
}
