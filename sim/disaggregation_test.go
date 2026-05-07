package sim

import (
	"fmt"
	"testing"
)

// TestNeverDisaggregate_AlwaysReturnsFalse verifies that NeverDisaggregate
// always returns Disaggregate=false regardless of input.
func TestNeverDisaggregate_AlwaysReturnsFalse(t *testing.T) {
	decider := &NeverDisaggregate{}
	req := &Request{ID: "req-1", InputTokens: make([]int, 100)}

	decision := decider.Decide(req, (*RouterState)(nil))

	if decision.Disaggregate {
		t.Error("NeverDisaggregate should return Disaggregate=false")
	}
}

// TestAlwaysDisaggregate_AlwaysReturnsTrue verifies that AlwaysDisaggregate
// always returns Disaggregate=true regardless of input.
func TestAlwaysDisaggregate_AlwaysReturnsTrue(t *testing.T) {
	decider := &AlwaysDisaggregate{}
	req := &Request{ID: "req-1", InputTokens: make([]int, 100)}

	decision := decider.Decide(req, (*RouterState)(nil))

	if !decision.Disaggregate {
		t.Error("AlwaysDisaggregate should return Disaggregate=true")
	}
}

// TestDisaggregationDecider_Interface verifies all implementations satisfy the interface.
func TestDisaggregationDecider_Interface(t *testing.T) {
	var _ DisaggregationDecider = &NeverDisaggregate{}
	var _ DisaggregationDecider = &AlwaysDisaggregate{}
	var _ DisaggregationDecider = &PrefixThresholdDecider{}
}

// TestNewDisaggregationDecider_Factory verifies factory dispatches correctly
// by asserting observable behavior (Decide output), not concrete type names.
func TestNewDisaggregationDecider_Factory(t *testing.T) {
	req := &Request{ID: "req-1", InputTokens: make([]int, 10)}
	tests := []struct {
		name             string
		wantDisaggregate bool
	}{
		{"", false},
		{"never", false},
		{"always", true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			d := NewDisaggregationDecider(tc.name)
			if d == nil {
				t.Fatal("NewDisaggregationDecider returned nil")
			}
			got := d.Decide(req, (*RouterState)(nil)).Disaggregate
			if got != tc.wantDisaggregate {
				t.Errorf("NewDisaggregationDecider(%q).Decide().Disaggregate = %v, want %v",
					tc.name, got, tc.wantDisaggregate)
			}
		})
	}
}

// TestNewDisaggregationDecider_UnknownPanics verifies factory panics on unknown name.
func TestNewDisaggregationDecider_UnknownPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for unknown decider name")
		}
	}()
	NewDisaggregationDecider("unknown-decider")
}

// TestNewDisaggregationDecider_ParameterizedPanic verifies factory panics with guidance
// for parameterized deciders that require typed constructors (R4: canonical constructor).
func TestNewDisaggregationDecider_ParameterizedPanic(t *testing.T) {
	tests := []struct {
		name string
	}{
		{"prefix-threshold"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic for parameterized decider %q", tc.name)
				}
			}()
			NewDisaggregationDecider(tc.name)
		})
	}
}

// TestIsValidDisaggregationDecider verifies the validation function.
func TestIsValidDisaggregationDecider(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"", true},
		{"never", true},
		{"always", true},
		{"prefix-threshold", true},
		{"unknown", false},
		{"NEVER", false}, // case-sensitive
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsValidDisaggregationDecider(tc.name); got != tc.want {
				t.Errorf("IsValidDisaggregationDecider(%q) = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

// TestValidDisaggregationDeciderNames verifies the names list.
func TestValidDisaggregationDeciderNames(t *testing.T) {
	names := ValidDisaggregationDeciderNames()
	if len(names) < 2 {
		t.Errorf("expected at least 2 decider names, got %d", len(names))
	}
	// Verify sorted order and no empty string
	for i, n := range names {
		if n == "" {
			t.Errorf("names[%d] is empty", i)
		}
		if i > 0 && names[i-1] >= n {
			t.Errorf("names not sorted: %q >= %q", names[i-1], n)
		}
	}
}

// TestDisaggregationDecider_INV9_OracleBoundary verifies that Decide() implementations
// do not access OutputTokens (INV-9 oracle boundary). This is a compile-time property
// enforced by reading only InputTokens and MaxOutputLen. The test verifies the observable
// behavior: decisions are the same regardless of OutputTokens content.
func TestDisaggregationDecider_INV9_OracleBoundary(t *testing.T) {
	req1 := &Request{ID: "req-1", InputTokens: make([]int, 100), OutputTokens: nil}
	req2 := &Request{ID: "req-2", InputTokens: make([]int, 100), OutputTokens: make([]int, 9999)}

	// Stub cacheQuery returning 0 blocks for any instance — exercises Decide's
	// formula path without depending on real cluster state.
	stubCache := map[string]func([]int) int{
		"decode_0": func([]int) int { return 0 },
	}
	state := &RouterState{
		Snapshots:        []RoutingSnapshot{{ID: "decode_0"}},
		SelectedInstance: "decode_0",
	}

	deciders := []DisaggregationDecider{
		&NeverDisaggregate{},
		&AlwaysDisaggregate{},
		NewPrefixThresholdDecider(512, 16, stubCache),
	}
	for _, d := range deciders {
		d1 := d.Decide(req1, state)
		d2 := d.Decide(req2, state)
		if d1 != d2 {
			t.Errorf("%T: decision differs with different OutputTokens (d1=%v, d2=%v); INV-9 violation",
				d, d1, d2)
		}
	}
}

// TestDisaggregationDecider_StateAgnostic verifies BC-2 (from PR #1265 / GAP-2):
// NeverDisaggregate and AlwaysDisaggregate ignore state entirely — nil vs populated
// state yields identical decisions. PrefixThresholdDecider is intentionally NOT in
// this list anymore: after GAP-3, its decision depends on state.SelectedInstance
// and the cache query map; it has its own dedicated tests below.
func TestDisaggregationDecider_StateAgnostic(t *testing.T) {
	req := &Request{ID: "req-1", InputTokens: make([]int, 100)}
	nilState := (*RouterState)(nil)
	populated := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "decode_0", QueueDepth: 7, KVUtilization: 0.4},
			{ID: "decode_1", QueueDepth: 2, KVUtilization: 0.9},
		},
		Clock:            12345,
		SelectedInstance: "decode_1",
	}

	deciders := []DisaggregationDecider{
		&NeverDisaggregate{},
		&AlwaysDisaggregate{},
	}
	for _, d := range deciders {
		gotNil := d.Decide(req, nilState)
		gotPop := d.Decide(req, populated)
		if gotNil != gotPop {
			t.Errorf("%T: decision differs between nil and populated state "+
				"(nil=%+v, populated=%+v); state-agnostic deciders must be invariant",
				d, gotNil, gotPop)
		}
	}
}

// TestDisaggregationDecider_BuiltinsReturnNoOverrides verifies that all built-in
// deciders leave DecodePodOverride and PrefillPodHint empty. The override fields
// are reserved for future joint D+P policies; preserving the empty-string default
// here pins the contract: every built-in caller can safely ignore these fields,
// and any regression that accidentally populates one will fail this test.
func TestDisaggregationDecider_BuiltinsReturnNoOverrides(t *testing.T) {
	req := &Request{ID: "req-1", InputTokens: make([]int, 100)}
	stubCache := map[string]func([]int) int{
		"decode_0": func([]int) int { return 0 },
	}
	state := &RouterState{
		Snapshots:        []RoutingSnapshot{{ID: "decode_0"}},
		SelectedInstance: "decode_0",
	}

	deciders := []DisaggregationDecider{
		&NeverDisaggregate{},
		&AlwaysDisaggregate{},
		NewPrefixThresholdDecider(512, 16, stubCache),
	}
	for _, d := range deciders {
		got := d.Decide(req, state)
		if got.DecodePodOverride != "" {
			t.Errorf("%T: DecodePodOverride = %q, want empty", d, got.DecodePodOverride)
		}
		if got.PrefillPodHint != "" {
			t.Errorf("%T: PrefillPodHint = %q, want empty", d, got.PrefillPodHint)
		}
	}
}

// --- PrefixThresholdDecider tests ---

// TestNewPrefixThresholdDecider_PanicsOnNegativeThreshold verifies constructor validates threshold.
func TestNewPrefixThresholdDecider_PanicsOnNegativeThreshold(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative threshold")
		}
	}()
	NewPrefixThresholdDecider(-1, 16, nil)
}

// TestNewPrefixThresholdDecider_PanicsOnZeroBlockSize verifies constructor validates blockSize.
func TestNewPrefixThresholdDecider_PanicsOnZeroBlockSize(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for zero blockSize")
		}
	}()
	NewPrefixThresholdDecider(512, 0, nil)
}

// coldState builds a RouterState with SelectedInstance set to selectedID.
// Pair with a coldCache(selectedID)-constructed decider to exercise the
// formula with cachedBlocks=0 (so nonCachedTokens == len(InputTokens)).
// RouterState itself carries no cache query; the cache query lives on the
// decider via cacheQuery.
func coldState(selectedID string) *RouterState {
	return &RouterState{
		Snapshots:        []RoutingSnapshot{{ID: selectedID}},
		SelectedInstance: selectedID,
	}
}

// coldCache builds a cacheQuery map with a single entry for selectedID whose
// closure always returns 0 cached blocks. Pair with coldState(selectedID) to
// exercise the decider's formula with cachedBlocks=0 — i.e. the selected pod
// is known but has none of the input cached.
func coldCache(selectedID string) map[string]func([]int) int {
	return map[string]func([]int) int{
		selectedID: func([]int) int { return 0 },
	}
}

// TestPrefixThresholdDecider_EmptyTokens verifies BC-3: empty input returns Disaggregate=false.
func TestPrefixThresholdDecider_EmptyTokens(t *testing.T) {
	decider := NewPrefixThresholdDecider(512, 16, coldCache("decode_0"))
	req := &Request{ID: "req-empty", InputTokens: []int{}}

	decision := decider.Decide(req, coldState("decode_0"))

	if decision.Disaggregate {
		t.Error("BC-3: empty InputTokens must return Disaggregate=false")
	}
}

// TestPrefixThresholdDecider_AboveThreshold verifies BC-2: non-cached tokens > threshold -> true.
// Uses threshold+1 as the boundary case to pin the strict > semantics (not >=).
func TestPrefixThresholdDecider_AboveThreshold(t *testing.T) {
	const blockSize = 16
	const threshold = 512
	decider := NewPrefixThresholdDecider(threshold, blockSize, coldCache("decode_0"))
	state := coldState("decode_0")

	tests := []struct {
		name   string
		tokens int
	}{
		{"threshold_plus_one", threshold + 1}, // tightest boundary: first value that must disaggregate
		{"well_above_threshold", 600},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tokens := make([]int, tc.tokens)
			for i := range tokens {
				tokens[i] = i + 1 // unique tokens
			}
			req := &Request{ID: fmt.Sprintf("req-%s", tc.name), InputTokens: tokens}

			decision := decider.Decide(req, state)

			if !decision.Disaggregate {
				t.Errorf("BC-2: %d uncached tokens with threshold=%d should return Disaggregate=true",
					tc.tokens, threshold)
			}
		})
	}
}

// TestPrefixThresholdDecider_BelowOrAtThreshold verifies BC-2: non-cached <= threshold -> false.
func TestPrefixThresholdDecider_BelowOrAtThreshold(t *testing.T) {
	const blockSize = 16
	const threshold = 512
	decider := NewPrefixThresholdDecider(threshold, blockSize, coldCache("decode_0"))
	state := coldState("decode_0")

	tests := []struct {
		name   string
		tokens int
	}{
		{"below_threshold", 100},
		{"exact_threshold", threshold},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tokens := make([]int, tc.tokens)
			for i := range tokens {
				tokens[i] = i + 1000 // unique tokens
			}
			req := &Request{ID: fmt.Sprintf("req-%s", tc.name), InputTokens: tokens}

			decision := decider.Decide(req, state)

			if decision.Disaggregate {
				t.Errorf("BC-2: %d non-cached tokens with threshold=%d should return Disaggregate=false",
					tc.tokens, threshold)
			}
		})
	}
}

// TestPrefixThresholdDecider_PerPodCacheQuery verifies BC-1: the decider reads
// cached block count from the SELECTED pod's closure, and BC-4: a cache hit on
// one pod does not influence a decision targeting a different pod.
//
// Scenario:
//   - decode_A has 40 cached blocks (640 tokens) for the input.
//   - decode_B has 0 cached blocks (cold).
//   - threshold=512, blockSize=16.
//   - Request has 840 input tokens.
//
// Expected:
//   - SelectedInstance="decode_A": nonCached = 840 - 40*16 = 200 ≤ 512 → false.
//   - SelectedInstance="decode_B": nonCached = 840 - 0*16  = 840 > 512 → true.
func TestPrefixThresholdDecider_PerPodCacheQuery(t *testing.T) {
	const blockSize = 16
	const threshold = 512
	cacheQuery := map[string]func([]int) int{
		"decode_A": func([]int) int { return 40 },
		"decode_B": func([]int) int { return 0 },
	}
	decider := NewPrefixThresholdDecider(threshold, blockSize, cacheQuery)

	tokens := make([]int, 840)
	for i := range tokens {
		tokens[i] = i + 1
	}
	req := &Request{ID: "req-multi-pod", InputTokens: tokens}

	tests := []struct {
		selectedInstance string
		wantDisagg       bool
		reason           string
	}{
		{"decode_A", false, "warm pod: 840-640=200 ≤ 512"},
		{"decode_B", true, "cold pod: 840-0=840 > 512"},
	}
	for _, tc := range tests {
		t.Run(tc.selectedInstance, func(t *testing.T) {
			state := &RouterState{
				Snapshots: []RoutingSnapshot{
					{ID: "decode_A"},
					{ID: "decode_B"},
				},
				SelectedInstance: tc.selectedInstance,
			}
			got := decider.Decide(req, state).Disaggregate
			if got != tc.wantDisagg {
				t.Errorf("SelectedInstance=%s: Disaggregate=%v, want %v (%s)",
					tc.selectedInstance, got, tc.wantDisagg, tc.reason)
			}
		})
	}
}

// TestPrefixThresholdDecider_MissingSelection_ReturnsFalse verifies BC-3 — the
// conservative-cold fallback paths. In any of these five scenarios the decider
// cannot locate a valid per-pod cache closure, so it returns Disaggregate=false:
//  1. state == nil
//  2. state.SelectedInstance == "" (upstream made no selection)
//  3. state.SelectedInstance is not a key in cacheQuery
//  4. cacheQuery[SelectedInstance] is nil (closure missing at the key)
//  5. cacheQuery itself is nil (unit tests without cluster state)
func TestPrefixThresholdDecider_MissingSelection_ReturnsFalse(t *testing.T) {
	const threshold = 100
	const blockSize = 16
	// Request is large enough that if the selected-pod path were taken with a
	// 0-blocks closure, the decision would be Disaggregate=true. So any test
	// that observes Disaggregate=false proves the early-return fired.
	tokens := make([]int, 400) // 400 > 100 threshold
	for i := range tokens {
		tokens[i] = i + 1
	}
	req := &Request{ID: "req-missing", InputTokens: tokens}

	tests := []struct {
		name       string
		cacheQuery map[string]func([]int) int
		state      *RouterState
	}{
		{
			name:       "nil_state",
			cacheQuery: coldCache("decode_0"),
			state:      nil,
		},
		{
			name:       "empty_selected_instance",
			cacheQuery: coldCache("decode_0"),
			state:      &RouterState{Snapshots: []RoutingSnapshot{{ID: "decode_0"}}, SelectedInstance: ""},
		},
		{
			name:       "unknown_selected_instance",
			cacheQuery: coldCache("decode_0"),
			state:      &RouterState{Snapshots: []RoutingSnapshot{{ID: "decode_0"}}, SelectedInstance: "decode_X"},
		},
		{
			name: "nil_closure_in_map",
			cacheQuery: map[string]func([]int) int{
				"decode_0": nil,
			},
			state: coldState("decode_0"),
		},
		{
			name:       "nil_cacheQuery_map",
			cacheQuery: nil,
			state:      coldState("decode_0"),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			decider := NewPrefixThresholdDecider(threshold, blockSize, tc.cacheQuery)
			got := decider.Decide(req, tc.state).Disaggregate
			if got {
				t.Errorf("BC-3 %s: expected Disaggregate=false (conservative fallback), got true", tc.name)
			}
		})
	}
}

// TestPrefixThresholdDecider_ZeroThreshold verifies threshold=0 means any
// non-empty request with non-zero non-cached tokens disaggregates. A fully-
// cached request (0 non-cached tokens) does NOT disaggregate (strict >).
func TestPrefixThresholdDecider_ZeroThreshold(t *testing.T) {
	const blockSize = 16

	// Case 1: cold pod, any non-empty input → disaggregate.
	decider := NewPrefixThresholdDecider(0, blockSize, coldCache("decode_0"))
	tokens := make([]int, 100)
	for i := range tokens {
		tokens[i] = i + 1
	}
	req := &Request{ID: "req-zero-thresh-cold", InputTokens: tokens}
	if !decider.Decide(req, coldState("decode_0")).Disaggregate {
		t.Error("threshold=0 with non-empty uncached tokens should return Disaggregate=true")
	}

	// Case 2: fully-cached prefix (cachedBlocks * blockSize == len(tokens))
	// → nonCached = 0, 0 > 0 is false → NOT disaggregate.
	//
	// This pins the strict > semantics at the boundary, guarding against an
	// accidental >= flip in the formula.
	fullCache := map[string]func([]int) int{
		"decode_0": func(t []int) int { return len(t) / blockSize },
	}
	fullyCachedDecider := NewPrefixThresholdDecider(0, blockSize, fullCache)
	exactBlocks := make([]int, 64) // 64 tokens = 4 full blocks
	for i := range exactBlocks {
		exactBlocks[i] = i + 1
	}
	reqFull := &Request{ID: "req-fully-cached", InputTokens: exactBlocks}
	if fullyCachedDecider.Decide(reqFull, coldState("decode_0")).Disaggregate {
		t.Error("threshold=0 with fully-cached tokens (nonCached=0) should return Disaggregate=false (strict >)")
	}
}

// TestPrefixThresholdDecider_QueriesOnlySelectedPod verifies BC-1 (isolation): when
// Decide runs, it invokes the closure for SelectedInstance exactly once and does not
// invoke closures for any other instance in the map. This guards against an accidental
// "scan all pods" regression (option (b) in the #1265 review comment was explicitly
// rejected; we implement option (a) — query only the selected pod).
func TestPrefixThresholdDecider_QueriesOnlySelectedPod(t *testing.T) {
	var aCalls, bCalls int
	cacheQuery := map[string]func([]int) int{
		"decode_A": func([]int) int { aCalls++; return 0 },
		"decode_B": func([]int) int { bCalls++; return 0 },
	}
	decider := NewPrefixThresholdDecider(512, 16, cacheQuery)

	req := &Request{ID: "req-isolation", InputTokens: make([]int, 100)}
	for i := range req.InputTokens {
		req.InputTokens[i] = i + 1
	}
	state := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "decode_A"},
			{ID: "decode_B"},
		},
		SelectedInstance: "decode_A",
	}
	decider.Decide(req, state)

	if aCalls != 1 {
		t.Errorf("decode_A closure called %d times, want 1", aCalls)
	}
	if bCalls != 0 {
		t.Errorf("decode_B closure called %d times, want 0 (must not query non-selected pods)", bCalls)
	}
}
