package sim

import (
	"fmt"
	"testing"
)

// TestNeverDisaggregate_AlwaysReturnsFalse verifies that NeverDisaggregate
// always returns Disaggregate=false regardless of input.
func TestNeverDisaggregate_AlwaysReturnsFalse(t *testing.T) {
	decider := &NeverDisaggregate{}
	req := &Request{ID: "req-1", InputTokens: make([]TokenID, 100)}

	decision := decider.Decide(req, (*RouterState)(nil))

	if decision.Disaggregate {
		t.Error("NeverDisaggregate should return Disaggregate=false")
	}
}

// TestAlwaysDisaggregate_AlwaysReturnsTrue verifies that AlwaysDisaggregate
// always returns Disaggregate=true regardless of input.
func TestAlwaysDisaggregate_AlwaysReturnsTrue(t *testing.T) {
	decider := &AlwaysDisaggregate{}
	req := &Request{ID: "req-1", InputTokens: make([]TokenID, 100)}

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
	req := &Request{ID: "req-1", InputTokens: make([]TokenID, 10)}
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
	req1 := &Request{ID: "req-1", InputTokens: make([]TokenID, 100), OutputTokens: nil}
	req2 := &Request{ID: "req-2", InputTokens: make([]TokenID, 100), OutputTokens: make([]TokenID, 9999)}

	// Stub cacheQuery returning 0 blocks for any instance — exercises Decide's
	// formula path without depending on real cluster state.
	stubCache := map[string]func([]TokenID) int{
		"decode_0": func([]TokenID) int { return 0 },
	}
	state := &RouterState{
		Snapshots:        []RoutingSnapshot{{ID: "decode_0"}},
		SelectedDecodeInstance: "decode_0",
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
// this list anymore: after GAP-3, its decision depends on state.SelectedDecodeInstance
// and the cache query map; it has its own dedicated tests below.
func TestDisaggregationDecider_StateAgnostic(t *testing.T) {
	req := &Request{ID: "req-1", InputTokens: make([]TokenID, 100)}
	nilState := (*RouterState)(nil)
	populated := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "decode_0", QueueDepth: 7, KVUtilization: 0.4},
			{ID: "decode_1", QueueDepth: 2, KVUtilization: 0.9},
		},
		Clock:            12345,
		SelectedDecodeInstance: "decode_1",
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
	req := &Request{ID: "req-1", InputTokens: make([]TokenID, 100)}
	stubCache := map[string]func([]TokenID) int{
		"decode_0": func([]TokenID) int { return 0 },
	}
	state := &RouterState{
		Snapshots:        []RoutingSnapshot{{ID: "decode_0"}},
		SelectedDecodeInstance: "decode_0",
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

// coldState builds a RouterState with SelectedDecodeInstance set to selectedID.
// Pair with a coldCache(selectedID)-constructed decider to exercise the
// formula with cachedBlocks=0 (so nonCachedTokens == len(InputTokens)).
// RouterState itself carries no cache query; the cache query lives on the
// decider via cacheQuery.
func coldState(selectedID string) *RouterState {
	return &RouterState{
		Snapshots:        []RoutingSnapshot{{ID: selectedID}},
		SelectedDecodeInstance: selectedID,
	}
}

// coldCache builds a cacheQuery map with a single entry for selectedID whose
// closure always returns 0 cached blocks. Pair with coldState(selectedID) to
// exercise the decider's formula with cachedBlocks=0 — i.e. the selected pod
// is known but has none of the input cached.
func coldCache(selectedID string) map[string]func([]TokenID) int {
	return map[string]func([]TokenID) int{
		selectedID: func([]TokenID) int { return 0 },
	}
}

// TestPrefixThresholdDecider_EmptyTokens verifies BC-3: empty input returns Disaggregate=false.
func TestPrefixThresholdDecider_EmptyTokens(t *testing.T) {
	decider := NewPrefixThresholdDecider(512, 16, coldCache("decode_0"))
	req := &Request{ID: "req-empty", InputTokens:  []TokenID{}}

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
			tokens := make([]TokenID, tc.tokens)
			for i := range tokens {
				tokens[i] = TokenID(i + 1) // unique tokens
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
			tokens := make([]TokenID, tc.tokens)
			for i := range tokens {
				tokens[i] = TokenID(i + 1000) // unique tokens
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
//   - SelectedDecodeInstance="decode_A": nonCached = 840 - 40*16 = 200 ≤ 512 → false.
//   - SelectedDecodeInstance="decode_B": nonCached = 840 - 0*16  = 840 > 512 → true.
func TestPrefixThresholdDecider_PerPodCacheQuery(t *testing.T) {
	const blockSize = 16
	const threshold = 512
	cacheQuery := map[string]func([]TokenID) int{
		"decode_A": func([]TokenID) int { return 40 },
		"decode_B": func([]TokenID) int { return 0 },
	}
	decider := NewPrefixThresholdDecider(threshold, blockSize, cacheQuery)

	tokens := make([]TokenID, 840)
	for i := range tokens {
		tokens[i] = TokenID(i + 1)
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
				SelectedDecodeInstance: tc.selectedInstance,
			}
			got := decider.Decide(req, state).Disaggregate
			if got != tc.wantDisagg {
				t.Errorf("SelectedDecodeInstance=%s: Disaggregate=%v, want %v (%s)",
					tc.selectedInstance, got, tc.wantDisagg, tc.reason)
			}
		})
	}
}

// TestPrefixThresholdDecider_MissingSelection_ReturnsFalse verifies BC-3 — the
// conservative-cold fallback paths. In any of these five scenarios the decider
// cannot locate a valid per-pod cache closure, so it returns Disaggregate=false:
//  1. state == nil
//  2. state.SelectedDecodeInstance == "" (upstream made no selection)
//  3. state.SelectedDecodeInstance is not a key in cacheQuery
//  4. cacheQuery[SelectedDecodeInstance] is nil (closure missing at the key)
//  5. cacheQuery itself is nil (unit tests without cluster state)
func TestPrefixThresholdDecider_MissingSelection_ReturnsFalse(t *testing.T) {
	const threshold = 100
	const blockSize = 16
	// Request is large enough that if the selected-pod path were taken with a
	// 0-blocks closure, the decision would be Disaggregate=true. So any test
	// that observes Disaggregate=false proves the early-return fired.
	tokens := make([]TokenID, 400) // 400 > 100 threshold
	for i := range tokens {
		tokens[i] = TokenID(i + 1)
	}
	req := &Request{ID: "req-missing", InputTokens: tokens}

	tests := []struct {
		name       string
		cacheQuery map[string]func([]TokenID) int
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
			state:      &RouterState{Snapshots: []RoutingSnapshot{{ID: "decode_0"}}, SelectedDecodeInstance: ""},
		},
		{
			name:       "unknown_selected_instance",
			cacheQuery: coldCache("decode_0"),
			state:      &RouterState{Snapshots: []RoutingSnapshot{{ID: "decode_0"}}, SelectedDecodeInstance: "decode_X"},
		},
		{
			name: "nil_closure_in_map",
			cacheQuery: map[string]func([]TokenID) int{
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
	tokens := make([]TokenID, 100)
	for i := range tokens {
		tokens[i] = TokenID(i + 1)
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
	fullCache := map[string]func([]TokenID) int{
		"decode_0": func(t []TokenID) int { return len(t) / blockSize },
	}
	fullyCachedDecider := NewPrefixThresholdDecider(0, blockSize, fullCache)
	exactBlocks := make([]TokenID, 64) // 64 tokens = 4 full blocks
	for i := range exactBlocks {
		exactBlocks[i] = TokenID(i + 1)
	}
	reqFull := &Request{ID: "req-fully-cached", InputTokens: exactBlocks}
	if fullyCachedDecider.Decide(reqFull, coldState("decode_0")).Disaggregate {
		t.Error("threshold=0 with fully-cached tokens (nonCached=0) should return Disaggregate=false (strict >)")
	}
}

// TestPrefixThresholdDecider_QueriesOnlySelectedPod verifies BC-1 (isolation): when
// Decide runs, it invokes the closure for SelectedDecodeInstance exactly once and does not
// invoke closures for any other instance in the map. This guards against an accidental
// "scan all pods" regression (option (b) in the #1265 review comment was explicitly
// rejected; we implement option (a) — query only the selected pod).
func TestPrefixThresholdDecider_QueriesOnlySelectedPod(t *testing.T) {
	var aCalls, bCalls int
	cacheQuery := map[string]func([]TokenID) int{
		"decode_A": func([]TokenID) int { aCalls++; return 0 },
		"decode_B": func([]TokenID) int { bCalls++; return 0 },
	}
	decider := NewPrefixThresholdDecider(512, 16, cacheQuery)

	req := &Request{ID: "req-isolation", InputTokens: make([]TokenID, 100)}
	for i := range req.InputTokens {
		req.InputTokens[i] = TokenID(i + 1)
	}
	state := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "decode_A"},
			{ID: "decode_B"},
		},
		SelectedDecodeInstance: "decode_A",
	}
	decider.Decide(req, state)

	if aCalls != 1 {
		t.Errorf("decode_A closure called %d times, want 1", aCalls)
	}
	if bCalls != 0 {
		t.Errorf("decode_B closure called %d times, want 0 (must not query non-selected pods)", bCalls)
	}
}

// TestDisaggregationDecider_IgnoresNewFields verifies BC-4 (issue #1339):
// built-in deciders (never, always, prefix-threshold) produce byte-identical
// decisions regardless of PrefillSnapshots content, per-snapshot CachedBlocks,
// or SelectedPrefillInstance. This pins the parity contract: the new
// cross-pool signal envelope is additive, and adopting it cannot regress
// existing decisions.
func TestDisaggregationDecider_IgnoresNewFields(t *testing.T) {
	tokens := make([]int, 200)
	for i := range tokens {
		tokens[i] = i + 1
	}
	req := &Request{ID: "req-parity", InputTokens: tokens}

	// Decode-pool snapshot with known CachedBlocks=0 (bare) vs non-zero (populated).
	decodeBare := []RoutingSnapshot{{ID: "decode_0"}, {ID: "decode_1"}}
	decodePopulated := []RoutingSnapshot{
		{ID: "decode_0", CachedBlocks: 3, KVUtilization: 0.4},
		{ID: "decode_1", CachedBlocks: 7, KVUtilization: 0.7},
	}
	prefillPopulated := []RoutingSnapshot{
		{ID: "prefill_0", CachedBlocks: 5, QueueDepth: 2},
		{ID: "prefill_1", CachedBlocks: 0, QueueDepth: 4},
	}

	// cacheQuery used only by prefix-threshold. Wire decode_0 to 0 blocks so the
	// decision depends solely on len(InputTokens) > threshold (200 > 100 → true).
	cacheQuery := map[string]func([]int) int{
		"decode_0": func([]int) int { return 0 },
		"decode_1": func([]int) int { return 0 },
	}

	stateBare := &RouterState{
		Snapshots:              decodeBare,
		SelectedDecodeInstance: "decode_0",
	}
	statePopulated := &RouterState{
		Snapshots:               decodePopulated,
		PrefillSnapshots:        prefillPopulated,
		SelectedDecodeInstance:  "decode_0",
		SelectedPrefillInstance: "prefill_0", // intentionally non-empty to prove built-ins ignore it
		Clock:                   12345,
	}

	deciders := []DisaggregationDecider{
		&NeverDisaggregate{},
		&AlwaysDisaggregate{},
		NewPrefixThresholdDecider(100, 16, cacheQuery),
	}
	for _, d := range deciders {
		bare := d.Decide(req, stateBare)
		populated := d.Decide(req, statePopulated)
		if bare != populated {
			t.Errorf("%T: decision differs between bare and populated state "+
				"(bare=%+v, populated=%+v); BC-4 requires built-in deciders to ignore "+
				"PrefillSnapshots, per-snapshot CachedBlocks, and SelectedPrefillInstance",
				d, bare, populated)
		}
	}
}

// ---------------------------------------------------------------------------
// EncodeDecider tests (GAP-4, issue #1264)
// ---------------------------------------------------------------------------

// TestAlwaysEncode verifies AlwaysEncode.ShouldEncode returns true for any
// request, regardless of modality counts or OutputTokens presence.
func TestAlwaysEncode(t *testing.T) {
	d := &AlwaysEncode{}
	cases := []*Request{
		{ID: "text", InputTokens: make([]TokenID, 32)},
		{ID: "image", ImageTokenCount: 10},
		{ID: "nil-tokens", OutputTokens: nil},
	}
	for _, r := range cases {
		if !d.ShouldEncode(r, "inst_0") {
			t.Errorf("AlwaysEncode.ShouldEncode(%q) = false, want true", r.ID)
		}
	}
}

// TestNeverEncode verifies NeverEncode.ShouldEncode returns false unconditionally.
func TestNeverEncode(t *testing.T) {
	d := &NeverEncode{}
	if d.ShouldEncode(&Request{ImageTokenCount: 999}, "x") {
		t.Error("NeverEncode.ShouldEncode must always return false")
	}
}

// TestMultimodalEncodeDecider_True verifies BC-EPD-2 positive case: the
// decider returns true when any per-modality token count is > 0.
func TestMultimodalEncodeDecider_True(t *testing.T) {
	d := &MultimodalEncodeDecider{}
	cases := []struct {
		name string
		req  *Request
	}{
		{"image only", &Request{ImageTokenCount: 10}},
		{"audio only", &Request{AudioTokenCount: 5}},
		{"video only", &Request{VideoTokenCount: 1}},
		{"image + text", &Request{TextTokenCount: 100, ImageTokenCount: 20}},
		{"all three", &Request{ImageTokenCount: 1, AudioTokenCount: 1, VideoTokenCount: 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !d.ShouldEncode(tc.req, "inst_0") {
				t.Errorf("ShouldEncode(%+v) = false, want true", tc.req)
			}
		})
	}
}

// TestMultimodalEncodeDecider_False verifies BC-EPD-2 negative case: the
// decider returns false for text-only requests (all modality counts zero).
func TestMultimodalEncodeDecider_False(t *testing.T) {
	d := &MultimodalEncodeDecider{}
	cases := []struct {
		name string
		req  *Request
	}{
		{"all zero", &Request{}},
		{"text-only", &Request{TextTokenCount: 500}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if d.ShouldEncode(tc.req, "inst_0") {
				t.Errorf("ShouldEncode(%+v) = true, want false", tc.req)
			}
		})
	}
}

// TestMultimodalEncodeDecider_IgnoresOutputTokens verifies BC-EPD-7 (INV-9
// oracle boundary): the decider makes its decision independently of
// Request.OutputTokens. The probe runs twice — once with OutputTokens=nil,
// once with a large populated OutputTokens slice — and asserts the decision
// is identical. Nil-safety is proven by the first run; decision independence
// is proven by the second run agreeing.
func TestMultimodalEncodeDecider_IgnoresOutputTokens(t *testing.T) {
	d := &MultimodalEncodeDecider{}
	reqNoOutput := &Request{ID: "oracle-probe-nil", ImageTokenCount: 5, OutputTokens: nil}
	reqLargeOutput := &Request{ID: "oracle-probe-large", ImageTokenCount: 5, OutputTokens: make([]TokenID, 9999)}
	decisionNil := d.ShouldEncode(reqNoOutput, "inst_0")
	decisionLarge := d.ShouldEncode(reqLargeOutput, "inst_0")
	if !decisionNil {
		t.Error("ShouldEncode(OutputTokens=nil) = false, want true for a multimodal request")
	}
	if decisionNil != decisionLarge {
		t.Errorf("decision depends on OutputTokens: nil=%v, large=%v — INV-9 violation", decisionNil, decisionLarge)
	}
}

// TestIsValidEncodeDecider verifies the encode-decider name validity set.
func TestIsValidEncodeDecider(t *testing.T) {
	cases := []struct {
		name string
		want bool
	}{
		{"", true}, // empty defaults to "never"
		{"never", true},
		{"always", true},
		{"multimodal", true},
		{"prefix-threshold", false}, // DisaggregationDecider name, not an encode decider
		{"bogus", false},
		{"Always", false}, // case-sensitive
	}
	for _, tc := range cases {
		if got := IsValidEncodeDecider(tc.name); got != tc.want {
			t.Errorf("IsValidEncodeDecider(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestNewEncodeDecider verifies factory dispatch and panic on unknown names.
func TestNewEncodeDecider(t *testing.T) {
	// Valid names return a non-nil decider.
	for _, name := range []string{"", "never", "always", "multimodal"} {
		d := NewEncodeDecider(name)
		if d == nil {
			t.Errorf("NewEncodeDecider(%q) returned nil", name)
		}
	}
	// Empty string defaults to the never-encode behavior (behavioral check, not
	// structural: would survive a rename of *NeverEncode).
	defaultDecider := NewEncodeDecider("")
	if defaultDecider.ShouldEncode(&Request{ImageTokenCount: 10}, "inst_0") {
		t.Errorf("NewEncodeDecider(\"\") must behave like NeverEncode (returned true for multimodal)")
	}
	// Unknown name panics.
	defer func() {
		if r := recover(); r == nil {
			t.Error("NewEncodeDecider(\"bogus\") did not panic")
		}
	}()
	_ = NewEncodeDecider("bogus")
}
