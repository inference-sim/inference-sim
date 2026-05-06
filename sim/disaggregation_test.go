package sim

import (
	"fmt"
	"testing"
)

// newTestRequestView builds a RequestView from an input-token slice for tests.
func newTestRequestView(id string, tokens []int) RequestView {
	return RequestView{ID: id, InputTokens: tokens}
}

// stubCacheQuery returns a DisaggregationContext whose DecodeCacheQuery returns the
// configured cachedBlocks count regardless of the token slice passed in.
func stubCacheQuery(instanceID string, cachedBlocks int) DisaggregationContext {
	return NewDisaggregationContext(instanceID, func(_ []int) int { return cachedBlocks })
}

// emptyCtx is a DisaggregationContext with no cache-query closure (e.g., when
// disaggregation is disabled at the cluster layer). DecodeCacheQuery returns 0.
func emptyCtx() DisaggregationContext { return DisaggregationContext{} }

// TestNeverDisaggregate_AlwaysReturnsFalse verifies that NeverDisaggregate
// always returns Disaggregate=false regardless of input.
func TestNeverDisaggregate_AlwaysReturnsFalse(t *testing.T) {
	decider := &NeverDisaggregate{}
	view := newTestRequestView("req-1", make([]int, 100))

	decision := decider.Decide(view, emptyCtx())

	if decision.Disaggregate {
		t.Error("NeverDisaggregate should return Disaggregate=false")
	}
}

// TestAlwaysDisaggregate_AlwaysReturnsTrue verifies that AlwaysDisaggregate
// always returns Disaggregate=true regardless of input.
func TestAlwaysDisaggregate_AlwaysReturnsTrue(t *testing.T) {
	decider := &AlwaysDisaggregate{}
	view := newTestRequestView("req-1", make([]int, 100))

	decision := decider.Decide(view, emptyCtx())

	if !decision.Disaggregate {
		t.Error("AlwaysDisaggregate should return Disaggregate=true")
	}
}

// TestDisaggregationDecider_Interface verifies all implementations satisfy the interface.
func TestDisaggregationDecider_Interface(t *testing.T) {
	var _ DisaggregationDecider = &NeverDisaggregate{}
	var _ DisaggregationDecider = &AlwaysDisaggregate{}
	var _ DisaggregationDecider = &PrefixBasedPDDecider{}
	var _ DisaggregationDecider = &GlobalPrefixThresholdDecider{}
}

// TestNewDisaggregationDecider_Factory verifies factory dispatches correctly
// by asserting observable behavior (Decide output), not concrete type names.
func TestNewDisaggregationDecider_Factory(t *testing.T) {
	view := newTestRequestView("req-1", make([]int, 10))
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
			got := d.Decide(view, emptyCtx()).Disaggregate
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
		{"prefix-based-pd-decider"},
		{"global-prefix-threshold"},
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
		{"prefix-based-pd-decider", true},
		{"global-prefix-threshold", true},
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

// TestRequestView_StructurallyHidesOutputTokens verifies BEH-5: RequestView does not
// expose OutputTokens (INV-9 enforced structurally). A compile-check test body is
// redundant because the *absence* of the field is itself the invariant; instead this
// test pins the contract that NewRequestView copies only the allowed fields and leaves
// the OutputTokens oracle behind.
func TestRequestView_StructurallyHidesOutputTokens(t *testing.T) {
	req := &Request{
		ID:           "req-1",
		InputTokens:  []int{1, 2, 3},
		OutputTokens: []int{4, 5, 6, 7, 8}, // deliberately distinct length from InputTokens
		MaxOutputLen: 128,
		SLOClass:     "standard",
		Priority:     3.0,
		ArrivalTime:  1000,
	}
	view := NewRequestView(req)

	if view.ID != "req-1" || len(view.InputTokens) != 3 ||
		view.MaxOutputLen != 128 || view.SLOClass != "standard" ||
		view.Priority != 3.0 || view.ArrivalTime != 1000 {
		t.Errorf("NewRequestView copied fields incorrectly: %+v", view)
	}
	// The following would not compile (proving structural enforcement):
	//   _ = view.OutputTokens
	// Keeping it as a documented comment rather than attempting a runtime assertion.
}

// TestDisaggregationDecider_INV9_OracleBoundary verifies that Decide() implementations
// produce the same decision regardless of OutputTokens content on the underlying
// *Request — which is now trivially true because the decider only sees RequestView.
// Test kept as a regression guard in case a future refactor re-widens the signature.
func TestDisaggregationDecider_INV9_OracleBoundary(t *testing.T) {
	req1 := &Request{ID: "req-1", InputTokens: make([]int, 100), OutputTokens: nil}
	req2 := &Request{ID: "req-2", InputTokens: make([]int, 100), OutputTokens: make([]int, 9999)}

	deciders := []DisaggregationDecider{
		&NeverDisaggregate{},
		&AlwaysDisaggregate{},
		NewPrefixBasedPDDecider(512, 16),
		NewGlobalPrefixThresholdDecider(512, 16),
	}
	for _, d := range deciders {
		d1 := d.Decide(NewRequestView(req1), emptyCtx())
		d2 := d.Decide(NewRequestView(req2), emptyCtx())
		if d1 != d2 {
			t.Errorf("%T: decision differs with different OutputTokens (d1=%v, d2=%v); INV-9 violation",
				d, d1, d2)
		}
	}
}

// TestDisaggregationContext_DecodeCacheQueryNilClosure verifies DisaggregationContext
// degrades gracefully when no cache-query closure is installed (returns 0 rather than
// panicking), so that a decider can still be invoked safely at the disaggregation-
// disabled call site.
func TestDisaggregationContext_DecodeCacheQueryNilClosure(t *testing.T) {
	ctx := NewDisaggregationContext("instance_0", nil)
	if got := ctx.DecodeCacheQuery([]int{1, 2, 3}); got != 0 {
		t.Errorf("DecodeCacheQuery with nil closure: got %d, want 0", got)
	}
	if got := ctx.DecodeInstanceID(); got != "instance_0" {
		t.Errorf("DecodeInstanceID() = %q, want %q", got, "instance_0")
	}
}

// --- PrefixBasedPDDecider tests ---

// TestNewPrefixBasedPDDecider_PanicsOnNegativeThreshold verifies constructor validates threshold.
func TestNewPrefixBasedPDDecider_PanicsOnNegativeThreshold(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative threshold")
		}
	}()
	NewPrefixBasedPDDecider(-1, 16)
}

// TestNewPrefixBasedPDDecider_PanicsOnZeroBlockSize verifies constructor validates blockSize.
func TestNewPrefixBasedPDDecider_PanicsOnZeroBlockSize(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for zero blockSize")
		}
	}()
	NewPrefixBasedPDDecider(512, 0)
}

// TestPrefixBasedPDDecider_EmptyTokens verifies BC-PD-20: empty input returns Disaggregate=false.
func TestPrefixBasedPDDecider_EmptyTokens(t *testing.T) {
	decider := NewPrefixBasedPDDecider(512, 16)
	view := newTestRequestView("req-empty", []int{})

	decision := decider.Decide(view, emptyCtx())

	if decision.Disaggregate {
		t.Error("BC-PD-20: empty InputTokens must return Disaggregate=false")
	}
}

// TestPrefixBasedPDDecider_AboveThreshold verifies BEH-2: when per-pod non-cached tokens
// exceed the threshold, Disaggregate=true.
func TestPrefixBasedPDDecider_AboveThreshold(t *testing.T) {
	const blockSize = 16
	const threshold = 512
	decider := NewPrefixBasedPDDecider(threshold, blockSize)

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
			view := newTestRequestView(fmt.Sprintf("req-%s", tc.name), tokens)

			decision := decider.Decide(view, stubCacheQuery("instance_0", 0))

			if !decision.Disaggregate {
				t.Errorf("BEH-2: %d uncached tokens with threshold=%d should return Disaggregate=true",
					tc.tokens, threshold)
			}
		})
	}
}

// TestPrefixBasedPDDecider_BelowOrAtThreshold verifies BEH-2: non-cached <= threshold -> false.
func TestPrefixBasedPDDecider_BelowOrAtThreshold(t *testing.T) {
	const blockSize = 16
	const threshold = 512
	decider := NewPrefixBasedPDDecider(threshold, blockSize)

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
			view := newTestRequestView(fmt.Sprintf("req-%s", tc.name), tokens)

			decision := decider.Decide(view, stubCacheQuery("instance_0", 0))

			if decision.Disaggregate {
				t.Errorf("BEH-2: %d non-cached tokens with threshold=%d should return Disaggregate=false",
					tc.tokens, threshold)
			}
		})
	}
}

// TestPrefixBasedPDDecider_PerPodCacheReducesNonCached verifies BEH-2: when the selected
// decode pod reports N cached blocks, nonCachedTokens = len(InputTokens) - N*blockSize.
// Scenario: 840 input tokens, decode pod has 40 cached blocks (640 tokens). nonCached =
// 840 - 640 = 200 <= 512 threshold -> do NOT disaggregate.
func TestPrefixBasedPDDecider_PerPodCacheReducesNonCached(t *testing.T) {
	const blockSize = 16
	const threshold = 512
	decider := NewPrefixBasedPDDecider(threshold, blockSize)

	tokens := make([]int, 840)
	view := newTestRequestView("req-cached", tokens)

	decision := decider.Decide(view, stubCacheQuery("instance_0", 40))

	if decision.Disaggregate {
		t.Errorf("BEH-2: decode pod reports 40 cached blocks (640 tokens) of 840 input; nonCached=200 <= threshold=%d should not disaggregate", threshold)
	}
}

// TestPrefixBasedPDDecider_ZeroThresholdDisablesDisaggregation verifies BEH-3: the
// threshold==0 short-circuit matches llm-d's NonCachedTokens==0 guard.
func TestPrefixBasedPDDecider_ZeroThresholdDisablesDisaggregation(t *testing.T) {
	const blockSize = 16
	decider := NewPrefixBasedPDDecider(0, blockSize)

	tokens := make([]int, 100)
	for i := range tokens {
		tokens[i] = i + 1
	}
	view := newTestRequestView("req-zero-thresh", tokens)

	// Even with no cached blocks (full input is non-cached), threshold=0 must
	// short-circuit to Disaggregate=false.
	decision := decider.Decide(view, stubCacheQuery("instance_0", 0))
	if decision.Disaggregate {
		t.Error("BEH-3: threshold==0 must short-circuit to Disaggregate=false")
	}
}

// TestPrefixBasedPDDecider_NegativeNonCachedGuard verifies defensive handling when a
// misbehaving cache-query closure reports more cached blocks than the input contains.
// The decider must not produce a spurious Disaggregate=true from a negative comparison.
func TestPrefixBasedPDDecider_NegativeNonCachedGuard(t *testing.T) {
	const blockSize = 16
	const threshold = 10
	decider := NewPrefixBasedPDDecider(threshold, blockSize)

	// 16 tokens input, buggy closure reports 100 cached blocks (1600 tokens).
	// Raw nonCached = -1584; decider must clamp to 0 and return Disaggregate=false.
	view := newTestRequestView("req-oversized-cache", make([]int, 16))

	decision := decider.Decide(view, stubCacheQuery("instance_0", 100))

	if decision.Disaggregate {
		t.Error("Defensive: negative nonCached must clamp to 0; Disaggregate must remain false")
	}
}

// TestDisaggregationContext_PassedByValue_NoSharedMutation verifies that the caller's
// cache-query closure is invoked through the value-typed context without exposing a
// mutable handle to the decider. Deciders cannot rebind the closure because the field
// is unexported.
func TestDisaggregationContext_PassedByValue_NoSharedMutation(t *testing.T) {
	callCount := 0
	ctx := NewDisaggregationContext("instance_0", func(_ []int) int {
		callCount++
		return 0
	})
	decider := NewPrefixBasedPDDecider(100, 16)

	view := newTestRequestView("req-1", make([]int, 200))
	decider.Decide(view, ctx)

	if callCount != 1 {
		t.Errorf("DecodeCacheQuery: got %d invocations, want 1", callCount)
	}
}

// --- GlobalPrefixThresholdDecider tests ---

// noopDisaggregationObserver is a no-op DisaggregationObserver used in tests to satisfy R13:
// DisaggregationObserver must work for >=2 backends.
type noopDisaggregationObserver struct{}

func (*noopDisaggregationObserver) ObserveRouting(_ *Request, _ string) {}

// TestGlobalPrefixThresholdDecider_Interface verifies GlobalPrefixThresholdDecider satisfies both interfaces.
// Also verifies DisaggregationObserver is a general extension point (R13: works for >=2 backends).
func TestGlobalPrefixThresholdDecider_Interface(t *testing.T) {
	var _ DisaggregationDecider = &GlobalPrefixThresholdDecider{}
	var _ DisaggregationObserver = &GlobalPrefixThresholdDecider{}
	var _ DisaggregationObserver = &noopDisaggregationObserver{} // R13: second backend
}

// TestGlobalPrefixThresholdDecider_EmptyTokens verifies BC-PD-20: empty input returns Disaggregate=false.
func TestGlobalPrefixThresholdDecider_EmptyTokens(t *testing.T) {
	decider := NewGlobalPrefixThresholdDecider(512, 16)
	view := newTestRequestView("req-empty", []int{})

	decision := decider.Decide(view, emptyCtx())

	if decision.Disaggregate {
		t.Error("BC-PD-20: empty InputTokens must return Disaggregate=false")
	}
}

// TestGlobalPrefixThresholdDecider_AboveThreshold verifies the counterfactual baseline:
// without observations, 400 > 200 threshold -> disaggregate.
func TestGlobalPrefixThresholdDecider_AboveThreshold(t *testing.T) {
	decider := NewGlobalPrefixThresholdDecider(200, 16)
	tokens := make([]int, 400)
	for i := range tokens {
		tokens[i] = i + 1
	}
	view := newTestRequestView("req", tokens)

	decision := decider.Decide(view, emptyCtx())
	if !decision.Disaggregate {
		t.Error("GlobalPrefixThresholdDecider: 400 non-cached tokens > 200 threshold should disaggregate")
	}
}

// TestGlobalPrefixThresholdDecider_ObserverWarmsGlobalLRU verifies the counterfactual
// cache semantics: a prior ObserveRouting call warms the global virtual LRU; a subsequent
// Decide on an overlapping prefix sees the cached blocks.
func TestGlobalPrefixThresholdDecider_ObserverWarmsGlobalLRU(t *testing.T) {
	const blockSize = 16
	const threshold = 300
	decider := NewGlobalPrefixThresholdDecider(threshold, blockSize)

	prefix := make([]int, 400) // 25 blocks * 16 tokens
	for i := range prefix {
		prefix[i] = i + 1
	}

	warmReq := &Request{ID: "req-warm", InputTokens: prefix}
	// Call Decide to populate cachedHashes, then ObserveRouting to record them.
	decider.Decide(NewRequestView(warmReq), emptyCtx())
	decider.ObserveRouting(warmReq, "instance_any") // instance ID ignored by the global variant

	// Extended request: prefix + 50 new tokens = 450 total.
	extended := make([]int, len(prefix)+50)
	copy(extended, prefix)
	for i := len(prefix); i < len(extended); i++ {
		extended[i] = 10000 + i
	}
	view := newTestRequestView("req-follow", extended)

	decision := decider.Decide(view, emptyCtx())
	if decision.Disaggregate {
		t.Error("GlobalPrefixThresholdDecider: 450 tokens with 400-token prefix in global LRU should leave 50 non-cached <= 300 threshold; expected Disaggregate=false")
	}
}

// TestGlobalPrefixThresholdDecider_ZeroThresholdDisablesDisaggregation verifies the
// counterfactual decider also honors BEH-3 (threshold==0 => disabled) for consistency.
func TestGlobalPrefixThresholdDecider_ZeroThresholdDisablesDisaggregation(t *testing.T) {
	decider := NewGlobalPrefixThresholdDecider(0, 16)
	view := newTestRequestView("req", make([]int, 100))

	decision := decider.Decide(view, emptyCtx())
	if decision.Disaggregate {
		t.Error("BEH-3: GlobalPrefixThresholdDecider threshold==0 must short-circuit to Disaggregate=false")
	}
}

// TestNewPrefixThresholdDecider_ReturnsPerPodVariant verifies the backward-compatible
// constructor now returns a *PrefixBasedPDDecider (behavior change documented in #1250).
func TestNewPrefixThresholdDecider_ReturnsPerPodVariant(t *testing.T) {
	d := NewPrefixThresholdDecider(512, 16)
	if _, ok := any(d).(*PrefixBasedPDDecider); !ok {
		t.Errorf("NewPrefixThresholdDecider must return *PrefixBasedPDDecider (per-pod variant), got %T", d)
	}
}
