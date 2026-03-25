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

	decision := decider.Decide(req)

	if decision.Disaggregate {
		t.Error("NeverDisaggregate should return Disaggregate=false")
	}
}

// TestAlwaysDisaggregate_AlwaysReturnsTrue verifies that AlwaysDisaggregate
// always returns Disaggregate=true regardless of input.
func TestAlwaysDisaggregate_AlwaysReturnsTrue(t *testing.T) {
	decider := &AlwaysDisaggregate{}
	req := &Request{ID: "req-1", InputTokens: make([]int, 100)}

	decision := decider.Decide(req)

	if !decision.Disaggregate {
		t.Error("AlwaysDisaggregate should return Disaggregate=true")
	}
}

// TestDisaggregationDecider_Interface verifies both implementations satisfy the interface.
func TestDisaggregationDecider_Interface(t *testing.T) {
	var _ DisaggregationDecider = &NeverDisaggregate{}
	var _ DisaggregationDecider = &AlwaysDisaggregate{}
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
			got := d.Decide(req).Disaggregate
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
		{"direct-to-decode"},
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
		{"direct-to-decode", true},
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

	deciders := []DisaggregationDecider{
		&NeverDisaggregate{},
		&AlwaysDisaggregate{},
		NewPrefixThresholdDecider(512, 16),
	}
	for _, d := range deciders {
		d1 := d.Decide(req1)
		d2 := d.Decide(req2)
		if d1 != d2 {
			t.Errorf("%T: decision differs with different OutputTokens (d1=%v, d2=%v); INV-9 violation",
				d, d1, d2)
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
	NewPrefixThresholdDecider(-1, 16)
}

// TestNewPrefixThresholdDecider_PanicsOnZeroBlockSize verifies constructor validates blockSize.
func TestNewPrefixThresholdDecider_PanicsOnZeroBlockSize(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for zero blockSize")
		}
	}()
	NewPrefixThresholdDecider(512, 0)
}

// noopDisaggregationObserver is a no-op DisaggregationObserver used in tests to satisfy R13:
// DisaggregationObserver must work for >=2 backends (not tightly coupled to PrefixThresholdDecider).
// Any future stateful decider (e.g., adaptive-rate, popularity-based) should also implement it.
type noopDisaggregationObserver struct{}

func (*noopDisaggregationObserver) ObserveRouting(_ *Request, _ string) {}

// TestPrefixThresholdDecider_Interface verifies PrefixThresholdDecider satisfies both interfaces.
// Also verifies DisaggregationObserver is a general extension point (R13: works for >=2 backends).
func TestPrefixThresholdDecider_Interface(t *testing.T) {
	var _ DisaggregationDecider = &PrefixThresholdDecider{}
	var _ DisaggregationObserver = &PrefixThresholdDecider{}
	var _ DisaggregationObserver = &noopDisaggregationObserver{} // R13: second backend
}

// TestPrefixThresholdDecider_EmptyTokens verifies BC-PD-20: empty input returns Disaggregate=false.
func TestPrefixThresholdDecider_EmptyTokens(t *testing.T) {
	decider := NewPrefixThresholdDecider(512, 16)
	req := &Request{ID: "req-empty", InputTokens: []int{}}

	decision := decider.Decide(req)

	if decision.Disaggregate {
		t.Error("BC-PD-20: empty InputTokens must return Disaggregate=false")
	}
}

// TestPrefixThresholdDecider_AboveThreshold verifies BC-PD-21: non-cached tokens > threshold -> true.
// Uses threshold+1 as the boundary case to pin the strict > semantics (not >=).
func TestPrefixThresholdDecider_AboveThreshold(t *testing.T) {
	const blockSize = 16
	const threshold = 512
	decider := NewPrefixThresholdDecider(threshold, blockSize)

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
				tokens[i] = i + 1 // unique tokens, no cache hit
			}
			req := &Request{ID: fmt.Sprintf("req-%s", tc.name), InputTokens: tokens}

			decision := decider.Decide(req)

			if !decision.Disaggregate {
				t.Errorf("BC-PD-21: %d uncached tokens with threshold=%d should return Disaggregate=true",
					tc.tokens, threshold)
			}
		})
	}
}

// TestPrefixThresholdDecider_BelowOrAtThreshold verifies BC-PD-22: non-cached <= threshold -> false.
func TestPrefixThresholdDecider_BelowOrAtThreshold(t *testing.T) {
	const blockSize = 16
	const threshold = 512
	decider := NewPrefixThresholdDecider(threshold, blockSize)

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
				tokens[i] = i + 1000 // unique tokens, no cache hit
			}
			req := &Request{ID: fmt.Sprintf("req-%s", tc.name), InputTokens: tokens}

			decision := decider.Decide(req)

			if decision.Disaggregate {
				t.Errorf("BC-PD-22: %d non-cached tokens with threshold=%d should return Disaggregate=false",
					tc.tokens, threshold)
			}
		})
	}
}

// TestPrefixThresholdDecider_CacheAware verifies BC-PD-24: cached prefix reduces non-cached count.
// Scenario: warm cache has N blocks cached. New request with same prefix + additional tokens.
// Non-cached tokens = total_tokens - cached_blocks * blockSize.
func TestPrefixThresholdDecider_CacheAware(t *testing.T) {
	const blockSize = 16
	const threshold = 512

	decider := NewPrefixThresholdDecider(threshold, blockSize)

	// Warm cache: record 40 blocks (640 tokens) for a known prefix.
	// Use consecutive tokens 1..640 as the shared prefix.
	prefix := make([]int, 640) // 40 blocks
	for i := range prefix {
		prefix[i] = i + 1
	}
	prefixReq := &Request{ID: "req-warm", InputTokens: prefix}
	// Calling Decide records cachedHashes, then ObserveRouting warms the cache.
	decider.Decide(prefixReq)
	decider.ObserveRouting(prefixReq, "instance_0")

	// New request: same 640-token prefix + 200 new tokens = 840 tokens total.
	// Cached: 40 blocks * 16 = 640 tokens. Non-cached: 840 - 640 = 200 tokens.
	// 200 <= 512 threshold -> should NOT disaggregate.
	extended := make([]int, len(prefix)+200)
	copy(extended, prefix)
	for i := len(prefix); i < len(extended); i++ {
		extended[i] = 10000 + i // unique suffix tokens
	}
	req := &Request{ID: "req-cached", InputTokens: extended}

	decision := decider.Decide(req)

	if decision.Disaggregate {
		t.Errorf("BC-PD-24: with 640 tokens cached (40 blocks), 200 non-cached tokens should not disaggregate (threshold=%d)", threshold)
	}
}

// TestPrefixThresholdDecider_CacheAware_SubRequestIDMismatch verifies BC-PD-24 via the
// disaggregated pipeline path: ObserveRouting is called with a sub-request whose ID
// differs from the parent request passed to Decide (e.g., "<parent>_prefill" vs "<parent>").
// ObserveRouting must recompute hashes and still correctly populate the cache.
func TestPrefixThresholdDecider_CacheAware_SubRequestIDMismatch(t *testing.T) {
	const blockSize = 16
	const threshold = 512
	decider := NewPrefixThresholdDecider(threshold, blockSize)

	// Shared prefix: 40 blocks (640 tokens), unique token values.
	prefix := make([]int, 640)
	for i := range prefix {
		prefix[i] = i + 1
	}

	// Step 1: Decide is called with the parent request.
	parentReq := &Request{ID: "req-parent", InputTokens: prefix}
	decider.Decide(parentReq)

	// Step 2: Overwrite cachedHashes with a stale unrelated request so that ObserveRouting
	// MUST recompute when it sees the mismatched sub-request ID. Without this step, the
	// original test cannot detect an inversion of the != check: parentReq and subReq share
	// the same token content, so stale hashes from parentReq would produce the same result.
	staleReq := &Request{ID: "req-stale", InputTokens: []int{99999, 99998}} // < blockSize, 0 hashes
	decider.Decide(staleReq) // overwrites cachedHashes (empty) and cachedReqID = "req-stale"

	// Step 3: ObserveRouting is called with a sub-request (different ID from stale, same
	// InputTokens as prefix). ID mismatch must trigger recompute — without it the empty stale
	// hashes would be recorded and the follow-up request would incorrectly disaggregate.
	subReq := &Request{ID: "req-parent_prefill", InputTokens: prefix}
	decider.ObserveRouting(subReq, "prefill_0") // ID mismatch → must recompute + record

	// Step 4: New request with same prefix + 200 unique tokens = 840 total.
	// Cached: 40 blocks * 16 = 640 tokens. Non-cached: 200 <= 512 → should NOT disaggregate.
	extended := make([]int, len(prefix)+200)
	copy(extended, prefix)
	for i := len(prefix); i < len(extended); i++ {
		extended[i] = 10000 + i
	}
	req := &Request{ID: "req-follow", InputTokens: extended}

	decision := decider.Decide(req)

	if decision.Disaggregate {
		t.Errorf("CacheAware/SubRequestIDMismatch: ObserveRouting with mismatched ID should still "+
			"populate the cache; 200 non-cached tokens should not disaggregate (threshold=%d)", threshold)
	}
}

// TestPrefixThresholdDecider_ZeroThreshold verifies threshold=0 means always disaggregate
// when tokens are non-empty (any non-cached token > 0).
func TestPrefixThresholdDecider_ZeroThreshold(t *testing.T) {
	const blockSize = 16
	decider := NewPrefixThresholdDecider(0, blockSize)

	// 100 uncached tokens -> 100 > 0 -> disaggregate
	tokens := make([]int, 100)
	for i := range tokens {
		tokens[i] = i + 1
	}
	req := &Request{ID: "req-zero-thresh", InputTokens: tokens}

	decision := decider.Decide(req)

	if !decision.Disaggregate {
		t.Error("threshold=0 with non-empty uncached tokens should return Disaggregate=true")
	}
}
