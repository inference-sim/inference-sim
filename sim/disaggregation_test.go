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

// TestNewDisaggregationDecider_Factory verifies factory dispatches correctly.
func TestNewDisaggregationDecider_Factory(t *testing.T) {
	tests := []struct {
		name     string
		wantType string
	}{
		{"", "*sim.NeverDisaggregate"},
		{"never", "*sim.NeverDisaggregate"},
		{"always", "*sim.AlwaysDisaggregate"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			decider := NewDisaggregationDecider(tc.name)
			if decider == nil {
				t.Fatal("NewDisaggregationDecider returned nil")
			}
			gotType := fmt.Sprintf("%T", decider)
			if gotType != tc.wantType {
				t.Errorf("NewDisaggregationDecider(%q) type = %s, want %s", tc.name, gotType, tc.wantType)
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

// TestIsValidDisaggregationDecider verifies the validation function.
func TestIsValidDisaggregationDecider(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"", true},
		{"never", true},
		{"always", true},
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
	if len(names) < 3 {
		t.Errorf("expected at least 3 decider names, got %d", len(names))
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

// TestIsValidDisaggregationDecider_PrefixThreshold verifies prefix-threshold is recognized.
func TestIsValidDisaggregationDecider_PrefixThreshold(t *testing.T) {
	if !IsValidDisaggregationDecider("prefix-threshold") {
		t.Error("prefix-threshold should be a valid disaggregation decider")
	}
}

// TestNewDisaggregationDecider_PrefixThresholdPanics verifies factory panics for prefix-threshold.
// prefix-threshold requires parameters, so it must be constructed via NewPrefixThresholdDecider.
func TestNewDisaggregationDecider_PrefixThresholdPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when constructing prefix-threshold via factory (use NewPrefixThresholdDecider)")
		}
	}()
	NewDisaggregationDecider("prefix-threshold")
}

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

// TestPrefixThresholdDecider_AboveThreshold verifies BC-PD-21: non-cached tokens > threshold → true.
func TestPrefixThresholdDecider_AboveThreshold(t *testing.T) {
	const blockSize = 16
	const threshold = 512
	decider := NewPrefixThresholdDecider(threshold, blockSize)

	// Construct tokens with no cache history: all tokens are non-cached.
	// 600 tokens > 512 threshold → disaggregate.
	tokens := make([]int, 600)
	for i := range tokens {
		tokens[i] = i + 1
	}
	req := &Request{ID: "req-above", InputTokens: tokens}

	decision := decider.Decide(req)

	if !decision.Disaggregate {
		t.Errorf("BC-PD-21: 600 uncached tokens with threshold=%d should return Disaggregate=true", threshold)
	}
}

// TestPrefixThresholdDecider_BelowOrAtThreshold verifies BC-PD-22: non-cached <= threshold → false.
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
	// 200 <= 512 threshold → should NOT disaggregate.
	// (Using 600 new tokens would give 1240 total; non-cached = 1240 - 640 = 600 > 512
	// which would still disaggregate — 200 was chosen so the cache benefit is decisive.)
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

// --- DirectToDecodeDecider tests ---

func TestDirectToDecodeDecider_ShortPromptDoesNotDisaggregate(t *testing.T) {
	d := NewDirectToDecodeDecider(256)
	req := &Request{InputTokens: make([]int, 100)} // 100 < 256
	decision := d.Decide(req)
	if decision.Disaggregate {
		t.Error("short prompt (100 tokens < threshold 256) should not disaggregate")
	}
}

func TestDirectToDecodeDecider_LongPromptDisaggregates(t *testing.T) {
	d := NewDirectToDecodeDecider(256)
	req := &Request{InputTokens: make([]int, 500)} // 500 >= 256
	decision := d.Decide(req)
	if !decision.Disaggregate {
		t.Error("long prompt (500 tokens >= threshold 256) should disaggregate")
	}
}

func TestDirectToDecodeDecider_ExactThresholdDisaggregates(t *testing.T) {
	d := NewDirectToDecodeDecider(256)
	req := &Request{InputTokens: make([]int, 256)} // 256 >= 256
	decision := d.Decide(req)
	if !decision.Disaggregate {
		t.Error("exact threshold (256 tokens >= threshold 256) should disaggregate")
	}
}

func TestDirectToDecodeDecider_EmptyInputDoesNotDisaggregate(t *testing.T) {
	d := NewDirectToDecodeDecider(256)
	req := &Request{InputTokens: nil}
	decision := d.Decide(req)
	if decision.Disaggregate {
		t.Error("empty input should not disaggregate")
	}
}

func TestDirectToDecodeDecider_ZeroThresholdAlwaysDisaggregates(t *testing.T) {
	d := NewDirectToDecodeDecider(0)
	req := &Request{InputTokens: make([]int, 1)}
	decision := d.Decide(req)
	if !decision.Disaggregate {
		t.Error("threshold 0 with non-empty input should always disaggregate")
	}
}

func TestNewDirectToDecodeDecider_PanicsOnNegativeThreshold(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on negative threshold")
		}
	}()
	NewDirectToDecodeDecider(-1)
}

// TestPrefixThresholdDecider_ZeroThreshold verifies threshold=0 means always disaggregate
// when tokens are non-empty (any non-cached token > 0).
func TestPrefixThresholdDecider_ZeroThreshold(t *testing.T) {
	const blockSize = 16
	decider := NewPrefixThresholdDecider(0, blockSize)

	// 100 uncached tokens → 100 > 0 → disaggregate
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
