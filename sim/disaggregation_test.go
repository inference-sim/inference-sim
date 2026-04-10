package sim

import (
	"testing"
)

// TestNeverDisaggregate_AlwaysReturnsFalse verifies that NeverDisaggregate
// always returns Disaggregate=false regardless of input.
func TestNeverDisaggregate_AlwaysReturnsFalse(t *testing.T) {
	decider := &NeverDisaggregate{}
	req := &Request{ID: "req-1", InputTokens: make([]int, 100)}

	decision := decider.Decide(req, DecodeContext{InstanceID: "i0", CachedBlockCount: 0})

	if decision.Disaggregate {
		t.Error("NeverDisaggregate should return Disaggregate=false")
	}
}

// TestAlwaysDisaggregate_AlwaysReturnsTrue verifies that AlwaysDisaggregate
// always returns Disaggregate=true regardless of input.
func TestAlwaysDisaggregate_AlwaysReturnsTrue(t *testing.T) {
	decider := &AlwaysDisaggregate{}
	req := &Request{ID: "req-1", InputTokens: make([]int, 100)}

	decision := decider.Decide(req, DecodeContext{InstanceID: "i0", CachedBlockCount: 0})

	if !decision.Disaggregate {
		t.Error("AlwaysDisaggregate should return Disaggregate=true")
	}
}

// TestDisaggregationDecider_Interface verifies all implementations satisfy the interface.
func TestDisaggregationDecider_Interface(t *testing.T) {
	var _ DisaggregationDecider = &NeverDisaggregate{}
	var _ DisaggregationDecider = &AlwaysDisaggregate{}
	var _ DisaggregationDecider = &DirectToDecodeDecider{}
	var _ DisaggregationDecider = &PrefixThresholdDecider{}
}

// TestNewDisaggregationDecider_Factory verifies factory dispatches correctly
// by asserting observable behavior (Decide output), not concrete type names.
func TestNewDisaggregationDecider_Factory(t *testing.T) {
	req := &Request{ID: "req-1", InputTokens: make([]int, 10)}
	ctx := DecodeContext{InstanceID: "i0", CachedBlockCount: 0}
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
			got := d.Decide(req, ctx).Disaggregate
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
	ctx := DecodeContext{InstanceID: "i0", CachedBlockCount: 0}
	req1 := &Request{ID: "req-1", InputTokens: make([]int, 100), OutputTokens: nil}
	req2 := &Request{ID: "req-2", InputTokens: make([]int, 100), OutputTokens: make([]int, 9999)}

	deciders := []DisaggregationDecider{
		&NeverDisaggregate{},
		&AlwaysDisaggregate{},
		NewPrefixThresholdDecider(512, 16),
		NewDirectToDecodeDecider(256),
	}
	for _, d := range deciders {
		d1 := d.Decide(req1, ctx)
		d2 := d.Decide(req2, ctx)
		if d1 != d2 {
			t.Errorf("%T: decision differs with different OutputTokens (d1=%v, d2=%v); INV-9 violation",
				d, d1, d2)
		}
	}
}

// --- DirectToDecodeDecider tests ---

// TestDirectToDecodeDecider_ShortPromptDoesNotDisaggregate verifies that requests below
// the threshold are routed directly to decode (Disaggregate=false).
func TestDirectToDecodeDecider_ShortPromptDoesNotDisaggregate(t *testing.T) {
	d := NewDirectToDecodeDecider(256)
	req := &Request{InputTokens: make([]int, 100)} // 100 < 256
	decision := d.Decide(req, DecodeContext{InstanceID: "i0"})
	if decision.Disaggregate {
		t.Error("short prompt (100 tokens < threshold 256) should not disaggregate")
	}
}

// TestDirectToDecodeDecider_LongPromptDisaggregates verifies that requests above the threshold
// go through the full PD pipeline (Disaggregate=true).
func TestDirectToDecodeDecider_LongPromptDisaggregates(t *testing.T) {
	d := NewDirectToDecodeDecider(256)
	req := &Request{InputTokens: make([]int, 500)} // 500 >= 256
	decision := d.Decide(req, DecodeContext{InstanceID: "i0"})
	if !decision.Disaggregate {
		t.Error("long prompt (500 tokens >= threshold 256) should disaggregate")
	}
}

// TestDirectToDecodeDecider_ExactThresholdDisaggregates verifies the >= boundary:
// exactly threshold tokens must disaggregate (>= semantics, not >).
func TestDirectToDecodeDecider_ExactThresholdDisaggregates(t *testing.T) {
	d := NewDirectToDecodeDecider(256)
	req := &Request{InputTokens: make([]int, 256)} // 256 >= 256
	decision := d.Decide(req, DecodeContext{InstanceID: "i0"})
	if !decision.Disaggregate {
		t.Error("exact threshold (256 tokens >= threshold 256) should disaggregate")
	}
}

// TestDirectToDecodeDecider_BelowThresholdDoesNotDisaggregate verifies the >= boundary:
// threshold-1 tokens must not disaggregate.
func TestDirectToDecodeDecider_BelowThresholdDoesNotDisaggregate(t *testing.T) {
	const threshold = 256
	d := NewDirectToDecodeDecider(threshold)
	req := &Request{InputTokens: make([]int, threshold-1)} // 255 < 256
	decision := d.Decide(req, DecodeContext{InstanceID: "i0"})
	if decision.Disaggregate {
		t.Errorf("threshold-1 tokens (%d) must not disaggregate (threshold=%d, boundary is >=)",
			threshold-1, threshold)
	}
}

// TestDirectToDecodeDecider_EmptyInputDoesNotDisaggregate verifies that empty input tokens
// always return Disaggregate=false regardless of threshold.
func TestDirectToDecodeDecider_EmptyInputDoesNotDisaggregate(t *testing.T) {
	d := NewDirectToDecodeDecider(256)
	req := &Request{InputTokens: nil}
	decision := d.Decide(req, DecodeContext{InstanceID: "i0"})
	if decision.Disaggregate {
		t.Error("empty input should not disaggregate")
	}
}

// TestDirectToDecodeDecider_ZeroThresholdAlwaysDisaggregates verifies that threshold=0
// disaggregates all non-empty requests (0 <= any positive length).
func TestDirectToDecodeDecider_ZeroThresholdAlwaysDisaggregates(t *testing.T) {
	d := NewDirectToDecodeDecider(0)
	req := &Request{InputTokens: make([]int, 1)}
	decision := d.Decide(req, DecodeContext{InstanceID: "i0"})
	if !decision.Disaggregate {
		t.Error("threshold 0 with non-empty input should always disaggregate")
	}
}

// TestNewDirectToDecodeDecider_PanicsOnNegativeThreshold verifies constructor validates threshold (R3).
func TestNewDirectToDecodeDecider_PanicsOnNegativeThreshold(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on negative threshold")
		}
	}()
	NewDirectToDecodeDecider(-1)
}

// TestDirectToDecodeDecider_PanicsOnNilRequest verifies that Decide panics on nil req,
// providing a clear error message rather than a nil-pointer dereference.
func TestDirectToDecodeDecider_PanicsOnNilRequest(t *testing.T) {
	d := NewDirectToDecodeDecider(256)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when req is nil")
		}
	}()
	d.Decide(nil, DecodeContext{InstanceID: "i0"})
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

// TestPrefixThresholdDecider_EmptyTokens verifies BC-PD-20: empty input returns Disaggregate=false.
func TestPrefixThresholdDecider_EmptyTokens(t *testing.T) {
	decider := NewPrefixThresholdDecider(512, 16)
	req := &Request{ID: "req-empty", InputTokens: []int{}}
	ctx := DecodeContext{InstanceID: "i0", CachedBlockCount: 0}

	decision := decider.Decide(req, ctx)

	if decision.Disaggregate {
		t.Error("BC-PD-20: empty InputTokens must return Disaggregate=false")
	}
}

// TestPrefixThresholdDecider_DecodeContext_TableDriven verifies INV-PD-8: the decider
// uses ctx.CachedBlockCount from the selected decode instance to determine non-cached token count.
// ≥20 cases covering zero cached, partial cached, full cached, at-threshold, above, below, etc.
func TestPrefixThresholdDecider_DecodeContext_TableDriven(t *testing.T) {
	const blockSize = 16
	const threshold = 512 // in tokens

	tests := []struct {
		name             string
		inputLen         int
		cachedBlockCount int
		wantDisaggregate bool
	}{
		// Zero cached blocks — non-cached = all tokens
		{"zero_cached_below_threshold", 100, 0, false},     // 100 <= 512
		{"zero_cached_at_threshold", 512, 0, false},        // 512 == 512, not > threshold
		{"zero_cached_above_threshold", 513, 0, true},      // 513 > 512
		{"zero_cached_well_above", 1000, 0, true},          // 1000 > 512
		{"zero_cached_single_token", 1, 0, false},          // 1 <= 512
		{"zero_cached_empty", 0, 0, false},                 // empty input always false
		// Partial cached — non-cached = input - cached_blocks * blockSize
		{"partial_cached_under_threshold", 800, 20, false}, // 800 - 20*16=320 <= 512
		{"partial_cached_at_threshold", 824, 20, false},    // 824 - 320=504 <= 512
		{"partial_cached_above_threshold", 900, 20, true},  // 900 - 320=580 > 512
		{"partial_cached_one_block", 520, 1, false},        // 520 - 16=504 <= 512
		{"partial_cached_many_blocks", 1500, 60, true},     // 1500 - 960=540 > 512
		// Fully cached — non-cached = 0 (clamped)
		{"fully_cached_exact", 320, 20, false},             // 320 - 320=0 <= 512
		{"fully_cached_over_count", 160, 20, false},        // 160 - 320 < 0 → clamped 0
		{"fully_cached_large", 1000, 100, false},           // 1000 - 1600 < 0 → clamped 0
		// Zero threshold — any non-cached token triggers disagg
		{"zero_threshold_all_cached", 32, 2, false},        // threshold=0, cached=32, non-cached=0
		// (use separate decider for zero threshold tests below)
		// Boundary around blockSize grain
		{"one_block_cached_under", blockSize + threshold, 1, false}, // (16+512) - 16 = 512, not > 512
		{"one_block_cached_over", blockSize + threshold + 1, 1, true}, // 513 > 512
		// Large input, high cache count
		{"large_input_high_cache", 5000, 250, false}, // 5000 - 4000=1000 > 512 → wait, 1000 > 512 → true
		// (re-check): 5000 - 250*16 = 5000 - 4000 = 1000 > 512 → disaggregate=true
		{"large_input_very_high_cache", 5000, 300, false}, // 5000 - 4800=200 <= 512
		{"large_input_moderate_cache", 5000, 30, true},    // 5000 - 480=4520 > 512
	}

	// Fix the large_input_high_cache entry
	for i := range tests {
		if tests[i].name == "large_input_high_cache" {
			tests[i].wantDisaggregate = true // 5000 - 4000 = 1000 > 512
		}
	}

	decider := NewPrefixThresholdDecider(threshold, blockSize)

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tokens := make([]int, tc.inputLen)
			for i := range tokens {
				tokens[i] = i + 1
			}
			req := &Request{ID: "req-" + tc.name, InputTokens: tokens}
			ctx := DecodeContext{InstanceID: "instance_0", CachedBlockCount: tc.cachedBlockCount}

			got := decider.Decide(req, ctx)

			if got.Disaggregate != tc.wantDisaggregate {
				nonCached := tc.inputLen - tc.cachedBlockCount*blockSize
				if nonCached < 0 {
					nonCached = 0
				}
				t.Errorf("inputLen=%d cachedBlocks=%d nonCached=%d threshold=%d: Disaggregate=%v, want %v",
					tc.inputLen, tc.cachedBlockCount, nonCached, threshold, got.Disaggregate, tc.wantDisaggregate)
			}
		})
	}
}

// TestPrefixThresholdDecider_ZeroThreshold verifies threshold=0 means always disaggregate
// when tokens are non-empty and no cache coverage.
func TestPrefixThresholdDecider_ZeroThreshold(t *testing.T) {
	const blockSize = 16
	decider := NewPrefixThresholdDecider(0, blockSize)

	// 100 uncached tokens -> 100 > 0 -> disaggregate
	tokens := make([]int, 100)
	for i := range tokens {
		tokens[i] = i + 1
	}
	req := &Request{ID: "req-zero-thresh", InputTokens: tokens}
	ctx := DecodeContext{InstanceID: "i0", CachedBlockCount: 0}

	decision := decider.Decide(req, ctx)

	if !decision.Disaggregate {
		t.Error("threshold=0 with non-empty uncached tokens should return Disaggregate=true")
	}
}

// TestPrefixThresholdDecider_ZeroThreshold_AllCached verifies threshold=0 with all tokens
// cached returns Disaggregate=false (0 non-cached tokens, 0 is not > 0).
func TestPrefixThresholdDecider_ZeroThreshold_AllCached(t *testing.T) {
	const blockSize = 16
	decider := NewPrefixThresholdDecider(0, blockSize)

	// 32 tokens = 2 blocks, both cached
	tokens := make([]int, 32)
	req := &Request{InputTokens: tokens}
	ctx := DecodeContext{InstanceID: "i0", CachedBlockCount: 2}

	decision := decider.Decide(req, ctx)

	if decision.Disaggregate {
		t.Error("threshold=0 with all tokens cached (0 non-cached) should not disaggregate")
	}
}

// TestPrefixThresholdDecider_InstanceLocalBehavior verifies INV-PD-8: the same request
// with the same input produces different disaggregation decisions depending on which
// decode instance is selected (and thus ctx.CachedBlockCount varies).
func TestPrefixThresholdDecider_InstanceLocalBehavior(t *testing.T) {
	const blockSize = 16
	const threshold = 200 // tokens
	decider := NewPrefixThresholdDecider(threshold, blockSize)

	tokens := make([]int, 300) // 300 total tokens
	for i := range tokens {
		tokens[i] = i + 1
	}
	req := &Request{ID: "req-locality", InputTokens: tokens}

	// Instance A: 10 blocks cached = 160 tokens → non-cached = 300-160=140 <= 200 → skip
	ctxA := DecodeContext{InstanceID: "instance_0", CachedBlockCount: 10}
	decisionA := decider.Decide(req, ctxA)
	if decisionA.Disaggregate {
		t.Error("INV-PD-8: instance A has prefix cached → should skip disaggregation")
	}

	// Instance B: 0 blocks cached → non-cached = 300 > 200 → disaggregate
	ctxB := DecodeContext{InstanceID: "instance_1", CachedBlockCount: 0}
	decisionB := decider.Decide(req, ctxB)
	if !decisionB.Disaggregate {
		t.Error("INV-PD-8: instance B has no cache → should disaggregate")
	}
}

// TestPrefixThresholdDecider_AboveThreshold verifies BC-PD-21: non-cached tokens > threshold -> true.
func TestPrefixThresholdDecider_AboveThreshold(t *testing.T) {
	const blockSize = 16
	const threshold = 512
	decider := NewPrefixThresholdDecider(threshold, blockSize)

	tests := []struct {
		name   string
		tokens int
	}{
		{"threshold_plus_one", threshold + 1},
		{"well_above_threshold", 600},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tokens := make([]int, tc.tokens)
			for i := range tokens {
				tokens[i] = i + 1
			}
			req := &Request{ID: "req-" + tc.name, InputTokens: tokens}
			ctx := DecodeContext{InstanceID: "i0", CachedBlockCount: 0}

			decision := decider.Decide(req, ctx)

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
				tokens[i] = i + 1000
			}
			req := &Request{ID: "req-" + tc.name, InputTokens: tokens}
			ctx := DecodeContext{InstanceID: "i0", CachedBlockCount: 0}

			decision := decider.Decide(req, ctx)

			if decision.Disaggregate {
				t.Errorf("BC-PD-22: %d non-cached tokens with threshold=%d should return Disaggregate=false",
					tc.tokens, threshold)
			}
		})
	}
}
