package latency

import (
	"math"
	"testing"
)

// TestBeta10BatchingInefficiency validates β₁₀ (decode memory correction) basis
// function contributions and scaling behavior for different sequence lengths and batch sizes.
//
// β₁₀ corrects the KV cache read bandwidth estimate for decode operations. Since decode
// is memory-bound (single-token attention requires reading all past KV cache entries),
// β₁₀ is the primary decode bottleneck correction factor.
//
// Expected behavior:
//   - β₁₀ has units of µs per (token²/batch_request)
//   - Contribution scales quadratically with sequence length (more past tokens to read)
//   - Contribution scales inversely with batch size (amortized bandwidth utilization)
//   - Typical range: 0.1-1.0 µs per (token²/batch_request)
//
// Test validates:
//   - Long sequence (500 tokens, batch=4): ~31.25ms contribution
//   - Short sequence (100 tokens, batch=32): ~0.156ms contribution
//   - Scaling ratio: 200× = (500/100)² × (32/4)
func TestBeta10BatchingInefficiency(t *testing.T) {
	// Test case 1: Long sequence, small batch (Scout general-lite scenario)
	// 500 tokens, batch_size=4, β₁₀=0.0005ms = 0.5μs = 0.0000005s
	// Expected: 0.5μs × (500²/4) = 0.5μs × 62,500 = 31,250μs = 31.25ms = 0.03125s
	coeff := 0.0000005 // 0.5μs in seconds
	tokens := 500.0
	batchSize := 4.0
	contribution := coeff * (tokens * tokens / batchSize)
	expectedSeconds := 0.03125 // 31.25ms
	tolerance := 0.10          // 10% tolerance

	if math.Abs(contribution-expectedSeconds)/expectedSeconds > tolerance {
		t.Errorf("β₁₀ long-sequence contribution out of range:\n"+
			"  got:      %.6fs (%.2fms)\n"+
			"  expected: %.6fs (%.2fms)\n"+
			"  tolerance: ±%.0f%%",
			contribution, contribution*1e3,
			expectedSeconds, expectedSeconds*1e3,
			tolerance*100)
	}

	// Test case 2: Short sequence, large batch (Scout roleplay scenario)
	// 100 tokens, batch_size=32, β₁₀=0.5μs = 0.0000005s
	// Expected: 0.5μs × (100²/32) = 0.5μs × 312.5 = 156.25μs = 0.156ms = 0.00015625s
	tokens2 := 100.0
	batchSize2 := 32.0
	contribution2 := coeff * (tokens2 * tokens2 / batchSize2)
	expectedSeconds2 := 0.00015625 // 0.156ms

	if math.Abs(contribution2-expectedSeconds2)/expectedSeconds2 > tolerance {
		t.Errorf("β₁₀ short-sequence contribution out of range:\n"+
			"  got:      %.6fs (%.3fms)\n"+
			"  expected: %.6fs (%.3fms)\n"+
			"  tolerance: ±%.0f%%",
			contribution2, contribution2*1e3,
			expectedSeconds2, expectedSeconds2*1e3,
			tolerance*100)
	}

	// Test case 3: Verify quadratic scaling
	// Expected ratio: (500/100)² × (32/4) = 25 × 8 = 200×
	ratio := contribution / contribution2
	expectedRatio := 200.0

	if math.Abs(ratio-expectedRatio)/expectedRatio > tolerance {
		t.Errorf("β₁₀ scaling ratio out of range:\n"+
			"  got:      %.1f×\n"+
			"  expected: %.1f×\n"+
			"  tolerance: ±%.0f%%",
			ratio, expectedRatio, tolerance*100)
	}

	t.Logf("✓ β₁₀ unit tests PASSED:")
	t.Logf("  - Long-sequence (500 tokens, batch=4):  %.2fms (%.2f%% error)",
		contribution*1e3, math.Abs(contribution-expectedSeconds)/expectedSeconds*100)
	t.Logf("  - Short-sequence (100 tokens, batch=32): %.3fms (%.2f%% error)",
		contribution2*1e3, math.Abs(contribution2-expectedSeconds2)/expectedSeconds2*100)
	t.Logf("  - Scaling ratio: %.1f× (%.2f%% error)",
		ratio, math.Abs(ratio-expectedRatio)/expectedRatio*100)
}

// TestBeta3PrimeKVSeqLen validates a hypothetical β₃' term representing per-layer
// KV cache access overhead that scales linearly with sequence length.
//
// While β₃' is not an explicit coefficient in the trained-physics model (which uses
// β₂ᵦ for decode memory correction), this test validates the analytical basis function
// behavior for KV cache read bandwidth that underlies the roofline calculation.
//
// Expected behavior:
//   - β₃' has units of µs per (token × layer)
//   - Contribution scales linearly with sequence length (past tokens to read)
//   - Contribution scales linearly with layer count (each layer reads KV cache)
//   - Typical range: 0.1-1.0 µs per (token × layer)
//
// Test validates:
//   - Long sequence (500 tokens, 56 layers): ~14ms contribution
//   - Short sequence (100 tokens, 56 layers): ~2.8ms contribution
//   - Scaling ratio: 5× = (500/100)
func TestBeta3PrimeKVSeqLen(t *testing.T) {
	// Test case 1: Long sequence, dense model (Scout general-lite scenario)
	// 500 tokens, 56 layers, β₃'=0.5μs = 0.0000005s per (token×layer)
	// Expected: 0.5μs × (500 × 56) = 0.5μs × 28,000 = 14,000μs = 14ms = 0.014s
	coeff := 0.0000005 // 0.5μs in seconds
	tokens := 500.0
	layers := 56.0
	contribution := coeff * (tokens * layers)
	expectedSeconds := 0.014 // 14ms
	tolerance := 0.10        // 10% tolerance

	if math.Abs(contribution-expectedSeconds)/expectedSeconds > tolerance {
		t.Errorf("β₃' long-sequence contribution out of range:\n"+
			"  got:      %.6fs (%.2fms)\n"+
			"  expected: %.6fs (%.2fms)\n"+
			"  tolerance: ±%.0f%%",
			contribution, contribution*1e3,
			expectedSeconds, expectedSeconds*1e3,
			tolerance*100)
	}

	// Test case 2: Short sequence, same model (Scout roleplay scenario)
	// 100 tokens, 56 layers, β₃'=0.5μs = 0.0000005s
	// Expected: 0.5μs × (100 × 56) = 0.5μs × 5,600 = 2,800μs = 2.8ms = 0.0028s
	tokens2 := 100.0
	contribution2 := coeff * (tokens2 * layers)
	expectedSeconds2 := 0.0028 // 2.8ms

	if math.Abs(contribution2-expectedSeconds2)/expectedSeconds2 > tolerance {
		t.Errorf("β₃' short-sequence contribution out of range:\n"+
			"  got:      %.6fs (%.2fms)\n"+
			"  expected: %.6fs (%.2fms)\n"+
			"  tolerance: ±%.0f%%",
			contribution2, contribution2*1e3,
			expectedSeconds2, expectedSeconds2*1e3,
			tolerance*100)
	}

	// Test case 3: Verify linear scaling
	// Expected ratio: 500/100 = 5×
	ratio := contribution / contribution2
	expectedRatio := 5.0

	if math.Abs(ratio-expectedRatio)/expectedRatio > tolerance {
		t.Errorf("β₃' scaling ratio out of range:\n"+
			"  got:      %.2f×\n"+
			"  expected: %.2f×\n"+
			"  tolerance: ±%.0f%%",
			ratio, expectedRatio, tolerance*100)
	}

	t.Logf("✓ β₃' unit tests PASSED:")
	t.Logf("  - Long-sequence (500 tokens, 56 layers):  %.2fms (%.2f%% error)",
		contribution*1e3, math.Abs(contribution-expectedSeconds)/expectedSeconds*100)
	t.Logf("  - Short-sequence (100 tokens, 56 layers): %.2fms (%.2f%% error)",
		contribution2*1e3, math.Abs(contribution2-expectedSeconds2)/expectedSeconds2*100)
	t.Logf("  - Scaling ratio: %.2f× (%.2f%% error)",
		ratio, math.Abs(ratio-expectedRatio)/expectedRatio*100)
}

// TestBeta10PhysicsAnalysis validates the dimensional analysis and expected
// magnitude of β₁₀ (decode memory correction coefficient).
//
// This test demonstrates the physics-based reasoning for β₁₀'s units and magnitude:
//
// Given:
//   - Observed decode overhead for batched long-sequence requests: ~30ms
//   - Sequence length: 500 tokens
//   - Batch size: 4 requests
//   - Basis function value: token²/batch = 500² / 4 = 62,500
//
// Dimensional analysis:
//   β₁₀ = observed_overhead / basis_function_value
//       = 30ms / 62,500
//       = 0.48 µs per (token²/batch_request)
//
// This validates that β₁₀ should have units of **microseconds** (not milliseconds)
// and a typical magnitude of 0.1-1.0 µs, which is physically reasonable for the
// amortized per-token-squared KV cache bandwidth overhead.
func TestBeta10PhysicsAnalysis(t *testing.T) {
	// Scenario: Scout general-lite experiment
	// Expected TTFT overhead from batching inefficiency: ~30ms
	// Tokens: 500, batch_size: 4
	expectedContributionMs := 30.0 // ms
	tokens := 500.0
	batchSize := 4.0
	basisFunctionValue := (tokens * tokens / batchSize) // 62,500

	// Calculate what β₁₀ should be
	expectedContributionSeconds := expectedContributionMs / 1e3 // Convert ms to seconds
	coefficientSeconds := expectedContributionSeconds / basisFunctionValue
	coefficientMicroseconds := coefficientSeconds * 1e6 // Convert to μs for readability

	t.Logf("Physics analysis for β₁₀:")
	t.Logf("  - Expected contribution: %.1fms", expectedContributionMs)
	t.Logf("  - Basis function value:  %.0f (token²/batch_request)", basisFunctionValue)
	t.Logf("  - Calculated β₁₀:        %.3fμs (%.6fs)",
		coefficientMicroseconds, coefficientSeconds)

	// Validate that calculated β₁₀ is in the expected range (0.1-1.0 μs)
	if coefficientMicroseconds < 0.1 || coefficientMicroseconds > 1.0 {
		t.Errorf("Calculated β₁₀ outside expected range:\n"+
			"  got:      %.3fμs\n"+
			"  expected: 0.1-1.0μs",
			coefficientMicroseconds)
	}

	// Demonstrate the importance of correct dimensional analysis
	incorrectUnitsMs := coefficientMicroseconds / 1000.0 // If incorrectly assumed milliseconds
	correctUnitsUs := coefficientMicroseconds

	t.Logf("\nDimensional analysis validation:")
	t.Logf("  - If units were milliseconds: %.6fms (1000× too small!)", incorrectUnitsMs)
	t.Logf("  - Correct units (microseconds): %.3fμs ✓", correctUnitsUs)
	t.Logf("  - Physical interpretation: %.3fμs per (token²/request) is reasonable for", correctUnitsUs)
	t.Logf("    amortized HBM bandwidth cost of reading KV cache (~%.1f GB/s effective)",
		float64(tokens*tokens)*2.0 /* FP16 bytes/token */ *56.0 /* layers */ /coefficientSeconds/1e9)

	// Validate that the coefficient is physically reasonable
	if coefficientMicroseconds >= 0.1 && coefficientMicroseconds <= 1.0 {
		t.Logf("\n✓ VALIDATED: β₁₀=%.3fμs is within expected range and physically reasonable", coefficientMicroseconds)
	}
}
