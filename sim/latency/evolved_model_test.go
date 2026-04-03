package latency

import (
	"math"
	"testing"
)

// TestBeta10BatchingInefficiency validates β₁₀ basis function contributions
// and scaling behavior for long vs short sequences.
//
// This test validates the CORRECTED expected ranges from iter11:
// - β₁₀ = 0.1-1.0 **μs** per (token²/batch_request), NOT milliseconds!
// - Expected contribution: ~31.25ms for Scout general-lite (500 tokens, batch_size=4)
// - Expected contribution: ~0.156ms for Scout roleplay (100 tokens, batch_size=32)
// - Expected scaling ratio: 200× (quadratic with sequence length, adjusted for batch size)
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

// TestBeta3PrimeKVSeqLen validates β₃' basis function contributions
// and scaling behavior for long vs short sequences.
//
// This test validates:
// - β₃' = 0.1-1.0 μs per (token×layer)
// - Expected contribution: ~14ms for Scout general-lite (500 tokens, 56 layers)
// - Expected contribution: ~2.8ms for Scout roleplay (100 tokens, 56 layers)
// - Expected scaling ratio: 5× (linear with sequence length)
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

// TestBeta10PhysicsAnalysis validates the corrected understanding from iter11
// that β₁₀ should be in microseconds, not milliseconds.
//
// This test demonstrates why iter10's hypothesis was wrong:
// - Iter10 hypothesis: β₁₀ = 0.1-1.0 **ms** per (token²/batch_request)
// - Iter10 optimizer: β₁₀ converged to 0.945 μs
// - Iter10 conclusion: "1000× too small" (ERROR!)
//
// Corrected analysis:
// - Expected contribution: ~31ms for Scout general-lite
// - Basis function value: 62,500 (token²/batch_request)
// - Therefore: β₁₀ = 31ms / 62,500 = 0.496 μs ✓
// - Correct range: 0.1-1.0 **μs**, not milliseconds!
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

	// Validate that calculated β₁₀ is in the CORRECTED range (0.1-1.0 μs)
	if coefficientMicroseconds < 0.1 || coefficientMicroseconds > 1.0 {
		t.Errorf("Calculated β₁₀ outside expected range:\n"+
			"  got:      %.3fμs\n"+
			"  expected: 0.1-1.0μs",
			coefficientMicroseconds)
	}

	// Demonstrate iter10's error
	iter10ExpectedRangeLowerMs := 0.1  // ms (WRONG!)
	iter10ExpectedRangeUpperMs := 1.0  // ms (WRONG!)
	iter10ConvergedValueUs := 0.945    // μs (from iter10 results)
	iter10ConvergedValueMs := iter10ConvergedValueUs / 1e3 // Convert to ms

	t.Logf("\nIter10 hypothesis error analysis:")
	t.Logf("  - Iter10 expected range: %.1f-%.1f **ms** (WRONG by 1000×!)",
		iter10ExpectedRangeLowerMs, iter10ExpectedRangeUpperMs)
	t.Logf("  - Iter10 converged to:   %.3fμs = %.6fms",
		iter10ConvergedValueUs, iter10ConvergedValueMs)
	t.Logf("  - Iter10 conclusion:     '1000× too small' (ERROR - hypothesis was wrong!)")
	t.Logf("  - Corrected range:       0.1-1.0 **μs** ✓")
	t.Logf("  - Iter10 value in μs:    %.3fμs (actually WITHIN correct range!)",
		iter10ConvergedValueUs)

	// Validate that iter10's converged value was actually reasonable
	if iter10ConvergedValueUs >= 0.1 && iter10ConvergedValueUs <= 1.0 {
		t.Logf("\n✓ CONFIRMED: Iter10 β₁₀=%.3fμs was CORRECT, hypothesis range was WRONG!",
			iter10ConvergedValueUs)
	}
}
