package latency

import "math"

// activatedExpertFraction returns the expected fraction of a MoE layer's routed
// experts that at least one token in the step routes to — the coupon-collector
// occupancy expectation shared by the roofline and trained-physics backends
// (#764/#790 introduced it for roofline; #1849 ported it to trained-physics so the
// codebase carries ONE MoE activation convention, not two divergent ones).
//
// Derivation. Model the step as B tokens, each independently routing to a uniformly
// random size-k subset of the N routed experts (the standard uniform top-k
// assumption). vLLM's fused-MoE kernel loads an expert's weights from HBM iff at
// least one token in the step routes to it, so the quantity that drives weight
// bandwidth is the expected number of DISTINCT experts activated.
//
// For a fixed expert e, a single token skips e with probability (N−k)/N, and tokens
// route independently, so P(e not activated by any of B tokens) = ((N−k)/N)^B. With
// X_e the indicator that e is activated, linearity of expectation (which needs no
// independence BETWEEN experts, only the per-expert probability) gives
//
//	nEff(B) = E[distinct experts] = Σ_e E[X_e] = N · (1 − ((N−k)/N)^B)
//
// and this function returns nEff(B)/N. Limiting cases:
//
//   - B·k ≪ N: (1−k/N)^B ≈ 1 − Bk/N, so nEff ≈ B·k — every routed slot hits a fresh
//     expert (the collision-free linear regime).
//   - B = 1: exactly k/N — a single token streams only the k experts it routes to.
//   - B → ∞: → 1 — every resident expert is streamed (saturation).
//
// It is monotonically non-decreasing in B and strictly refines the older
// min(N, max(k, B·k)) linear-then-clamp form, which overestimates through the
// transition region.
//
// PESSIMISM: uniform routing MAXIMISES the distinct-expert count. Real routers are
// skewed, so tokens re-hit the same experts and the true fraction is ≤ this one,
// making the result a conservative upper bound on weight bandwidth. See #789 for the
// non-uniform-routing refinement (it applies to both backends).
//
// Degenerate inputs are resolved here rather than at the call sites:
//
//   - tokens ≤ 0 — no tokens in the step, so no expert is activated: 0. (The
//     distinct-expert expectation of an empty batch is genuinely zero; every other
//     token-population term in a zero-token step is likewise zero.)
//   - numExperts ≤ 0 — dense, or a malformed config with no routed experts: 1, which
//     is inert because the resident expert count it scales is itself 0.
//   - kEff ≥ numExperts — every token routes to every expert: 1. Returning early also
//     keeps math.Pow off a negative base, which would yield NaN for kEff > numExperts.
func activatedExpertFraction(numExperts, kEff int, tokens float64) float64 {
	if tokens <= 0 {
		return 0
	}
	if numExperts <= 0 || kEff >= numExperts {
		return 1
	}
	n := float64(numExperts)
	k := float64(kEff)
	return 1.0 - math.Pow((n-k)/n, tokens)
}
