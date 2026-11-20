package runconfig

import "testing"

// X6: byte-identical output across platforms needs the product rounded before the sum, since
// the Go compiler fuses x*y + z on arm64 only (inference-sim#2078). On amd64 this cannot fail;
// on arm64 it fails if the product is fused.
func TestEffectiveTokensPerStepRoundsTheProductFirst(t *testing.T) {
	seed := uint64(88172645463325252)
	for i := 0; i < 500; i++ {
		seed ^= seed << 13
		seed ^= seed >> 7
		seed ^= seed << 17
		a := float64(seed%1_000_003) / 1_000_003
		k := int(seed>>20)%7 + 1
		prod := a * float64(k)
		want := 1.0 + float64(prod)
		if got := (SpeculativeConfig{Acceptance: a, K: k}).EffectiveTokensPerStep(); got != want {
			t.Fatalf("acceptance %.17g K %d: %.17g, want %.17g", a, k, got, want)
		}
	}
}
