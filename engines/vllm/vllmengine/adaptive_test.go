package vllmengine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"

	"github.com/inference-sim/blis-vllm/core/vllm"
)

// adaptiveScenario is the Muse Spark recipe's default deployment: GB200, TP8 x EP, DSpark K=7
// with adaptive verification on, at the scenario's calibrated acceptance (AL 2.84).
const adaptiveScenario = "muse-gb200-nvl-tep8-dspark-adaptive.yaml"

// withoutAdaptive is the same scenario with enable_adaptive_verification dropped, written beside
// the test so the two differ in that line alone.
func withoutAdaptive(t *testing.T) map[string]string {
	t.Helper()
	r := roots(t, adaptiveScenario)
	raw, err := os.ReadFile(r[RootScenario])
	if err != nil {
		t.Fatal(err)
	}
	const flag = "        enable_adaptive_verification: true\n"
	if !strings.Contains(string(raw), flag) {
		t.Fatalf("%s states no enable_adaptive_verification", adaptiveScenario)
	}
	path := filepath.Join(t.TempDir(), adaptiveScenario)
	if err := os.WriteFile(path, []byte(strings.Replace(string(raw), flag, "", 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	r[RootScenario] = path
	return r
}

// A decode batch of n requests at ctx tokens of context, each verifying all K drafts, as vLLM's
// scheduler still budgets them under adaptive verification.
func decodeBatch(n, ctx, k int) engine.Batch {
	var b engine.Batch
	for i := 0; i < n; i++ {
		b.Reqs = append(b.Reqs, engine.Scheduled{ID: string(rune('a' + i)), Tokens: 1 + k, Computed: ctx,
			PromptLen: ctx - 64, Drafts: k})
	}
	return b
}

// The recipe default prices a K=7 decode below the same deployment without adaptive
// verification, and says which drafts it verified: the verify forward runs the trimmed width
// (blis-latency-kernel#89; vllm/v1/attention/selector.py:143-152 and
// vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:282-350@5df63a72). Without the flag
// nothing is trimmed (VerifiedDrafts nil).
func TestAdaptiveVerificationPricesTheTrimmedVerify(t *testing.T) {
	spec := func(r map[string]string) engine.Spec {
		return engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: r}
	}
	on, err := pricerFor(spec(roots(t, adaptiveScenario)))
	if err != nil {
		t.Fatal(err)
	}
	off, err := pricerFor(spec(withoutAdaptive(t)))
	if err != nil {
		t.Fatal(err)
	}
	vOn, vOff := on.(vllm.VerifyingPricer), off.(vllm.VerifyingPricer)
	trimmedAny := false
	for _, n := range []int{1, 16, 64} {
		b := decodeBatch(n, 8192, 7)
		if w := vOff.VerifiedDrafts(b); w != nil {
			t.Errorf("%d decodes, flag off: verified widths %v, want nil", n, w)
		}
		w := vOn.VerifiedDrafts(b)
		if len(w) != n {
			t.Fatalf("%d decodes, flag on: %d verified widths", n, len(w))
		}
		total := 0
		for _, x := range w {
			if x < 0 || x > 7 {
				t.Errorf("%d decodes: a verified width %d outside [0, 7]", n, x)
			}
			total += x
		}
		pOn, pOff := on.StepTime(b), off.StepTime(b)
		t.Logf("%d decodes: %d of %d drafts verified; step %dus adaptive, %dus fixed", n, total, 7*n, pOn, pOff)
		if total == 7*n {
			continue
		}
		trimmedAny = true
		if pOn >= pOff {
			t.Errorf("%d decodes: trimmed to %d of %d drafts, but priced %dus, not below the untrimmed %dus",
				n, total, 7*n, pOn, pOff)
		}
	}
	if !trimmedAny {
		// The kernel's adaptive budget (blis-latency-kernel#89) has not landed: it verifies every
		// draft at the recipe's acceptance, so there is no trimmed price to compare.
		t.Skip("the kernel verifies every scheduled draft under adaptive verification: no trimmed budget to price yet")
	}
}

// The recipe default runs end to end through the production factory, keeping every law of the
// engine contract, and no request is credited more accepted drafts than the kernel verified for
// it, though the simulator offers every scheduled draft.
func TestAdaptiveVerificationRecipeRunsEndToEnd(t *testing.T) {
	Register()
	spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, adaptiveScenario)}
	p, err := pricerFor(spec)
	if err != nil {
		t.Fatal(err)
	}
	vp := p.(vllm.VerifyingPricer)
	arr := []enginetest.Arrival{{At: 0, Req: req("a", 4096, 48)}, {At: 2000, Req: req("b", 8192, 40)},
		{At: 4000, Req: req("c", 2048, 64)}}
	tr := enginetest.TestEngine(t, func() (engine.Engine, error) { return engine.New(spec) }, arr,
		enginetest.Options{Accepted: func(_ engine.Batch, s engine.Scheduled) int { return s.Drafts }})
	verifies := 0
	for _, st := range tr.Steps {
		w := vp.VerifiedDrafts(st.Batch)
		for i, s := range st.Batch.Reqs {
			if s.Drafts == 0 {
				continue
			}
			verifies++
			if w == nil {
				t.Fatalf("batch %d: drafts verified with no verified widths", st.Batch.ID)
			}
			if n := st.Result.Emitted[s.ID]; n > 1+w[i] {
				t.Fatalf("batch %d: %s emitted %d tokens with %d of %d drafts verified", st.Batch.ID, s.ID, n, w[i], s.Drafts)
			}
		}
	}
	if verifies == 0 {
		t.Error("the run verified no drafts, so the cap was never exercised")
	}
}
