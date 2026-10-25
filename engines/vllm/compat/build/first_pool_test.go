// Original to blis-vllm.

package build

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

func openFirst(t *testing.T, scenario string) (*kernelmodel.Model, kernelmodel.Settings) {
	t.Helper()
	m, err := kernelmodel.Open(scenario, vendoredRepos(t))
	if err != nil {
		t.Fatal(err)
	}
	s, err := m.Settings()
	if err != nil {
		t.Fatal(err)
	}
	return m, s
}

// The first pool's engine is the kernel's resolution of it, field for field, with the window
// check and the sequence-state note the CLI applied before the move.
func TestFirstPoolSettingsOf_IsTheKernelsResolution(t *testing.T) {
	for _, scenario := range []string{"glm-5-h200-tp8-mtp3.yaml", "deepseek-v4.1-flash-h100-tp8-dspark5.yaml"} {
		_, s := openFirst(t, scenario)
		fp, err := FirstPoolSettingsOf(scenario, s)
		if err != nil {
			t.Fatalf("%s: %v", scenario, err)
		}
		want := FirstPoolSettings{
			TotalKVBlocks: s.KVBlocks, SequenceState: s.SequenceState, BlockSize: int64(s.BlockSize),
			MaxNumSeqs: int64(s.MaxNumSeqs), MaxNumBatchedTokens: int64(s.MaxNumBatchedTokens),
			MaxModelLen: int64(s.MaxModelLen), MaxModelLenFrom: s.MaxModelLenFrom,
			PrefixCachingDisabled: s.PrefixCachingDisabled,
			SpeculativeTokens:     s.SpeculativeTokens, SpeculativeMethod: s.SpeculativeMethod,
		}
		if fp.TotalKVBlocks != want.TotalKVBlocks || fp.BlockSize != want.BlockSize || fp.MaxNumSeqs != want.MaxNumSeqs ||
			fp.MaxNumBatchedTokens != want.MaxNumBatchedTokens || fp.MaxModelLen != want.MaxModelLen ||
			fp.MaxModelLenFrom != want.MaxModelLenFrom || fp.PrefixCachingDisabled != want.PrefixCachingDisabled ||
			fp.SpeculativeTokens != want.SpeculativeTokens || fp.SpeculativeMethod != want.SpeculativeMethod {
			t.Errorf("%s: %+v, want the kernel's settings %+v", scenario, fp, want)
		}
		if (fp.SequenceState.Hold == nil) != (s.SequenceState.Hold == nil) || (fp.SequenceState.Gate == nil) != (s.SequenceState.Gate == nil) {
			t.Errorf("%s: the sequence state was not carried", scenario)
		}
		if note := SequenceFixedCharge("first pool", s); note != "" && (len(fp.Notes) != 1 || fp.Notes[0] != note) {
			t.Errorf("%s: notes %q, want the sequence charge %q", scenario, fp.Notes, note)
		}
		if fp.SpeculativeTokens == 0 || fp.SpeculativeMethod == "" {
			t.Errorf("%s: non-vacuity: the fixture drafts, so its speculative settings must be carried", scenario)
		}
	}
	// The DeepSeek-V4.1 fixture holds a sequence state, so its note and state are not vacuous.
	_, s := openFirst(t, "deepseek-v4.1-flash-h100-tp8-dspark5.yaml")
	if s.SequenceState.Hold == nil {
		t.Fatal("non-vacuity: the DeepSeek-V4.1 fixture holds no sequence state")
	}
	if fp, _ := FirstPoolSettingsOf("x.yaml", s); len(fp.Notes) != 1 || fp.SequenceState.Hold == nil {
		t.Errorf("a pool with a sequence state: notes %q, state %+v; want its charge note and state", fp.Notes, fp.SequenceState)
	}

	// No window at all is refused, naming the pool.
	s.MaxModelLen = 0
	if _, err := FirstPoolSettingsOf("x.yaml", s); err == nil || !strings.Contains(err.Error(), "first pool") {
		t.Errorf("no window: %v, want a refusal naming the first pool", err)
	}
}

// The LoRA reservation shrinks the first pool's KV budget, as the kernel sizes it.
func TestKVBlocksReserving_ShrinksTheBudget(t *testing.T) {
	m, s := openFirst(t, "glm-5-h200-tp8-mtp3.yaml")
	full, err := KVBlocksReserving("glm-5-h200-tp8-mtp3.yaml", m, 0)
	if err != nil || full != s.KVBlocks {
		t.Fatalf("no reservation: %d, %v; want the kernel's %d", full, err, s.KVBlocks)
	}
	less, err := KVBlocksReserving("glm-5-h200-tp8-mtp3.yaml", m, 8<<30)
	if err != nil || less >= full || less <= 0 {
		t.Errorf("an 8 GiB reservation: %d blocks (%v), want fewer than %d", less, err, full)
	}
	if _, err := KVBlocksReserving("glm-5-h200-tp8-mtp3.yaml", m, 1<<50); err == nil ||
		!strings.Contains(err.Error(), "LoRA adapter reservation") {
		t.Errorf("a reservation larger than the device: %v, want a refusal naming the reservation", err)
	}
}

// The prefix-hit resolution is the kernel model's, and a layout the engine does not map comes
// back with the warning the caller logs.
func TestPrefixHitOf_IsTheModelsResolution(t *testing.T) {
	m, _ := openFirst(t, "deepseek-v4.1-flash-h100-tp8-dspark5.yaml")
	on, off := true, false
	for _, override := range []*bool{nil, &on, &off} {
		r, warn := PrefixHitOf("first pool", m, override)
		want := m.PrefixHit(override)
		if r.Hit != want.Hit || r.Runner != want.Runner || r.Note != want.Note ||
			(r.BoundedReplay == nil) != (want.BoundedReplay == nil) || r.BoundedReplayFrom != want.BoundedReplayFrom {
			t.Errorf("override %v: %+v, want the model's %+v", override, r, want)
		}
		if (warn == "") != (r.Note == "") || (warn != "" && !strings.Contains(warn, r.Note)) {
			t.Errorf("override %v: warning %q for note %q", override, warn, r.Note)
		}
	}
	if r, _ := PrefixHitOf("first pool", m, &off); r.BoundedReplay == nil || *r.BoundedReplay {
		t.Errorf("an explicit false override must reach the model: %+v", r.BoundedReplay)
	}
}

// An unmapped layout's note becomes the warning the CLI logged before the move, word for word;
// no note, no warning. (No vendored fixture has an unmapped layout, so this is checked directly.)
func TestPrefixHitWarning_IsTheCLIsWarning(t *testing.T) {
	if got := prefixHitWarning("first pool", ""); got != "" {
		t.Errorf("no note: %q, want no warning", got)
	}
	want := "prefix cache (decode pool): the hit is the full-attention prefix alone: the window caches differ in window or page"
	if got := prefixHitWarning("decode pool", "the window caches differ in window or page"); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
