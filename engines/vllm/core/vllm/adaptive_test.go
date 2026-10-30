package vllm

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"
)

// trimPricer is a VerifyingPricer that trims each drafting request of a batch to width(i, s)
// verified drafts, as the latency kernel does under DSpark adaptive verification; a nil width
// verifies every scheduled draft.
type trimPricer struct {
	FixedPricer
	width func(i int, s engine.Scheduled) int
	asked *int
}

func (p trimPricer) VerifiedDrafts(b engine.Batch) []int {
	if p.asked != nil {
		*p.asked++
	}
	if p.width == nil {
		return nil
	}
	out := make([]int, len(b.Reqs))
	for i, s := range b.Reqs {
		if s.Drafts > 0 {
			out[i] = p.width(i, s)
		}
	}
	return out
}

const adaptiveBlock = "use_v2_model_runner: true\nspeculative: {method: dspark, num_spec_tokens: 3, enable_adaptive_verification: true}\n"

func buildPriced(t testing.TB, extra string, p Pricer) func() (engine.Engine, error) {
	t.Helper()
	s := spec(t, extra)
	return func() (engine.Engine, error) {
		return Factory{Pricer: func(engine.Spec) (Pricer, error) { return p, nil }}.New(s)
	}
}

func adaptiveArrivals() []enginetest.Arrival {
	return arrivals(req("a", 40, 30), req("b", 70, 25), req("c", 20, 40), req("d", 90, 12), req("e", 33, 21))
}

// A pricer that trims nothing (VerifiedDrafts nil) runs exactly as one that cannot trim: flag
// off, or a kernel that verifies every draft, is byte-identical to the engine before adaptive
// verification, sync and async, accepting every draft so a changed cap would show.
func TestUntrimmedVerificationIsUnchanged(t *testing.T) {
	all := enginetest.Options{Accepted: func(_ engine.Batch, s engine.Scheduled) int { return s.Drafts }}
	for _, extra := range []string{adaptiveBlock, adaptiveBlock + "async_scheduling: false\n",
		"speculative: {method: eagle3, num_spec_tokens: 3}\n"} {
		asked := 0
		want := enginetest.TestEngine(t, buildPriced(t, extra, FixedPricer(100)), adaptiveArrivals(), all)
		got := enginetest.TestEngine(t, buildPriced(t, extra, trimPricer{FixedPricer: 100, asked: &asked}),
			adaptiveArrivals(), all)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%q: a pricer that trims nothing changed the run", extra)
		}
		if asked == 0 {
			t.Errorf("%q: the engine never asked the pricer for verified widths, so the case checks nothing", extra)
		}
	}
}

// Under adaptive verification no request is credited more accepted drafts than the verifier ran
// for it: the rejection sampler sees only the compacted logits, each request's cu_num_logits span
// holding the drafts the budget admitted (reallocate_drafts,
// vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:430-435; rejection_sampler_utils.py:
// 343-351@5df63a72). The simulator offers every scheduled draft as accepted, so each verifying
// request emits exactly one sampled token plus its verified width while output remains, and every
// contract law holds.
func TestAdaptiveVerificationCapsAcceptedAtTheVerifiedWidth(t *testing.T) {
	width := func(i int, s engine.Scheduled) int { return (i + s.Computed) % (s.Drafts + 1) }
	for _, extra := range []string{adaptiveBlock, adaptiveBlock + "async_scheduling: false\n"} {
		all := enginetest.Options{Accepted: func(_ engine.Batch, s engine.Scheduled) int { return s.Drafts }}
		tr := enginetest.TestEngine(t, buildPriced(t, extra, trimPricer{FixedPricer: 100, width: width}),
			adaptiveArrivals(), all)
		left := map[string]int{}
		for _, a := range adaptiveArrivals() {
			left[a.Req.ID] = len(a.Req.Output)
		}
		trimmed, full := 0, 0
		for _, st := range tr.Steps {
			for i, s := range st.Batch.Reqs {
				n := st.Result.Emitted[s.ID]
				if s.Drafts > 0 {
					w := width(i, s)
					if n > 1+w {
						t.Fatalf("%q: batch %d: %s emitted %d tokens, verified %d of %d drafts: credited past its verified width",
							extra, st.Batch.ID, s.ID, n, w, s.Drafts)
					}
					if n == 1+w && w < s.Drafts {
						trimmed++
					}
					if n == 1+w && w == s.Drafts {
						full++
					}
				}
				left[s.ID] -= n
			}
		}
		// Both a trimmed request and an untrimmed one emitted their whole verified width, so the
		// cap is the width itself, not the width less one or the scheduled drafts.
		if trimmed == 0 || full == 0 {
			t.Errorf("%q: %d trimmed and %d untrimmed verifies emitted their verified width; want both", extra, trimmed, full)
		}
		for id, n := range left {
			if n != 0 {
				t.Errorf("%q: %s has %d output tokens unemitted", extra, id, n)
			}
		}
	}
}

// A pricer answering for the wrong number of requests is the engine's bug to report, not to
// price around.
func TestVerifiedWidthsMustMatchTheBatch(t *testing.T) {
	short := shortPricer{FixedPricer: 100}
	_, err := enginetest.Drive(buildPriced(t, adaptiveBlock, short), adaptiveArrivals(),
		enginetest.Options{Accepted: func(_ engine.Batch, s engine.Scheduled) int { return s.Drafts }})
	if err == nil {
		t.Error("a pricer giving fewer verified widths than requests was accepted")
	}
}

type shortPricer struct{ FixedPricer }

func (shortPricer) VerifiedDrafts(b engine.Batch) []int {
	for _, s := range b.Reqs {
		if s.Drafts > 0 {
			return []int{}
		}
	}
	return nil
}
