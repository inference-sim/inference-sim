package harness

import (
	"os"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/tools/harness/internal/artifacts"
)

// The committed evidence carries its provenance -- the generator, the environment, every pinned
// revision with its roles and a sha256 for every file read -- and every point cites only
// revisions the header pins, at the run date and today. Every verified point's vectors are in
// the fixture, so what the scorer replays is, per the evidence, what the client sent.
func TestThePromptLengthEvidenceIsPinnedAndReplayed(t *testing.T) {
	e, err := LoadPromptLengthEvidence(DefaultPromptLengthsPath())
	if err != nil {
		t.Fatal(err)
	}
	if e.Generator != "scripts/inferencex_prompt_lengths.py" || e.Environment["numpy"] == "" ||
		e.Environment["transformers"] == "" {
		t.Errorf("provenance header: generator %q, environment %v", e.Generator, e.Environment)
	}
	for _, r := range e.Revisions {
		tok := false
		for f := range r.Files {
			tok = tok || strings.HasPrefix(f, "tokeniz") || f == "tiktoken.model" || f == "vocab.json"
		}
		if !tok {
			t.Errorf("%s@%s records no tokenizer file: %v", r.Repo, r.Revision, r.Files)
		}
	}
	ws, err := LoadBenchmarkWorkloads(DefaultWorkloadsPath(0))
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range e.Points {
		roles := map[string]bool{}
		for _, ev := range p.Evidence {
			if ev == nil {
				continue
			}
			for _, r := range e.Revisions {
				if r.Repo == p.Repo && r.Revision == ev.Revision {
					for _, role := range r.Roles {
						roles[strings.Fields(role)[0]] = true
					}
				}
			}
		}
		if !roles["today"] {
			t.Errorf("%s %s c=%d is not checked at today's revision", p.Scenario, p.Label, p.Concurrency)
		}
		if ok, _ := e.Verified(Sweep{Scenario: p.Scenario, Label: p.Label}, p.Concurrency); ok {
			if _, err := ws.At(p.ISL, p.OSL, p.Concurrency); err != nil {
				t.Errorf("verified point %s %s c=%d is not in the replayed fixture: %v", p.Scenario, p.Label, p.Concurrency, err)
			}
		}
	}
}

// A point the evidence verifies carries no prompt-length gap; one it does not hold, or holds
// with a mismatch, carries the gap that keeps it out of STRICT and MEASURED-OR-RESOLVED.
func TestAnUnverifiedPromptLengthIsAGap(t *testing.T) {
	zero, two := 0, 2
	e := &PromptLengthEvidence{byKey: map[string]*PromptLengthPoint{
		promptKey("s", "l", 4): {Repo: "r", Evidence: []*RevisionEvidence{{Revision: "a", ServerAddedTokens: []int{0}}},
			MaxAbsMismatch: &zero},
		promptKey("s", "l", 8): {Repo: "r", Evidence: []*RevisionEvidence{{Revision: "a", MaxAbsMismatch: 2,
			ServerAddedTokens: []int{0}}}, MaxAbsMismatch: &two},
		promptKey("s", "l", 16): {Repo: "r", Evidence: []*RevisionEvidence{{Revision: "a", ServerAddedTokens: []int{1}}},
			MaxAbsMismatch: &zero},
	}}
	sw := Sweep{Scenario: "s", Label: "l"}
	for conc, want := range map[int]bool{4: true, 8: false, 16: false, 32: false} {
		if ok, why := e.Verified(sw, conc); ok != want {
			t.Errorf("c=%d: verified %v (%s), want %v", conc, ok, why, want)
		}
	}

	committed := committedSweep("vllm")
	empty := &PromptLengthEvidence{path: "none.json", byKey: map[string]*PromptLengthPoint{}}
	cv := AssessCoverage(&Corpus{Sweeps: []Sweep{committed}}, Config{Repos: testRepos(), PromptLengths: empty})
	if !hasGap(cv.Gaps, OwnerSim, promptLengthCause) || cv.TierOf(committed) != TierApproximate {
		t.Errorf("an unverified point is not an approximate prompt-length gap: tier %v", cv.TierOf(committed))
	}
}

// Every MEASURED-OR-RESOLVED point's sent prompt lengths are its targets per the evidence -- the
// tier requires it -- so the scorer, which replays the targets, replays exactly what was sent.
func TestEveryMeasuredOrResolvedPointSentItsTarget(t *testing.T) {
	c, err := LoadCorpus(artifacts.Measurement(t, "aisimulate_e2e.json"))
	if err != nil {
		t.Fatal(err)
	}
	set, err := LoadEngineSettings(artifacts.Measurement(t, "inferencex_engine_settings.json"))
	if err != nil {
		t.Fatal(err)
	}
	e, err := LoadPromptLengthEvidence(DefaultPromptLengthsPath())
	if err != nil {
		t.Fatal(err)
	}
	cv := AssessCoverage(c, Config{Repos: testRepos(), EngineSettings: set, PromptLengths: e})
	n := 0
	for _, sw := range c.Sweeps {
		if sw.Framework != "vllm" || !cv.TierOf(sw).In(TierMeasuredOrResolved) {
			continue
		}
		for _, p := range sw.Points {
			n++
			if ok, why := e.Verified(sw, p.Concurrency); !ok {
				t.Errorf("%s %s c=%d is MEASURED-OR-RESOLVED but its sent lengths are not verified: %s",
					sw.Scenario, sw.Label, p.Concurrency, why)
			}
		}
	}
	if n == 0 {
		t.Fatal("non-vacuity: no MEASURED-OR-RESOLVED point")
	}
	if _, err := os.Stat(DefaultPromptLengthsPath()); err != nil {
		t.Fatal(err)
	}
}
