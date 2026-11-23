package cmd

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"
	"testing"
)

// P/D pools state a speculative configuration of their own (inference-sim#2125): vLLM takes each
// server's speculative_config, and its P/D connector fingerprint carries none, so a prefill pool
// may draft one token and the decode pool three. Each pool's instances advance by that pool's
// own draft length and acceptance.

const (
	prefillDraft = "      compilation_config: {cudagraph_mode: PIECEWISE}\n  - role: decode"
	decodeDraft  = "      compilation_config: {cudagraph_mode: PIECEWISE}\n\npd_transfer"
)

// draftingPools is the 3P1D fixture with each pool drafting its stated number of MTP tokens (0
// states no speculative block), in a scenario directory of its own.
func draftingPools(t *testing.T, prefillK, decodeK int) string {
	t.Helper()
	line := func(k int) string {
		if k == 0 {
			return ""
		}
		return "      speculative_config: {method: mtp, num_speculative_tokens: " + strconv.Itoa(k) + "}\n"
	}
	return writeScenarioVariant(t,
		prefillDraft, strings.Replace(prefillDraft, "PIECEWISE}\n", "PIECEWISE}\n"+line(prefillK), 1),
		decodeDraft, strings.Replace(decodeDraft, "PIECEWISE}\n", "PIECEWISE}\n"+line(decodeK), 1))
}

// speculationRun is a P/D run of the fixture's pools, parsed.
type speculationRun struct {
	doc    map[string]any
	stdout string
}

func runSpeculatingPools(t *testing.T, dir string, extra ...string) speculationRun {
	t.Helper()
	args := append(runArgs(pdBase(dir, "variant.yaml"), burst(24, 12, 1500, 128)...), extra...)
	stdout, stderr, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatalf("blis %s: %v\n%s", strings.Join(args, " "), err, lastLines(stderr, 6))
	}
	var doc map[string]any
	if err := json.Unmarshal([]byte(stdout), &doc); err != nil {
		t.Fatal(err)
	}
	if !completedIs(stdout, 24) {
		t.Fatalf("the run did not complete all 24 requests:\n%s", lastLines(stdout, 5))
	}
	return speculationRun{doc: doc, stdout: stdout}
}

func at(m any, path ...string) any {
	for _, p := range path {
		mm, ok := m.(map[string]any)
		if !ok {
			return nil
		}
		m = mm[p]
	}
	return m
}

func (r speculationRun) num(t *testing.T, path ...string) float64 {
	t.Helper()
	v, ok := at(r.doc, path...).(float64)
	if !ok {
		t.Fatalf("the Result has no number at %s", strings.Join(path, "."))
	}
	return v
}

// decodeSteps is the decode pool's steps: the inter-chunk interval has one sample per verified
// step, however many tokens it advanced (the one decode instance of the fixture's 3P1D).
func (r speculationRun) decodeSteps(t *testing.T) float64 {
	t.Helper()
	return r.num(t, "metrics", "icl_mean_ms", "samples")
}

func (r speculationRun) outputTokens(t *testing.T) float64 {
	t.Helper()
	return r.num(t, "reports", "engine", "cluster", "total_output_tokens")
}

// A prefill pool drafting one token and a decode pool drafting three run, and each pool's
// record states its own K: with the mean acceptance length stated for the run, a pool's rate is
// (AL-1)/K of its own K. The decode pool advances requests by the stated AL tokens a step, so
// its steps are the output tokens over AL; priced at the prefill pool's rate (0.8 at K=1) it
// would advance 1 + 3*0.8 = 3.4 tokens a step.
func TestPDPools_EachPoolAdvancesByItsOwnDraftLengthAndAcceptance(t *testing.T) {
	const al = 1.8
	dir := draftingPools(t, 1, 3)
	r := runSpeculatingPools(t, dir, "--speculative-acceptance-length", "1.8")
	for _, p := range []struct {
		role string
		k    float64
	}{{"prefill", 1}, {"decode", 3}} {
		rec := at(r.doc, "reports", "engine", p.role, "acceptance")
		if rec == nil {
			t.Fatalf("no acceptance record for the %s pool", p.role)
		}
		if got, want := r.num(t, "reports", "engine", p.role, "acceptance", "rate"), (al-1)/p.k; math.Abs(got-want) > 1e-9 {
			t.Errorf("%s pool (K=%v): rate %v, want (AL-1)/K = %v", p.role, p.k, got, want)
		}
		if got := r.num(t, "reports", "engine", p.role, "acceptance", "length"); math.Abs(got-al) > 1e-9 {
			t.Errorf("%s pool: length %v, want the stated %v", p.role, got, al)
		}
	}
	if at(r.doc, "reports", "engine", "cluster", "acceptance") != nil {
		t.Errorf("pools with different configurations share a cluster-wide acceptance record; it would state one pool's K for both")
	}
	steps, tokens := r.decodeSteps(t), r.outputTokens(t)
	if got := tokens / steps; math.Abs(got-al)/al > 0.05 {
		t.Errorf("the decode pool advanced %.3f tokens a step (%v tokens in %v steps), want about the stated AL %v; "+
			"%.1f would be the prefill pool's rate applied to it", got, tokens, steps, al, 1+3*(al-1))
	}
}

// Law across draft lengths: accepting every draft, the decode pool advances 1+K tokens a step
// with K its own, whatever the prefill pool drafts.
func TestPDPools_DecodeStepsFollowTheDecodePoolsK(t *testing.T) {
	for _, c := range []struct{ prefillK, decodeK int }{{1, 1}, {1, 3}, {0, 3}, {3, 3}} {
		r := runSpeculatingPools(t, draftingPools(t, c.prefillK, c.decodeK), "--speculative-acceptance-rate", "1.0")
		got := r.outputTokens(t) / r.decodeSteps(t)
		if want := float64(1 + c.decodeK); math.Abs(got-want)/want > 0.05 {
			t.Errorf("prefill K=%d, decode K=%d: %.3f tokens a step, want 1+K = %v", c.prefillK, c.decodeK, got, want)
		}
	}
}

// A pool that drafts nothing beside a pool that does takes no acceptance: the run-wide
// command-line acceptance applies to the pools that draft, and only the decode pool has a record.
func TestPDPools_APoolThatDoesNotDraftBesideOneThatDoes(t *testing.T) {
	r := runSpeculatingPools(t, draftingPools(t, 0, 2), "--speculative-acceptance-rate", "0.5")
	if got := r.num(t, "reports", "engine", "decode", "acceptance", "length"); math.Abs(got-2) > 1e-9 {
		t.Errorf("decode pool length %v, want 1 + 2*0.5 = 2", got)
	}
	if at(r.doc, "reports", "engine", "prefill", "acceptance") != nil {
		t.Errorf("the prefill pool drafts nothing and has an acceptance record")
	}
	if got := r.outputTokens(t) / r.decodeSteps(t); math.Abs(got-2)/2 > 0.05 {
		t.Errorf("decode pool advanced %.3f tokens a step, want 2", got)
	}
}

// Pools that draft alike keep the run's single acceptance record and no per-pool one: a result
// that did not differ before the pools could differ does not change.
func TestPDPools_PoolsDraftingAlikeKeepTheClusterRecord(t *testing.T) {
	r := runSpeculatingPools(t, draftingPools(t, 3, 3), "--speculative-acceptance-rate", "1.0")
	if got := r.num(t, "reports", "engine", "cluster", "acceptance", "length"); got != 4 {
		t.Errorf("cluster acceptance length %v, want 4", got)
	}
	for _, role := range []string{"prefill", "decode"} {
		if at(r.doc, "reports", "engine", role, "acceptance") != nil {
			t.Errorf("the %s pool states an acceptance record of its own beside the cluster's", role)
		}
	}
}

// INV-13: a replay of a run whose pools draft differently is the run.
func TestPDPools_DifferentDraftLengthsReplayAsTheyRan(t *testing.T) {
	o := capRunReplay(t, with(pdBase(draftingPools(t, 1, 3), "variant.yaml"), "--speculative-acceptance-length", "1.8"),
		burst(24, 12, 1500, 128), []string{})
	o.conserves(t, 24)
}

// A command-line length no pool's K can hold is still refused by the resolver, naming the flag:
// the K=1 prefill pool cannot accept 2.5 tokens a step.
func TestPDPools_ALBeyondAPoolsKIsRefused(t *testing.T) {
	args := append(runArgs(pdBase(draftingPools(t, 1, 3), "variant.yaml"), burst(24, 12, 1500, 128)...),
		"--speculative-acceptance-length", "2.5")
	_, stderr, err := runKernelCLI(t, args...)
	if err == nil || !strings.Contains(stderr, "speculative-acceptance-length") {
		t.Fatalf("an acceptance length of 2.5 at a pool drafting 1 token was not refused naming the flag: %v\n%s", err, lastLines(stderr, 5))
	}
}
