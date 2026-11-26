package cmd

import (
	"path/filepath"
	"strings"
	"testing"
)

// An SGLang pool that speculates (EAGLE at top-k 1) runs with an acceptance flag: the host
// resolves the acceptance from the speculation the pool states (its drafted tokens a step), the
// engine verifies speculative_num_draft_tokens rows a request, and the run records the acceptance
// it advanced requests by. A scenario that does not speculate is still told it drafts no tokens.
const sglangEagle3 = "      speculative_algorithm: EAGLE\n      speculative_num_steps: 3\n      speculative_eagle_topk: 1\n      speculative_num_draft_tokens: 4\n"

func sglangSpecRun(t *testing.T, spec string, extra ...string) string {
	t.Helper()
	const name = "glm-5-h200-fp8-sglang-tp8.yaml"
	dir := sglangScenarioCopy(t, name, func(s string) string { return s + spec })
	args := append([]string{"run", "--scenarios", dir, "--scenario", name, "--rate", "2", "--num-requests", "8",
		"--prompt-tokens", "512", "--output-tokens", "96", "--seed", "1"}, extra...)
	stdout, stderr, err := runKernelCLI(t, args...)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 4))
	}
	if !completedIs(stdout, 8) {
		t.Fatalf("want 8 completed requests\n%s", lastLines(stderr, 4))
	}
	return stdout
}

func TestSGLangSpeculationRunsAndRecordsItsAcceptance(t *testing.T) {
	for _, flag := range [][]string{{"--speculative-acceptance-length", "2.5"}, {"--speculative-acceptance-rate", "0.5"}} {
		out := sglangSpecRun(t, sglangEagle3, flag...)
		if !strings.Contains(out, `"source": "`+flag[0]+`"`) {
			t.Errorf("%s: the Result records no acceptance for a pool that drafts", flag[0])
		}
	}
}

// At acceptance length 1 the drafts gain nothing, so a token costs the verify's width and the draft
// passes: slower than the pool without speculation, by the draft work only. At length 3 each verify
// step advances a request by 3 tokens, so the time per token falls about 3x.
func TestSGLangSpeculationAtAcceptanceLengthOneCostsOnlyTheVerifyAndTheDrafter(t *testing.T) {
	plain := sglangSpecRun(t, "")
	spec1 := sglangSpecRun(t, sglangEagle3, "--speculative-acceptance-length", "1")
	spec3 := sglangSpecRun(t, sglangEagle3, "--speculative-acceptance-length", "3")
	tp, t1, t3 := metric(t, plain, "tpot_mean_ms"), metric(t, spec1, "tpot_mean_ms"), metric(t, spec3, "tpot_mean_ms")
	if !(t1 > tp) || t1 > 2.5*tp {
		t.Errorf("tpot at acceptance length 1 is %.3f ms against %.3f ms without speculation: want slower, by the draft work only", t1, tp)
	}
	if r := t1 / t3; r < 2.0 || r > 3.2 {
		t.Errorf("tpot falls %.2fx from acceptance length 1 to 3, want about 3x", r)
	}
}

func TestRunCmd_SGLangAcceptanceFlagOnAPoolThatDoesNotSpeculateIsRefused(t *testing.T) {
	const name = "deepseek-r1-h200-fp8-sglang-tp8.yaml"
	plain, err := filepath.Abs(filepath.Join("..", "testdata", "kernel", "sglang"))
	if err != nil {
		t.Fatal(err)
	}
	_, stderr, err := runKernelCLI(t, "run", "--scenarios", plain, "--scenario", name,
		"--rate", "4", "--num-requests", "4", "--seed", "1", "--speculative-acceptance-length", "2.5")
	if err == nil || !strings.Contains(stderr, "drafts no tokens") {
		t.Fatalf("want a refusal naming \"drafts no tokens\", got %v\n%s", err, lastLines(stderr, 3))
	}
}
