package cmd

import (
	"path/filepath"
	"strings"
	"testing"
)

// An acceptance flag on a scenario that speculates through the SGLang engine is refused for the
// reason that is true of it (inference-sim#2315). The host resolves acceptance from the
// vLLM-form speculative block, which an SGLang scenario does not state, so K reads as zero and
// the flag was refused as "drafts no tokens": a scenario that states speculative_num_draft_tokens
// does draft. The refusal names the SGLang algorithm and the draft width it states and the
// engine's issue (inference-sim/blis-sglang#68), and a scenario that does not speculate is still
// told it drafts no tokens.
func TestRunCmd_SGLangSpeculationWithAnAcceptanceFlagIsRefusedForItsOwnReason(t *testing.T) {
	const name = "deepseek-r1-h200-fp8-sglang-tp8.yaml"
	speculating := sglangScenarioCopy(t, name, func(s string) string {
		return strings.Replace(s, "      context_length: 163840\n",
			"      context_length: 163840\n      speculative_algorithm: \"EAGLE\"\n      speculative_num_steps: 3\n"+
				"      speculative_eagle_topk: 1\n      speculative_num_draft_tokens: 4\n", 1)
	})
	plain, err := filepath.Abs(filepath.Join("..", "testdata", "kernel", "sglang"))
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name, dir, flag, value string
		want                   []string
		notWant                string
	}{
		{"length, speculating", speculating, "--speculative-acceptance-length", "2.5",
			[]string{"speculative_algorithm EAGLE", "3 drafts", "inference-sim/blis-sglang#68"}, "drafts no tokens"},
		{"rate, speculating", speculating, "--speculative-acceptance-rate", "0.6",
			[]string{"speculative_algorithm EAGLE", "3 drafts", "inference-sim/blis-sglang#68"}, "drafts no tokens"},
		{"length, not speculating", plain, "--speculative-acceptance-length", "2.5",
			[]string{"drafts no tokens"}, "blis-sglang#68"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, stderr, err := runKernelCLI(t, "run", "--scenarios", tt.dir, "--scenario", name,
				"--rate", "4", "--num-requests", "4", "--seed", "1", tt.flag, tt.value)
			if err == nil {
				t.Fatalf("accepted; want a refusal naming %q", tt.want)
			}
			for _, w := range tt.want {
				if !strings.Contains(stderr, w) {
					t.Errorf("the refusal does not name %q:\n%s", w, lastLines(stderr, 3))
				}
			}
			if strings.Contains(stderr, tt.notWant) {
				t.Errorf("the refusal names %q, which is not the cause:\n%s", tt.notWant, lastLines(stderr, 3))
			}
		})
	}
}
