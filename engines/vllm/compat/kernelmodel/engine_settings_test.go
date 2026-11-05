package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// openEdited opens glm-5-h200-tp8-mtp3 (h200, block_size 64, max_num_batched_tokens 8192,
// max_num_seqs 256, max_model_len 32768) at a release, with each named line removed.
func openEdited(t *testing.T, release string, drop ...string) *Model {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "scenarios", "glm-5-h200-tp8-mtp3.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	body := strings.Replace(string(raw), `engine_version: "0.29.0"`, `engine_version: "`+release+`"`, 1)
	for _, d := range drop {
		line := "      " + d + "\n"
		if !strings.Contains(body, line) {
			t.Fatalf("fixture has no %q", d)
		}
		body = strings.Replace(body, line, "", 1)
	}
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Open("v.yaml", r)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// At vLLM 0.32.0, Engine returns the batch settings as vLLM resolves them: a stated value is
// kept, and an unstated one is vLLM's default for the GPU (h200: 1024 sequences, 8192 tokens;
// vllm/engine/arg_utils.py:2933-2942@5df63a72), each with its source in EngineSources.
func TestEngineResolvesUnstatedBatchSettingsAt032(t *testing.T) {
	for _, c := range []struct {
		name         string
		drop         []string
		seqs, tokens int
		seqsStated   bool
		tokensStated bool
	}{
		{"both stated", nil, 256, 8192, true, true},
		{"max_num_seqs unstated", []string{"max_num_seqs: 256"}, 1024, 8192, false, true},
		{"max_num_batched_tokens unstated", []string{"max_num_batched_tokens: 8192"}, 256, 8192, true, false},
		{"both unstated", []string{"max_num_seqs: 256", "max_num_batched_tokens: 8192"}, 1024, 8192, false, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			m := openEdited(t, "0.32.0", c.drop...)
			e, err := m.Engine()
			if err != nil {
				t.Fatal(err)
			}
			if e.MaxNumSeqs != c.seqs || e.MaxNumBatchedTokens != c.tokens || e.BlockSize != 64 {
				t.Errorf("Engine: max_num_seqs %d, max_num_batched_tokens %d, block_size %d; want %d, %d, 64",
					e.MaxNumSeqs, e.MaxNumBatchedTokens, e.BlockSize, c.seqs, c.tokens)
			}
			src := m.EngineSources()
			for name, stated := range map[string]bool{"max_num_seqs": c.seqsStated, "max_num_batched_tokens": c.tokensStated} {
				switch s := src[name]; {
				case stated && s != "stated":
					t.Errorf("%s source %q, want stated", name, s)
				case !stated && !strings.Contains(s, "vllm/engine/arg_utils.py"):
					t.Errorf("%s source %q, want the rule with its citation", name, s)
				}
			}
		})
	}
}

// An unstated block_size at 0.32.0 is still refused, naming the open issue (#68); at 0.31.0 the
// block is returned as stated, and an unstated batch setting is refused, as before.
func TestEngineRefusesWhatItDoesNotResolve(t *testing.T) {
	_, err := openEdited(t, "0.32.0", "block_size: 64").Engine()
	if err == nil || !strings.Contains(err.Error(), "blis-vllm/issues/68") {
		t.Errorf("0.32.0 without block_size: %v, want a refusal citing blis-vllm#68", err)
	}
	m := openEdited(t, "0.31.0", "max_num_seqs: 256")
	if _, err := m.Engine(); err == nil || !strings.Contains(err.Error(), "states no max_num_seqs") {
		t.Errorf("0.31.0 without max_num_seqs: %v, want the refusal", err)
	}
	if src := m.EngineSources(); len(src) != 0 {
		t.Errorf("0.31.0 sources %v, want none", src)
	}
}
