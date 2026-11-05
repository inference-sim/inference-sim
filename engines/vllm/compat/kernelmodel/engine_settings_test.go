package kernelmodel

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	latencykernel "github.com/inference-sim/blis-substrate/kernel"
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

// openWithoutBlockSize opens a committed 0.32.0 fixture with its block_size removed, as a
// generated deployment states none (X19).
func openWithoutBlockSize(t *testing.T, fixture string) (*Model, error) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "kerneltests", "testdata", "aisimulate", fixture))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	if !regexp.MustCompile(`(?m)^\s*block_size:\s*\d+\s*\n`).MatchString(body) {
		t.Fatalf("%s states no block_size to remove", fixture)
	}
	body = regexp.MustCompile(`(?m)^\s*block_size:\s*\d+\s*\n`).ReplaceAllString(body, "")
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return Open("v.yaml", r)
}

// An unstated block_size at 0.32.0 is vLLM's own for the models, GPUs and KV dtypes the
// block-size oracle resolves (core/vllm.ResolveBlockSize), with its source; the kernel prices the
// same size, so it and Engine describe one engine. Anything else is refused, citing #68.
func TestEngineResolvesAnUnstatedBlockSizeAt032(t *testing.T) {
	for _, c := range []struct {
		fixture string
		size    int
		refusal string
	}{
		// FlashAttention on Hopper, FlashInfer on Blackwell: 16 (the default every backend accepts).
		{"llama-3.1-70b-instruct-h200-fp8-vllm-tp2.yaml", 16, ""},
		{"llama-3.1-70b-instruct-b200-fp8-vllm-tp2.yaml", 16, ""},
		// MLA on Blackwell selects FlashInfer MLA, which supports 32 and 64 (mla/flashinfer_mla.py:204-205).
		{"deepseek-v3-b200-fp8-sglang-tp8.yaml", 32, ""},
		// MLA on Hopper with fp8 KV depends on whether the wheel carries FlashMLA.
		{"deepseek-v3-h200-fp8-sglang-tp8.yaml", 0, "flashmla_built"},
	} {
		t.Run(c.fixture, func(t *testing.T) {
			m, err := openWithoutBlockSize(t, c.fixture)
			if err != nil {
				t.Fatal(err)
			}
			e, err := m.Engine()
			if c.refusal != "" {
				if err == nil || !strings.Contains(err.Error(), c.refusal) || !strings.Contains(err.Error(), "blis-vllm/issues/68") {
					t.Fatalf("Engine: %v, want a refusal containing %q citing #68", err, c.refusal)
				}
				return
			}
			if err != nil || e.BlockSize != c.size {
				t.Fatalf("Engine: block_size %d (%v), want %d", e.BlockSize, err, c.size)
			}
			if src := m.EngineSources()["block_size"]; !strings.Contains(src, "vllm/platforms/interface.py:757-781@5df63a72") {
				t.Errorf("block_size source %q, want the rule with its citation", src)
			}
			// The kernel prices the size Engine reports.
			if k, ok := m.Kernel().(*latencykernel.Kernel); !ok {
				t.Errorf("the kernel is a %T, not the cost model", m.Kernel())
			} else if got := k.Engine().BlockSize; got != c.size {
				t.Errorf("the kernel prices block_size %d, Engine reports %d", got, c.size)
			}
		})
	}
}

// A stated block_size is kept, whatever the oracle would resolve, and is reported as stated.
func TestEngineKeepsAStatedBlockSize(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "kerneltests", "testdata", "aisimulate", "deepseek-v3-b200-fp8-sglang-tp8.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	r := repos()
	r.Scenarios = t.TempDir()
	body := regexp.MustCompile(`(?m)^(\s*)block_size:\s*\d+`).ReplaceAllString(string(raw), "${1}block_size: 64")
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Open("v.yaml", r)
	if err != nil {
		t.Fatal(err)
	}
	if e, err := m.Engine(); err != nil || e.BlockSize != 64 || m.EngineSources()["block_size"] != "stated" {
		t.Errorf("Engine: block_size %d (%v), source %q; want 64, stated", e.BlockSize, err, m.EngineSources()["block_size"])
	}
}
