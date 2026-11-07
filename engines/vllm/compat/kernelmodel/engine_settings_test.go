package kernelmodel

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	latencykernel "github.com/inference-sim/blis-substrate/kernel"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
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
	// 0.31.0 resolves only gpu_memory_utilization (TestEngineResolvesAnUnstatedMemoryUtilization);
	// its batch settings and block size are returned as stated.
	if src := m.EngineSources(); len(src) != 1 || src["gpu_memory_utilization"] == "" {
		t.Errorf("0.31.0 sources %v, want gpu_memory_utilization only", src)
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

// openWithMemoryUtilization opens glm-5-h200-tp8-mtp3 at a release with gpu_memory_utilization
// set to line ("" removes it).
func openWithMemoryUtilization(t *testing.T, release, line string) (*Model, error) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "scenarios", "glm-5-h200-tp8-mtp3.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	re := regexp.MustCompile(`(?m)^(\s*)gpu_memory_utilization:.*\n`)
	if !re.Match(raw) {
		t.Fatal("fixture states no gpu_memory_utilization")
	}
	repl := ""
	if line != "" {
		repl = "${1}gpu_memory_utilization: " + line + "\n"
	}
	body := re.ReplaceAllString(string(raw), repl)
	body = strings.Replace(body, `engine_version: "0.29.0"`, `engine_version: "`+release+`"`, 1)
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return Open("v.yaml", r)
}

// An unstated gpu_memory_utilization is vLLM's default, 0.92, in each release the engine models,
// with the line it is written on (vllm/config/cache.py:111, 103 and 102); a stated value is kept;
// and the KV budget is sized from the resolved value, so the kernel and Engine agree.
func TestEngineResolvesAnUnstatedMemoryUtilization(t *testing.T) {
	for _, c := range []struct{ release, source string }{
		{"0.29.0", "vllm/config/cache.py:111@98dff2a81d"},
		{"0.31.0", "vllm/config/cache.py:103@db9527a468"},
		{"0.32.0", "vllm/config/cache.py:102@5df63a7203"},
	} {
		t.Run(c.release, func(t *testing.T) {
			m, err := openWithMemoryUtilization(t, c.release, "")
			if err != nil {
				t.Fatal(err)
			}
			e, err := m.Engine()
			if err != nil || e.GPUMemoryUtilization != 0.92 {
				t.Fatalf("Engine: gpu_memory_utilization %v (%v), want 0.92", e.GPUMemoryUtilization, err)
			}
			if src := m.EngineSources()["gpu_memory_utilization"]; !strings.Contains(src, c.source) {
				t.Errorf("source %q, want it to cite %s", src, c.source)
			}
			if k, ok := m.Kernel().(*latencykernel.Kernel); !ok || k.Engine().GPUMemoryUtilization != 0.92 {
				t.Errorf("the kernel does not price the resolved 0.92")
			}
			// The KV budget is sized from it: a stated 0.92 gives the same budget.
			stated, err := openWithMemoryUtilization(t, c.release, "0.92")
			if err != nil {
				t.Fatal(err)
			}
			a, errA := m.KVMemory()
			b, errB := stated.KVMemory()
			if errA != nil || errB != nil || a.AllocatableBytes != b.AllocatableBytes {
				t.Errorf("KV budget %v (%v) differs from a stated 0.92's %v (%v)", a.AllocatableBytes, errA, b.AllocatableBytes, errB)
			}
		})
	}
}

// A stated gpu_memory_utilization is kept and reported as stated; a stated value outside (0, 1]
// is still refused, not replaced by the default.
func TestEngineKeepsOrRefusesAStatedMemoryUtilization(t *testing.T) {
	m, err := openWithMemoryUtilization(t, "0.32.0", "0.85")
	if err != nil {
		t.Fatal(err)
	}
	if e, err := m.Engine(); err != nil || e.GPUMemoryUtilization != 0.85 || m.EngineSources()["gpu_memory_utilization"] != "stated" {
		t.Errorf("Engine: %v (%v), source %q; want 0.85, stated", e.GPUMemoryUtilization, err, m.EngineSources()["gpu_memory_utilization"])
	}
	// Document validation refuses a stated value outside [0, 1] before a model opens.
	for _, v := range []string{"1.5", "-0.1"} {
		if _, err := openWithMemoryUtilization(t, "0.32.0", v); err == nil {
			t.Errorf("stated %s opened, want it refused", v)
		}
	}
	// And the resolver never replaces such a value with the default, whatever reaches it: only
	// 0, which the schema reads as unset (rules/engineconfig/validate.go:65), is replaced.
	in := latencykernel.Inputs{Scenario: &scenario.Scenario{EngineVersion: "0.32.0"},
		Deployment: &deployment.Deployment{Pools: []deployment.Pool{{}}}, Chip: &hardware.Chip{MemoryGiB: 141}, Model: &model.Graph{}}
	for _, c := range []struct {
		util float64
		want float64
	}{{0, 0.92}, {1.5, 1.5}, {-0.1, -0.1}, {0.5, 0.5}, {1, 1}} {
		got := resolveEngine(in, engineconfig.Engine{GPUMemoryUtilization: c.util}).engine.GPUMemoryUtilization
		if got != c.want {
			t.Errorf("resolveEngine with gpu_memory_utilization %v gave %v, want %v", c.util, got, c.want)
		}
	}
}

// A pool wider than the block-size oracle ran (core/vllm.MaxBlockSizeTP) is refused, citing #68,
// when its block_size is unstated: the per-rank head counts that feed vLLM's backend selection
// change with the width. A pool at or under the limit is unaffected. The width reaches the
// resolution through BlockSizeInput.TP.
func TestEngineRefusesAnUnstatedBlockSizeOnAPoolWiderThanTheOracleRan(t *testing.T) {
	const fixture = "minimax-m2.5-b200-fp4-vllm-tp4.yaml"
	raw, err := os.ReadFile(filepath.Join("..", "..", "kerneltests", "testdata", "aisimulate", fixture))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		tp      int
		refused bool
	}{{4, false}, {16, false}, {32, true}} {
		body := string(raw)
		// gb200-nvl72 holds a rack of 72 GPUs, so the widest pool fits one cluster.
		nodes := strconv.Itoa(max(c.tp/4, 1))
		for _, e := range [][2]string{
			{"  hardware: b200\n  nodes: 1\n  gpus_per_node: 8\n", "  hardware: gb200-nvl72\n  fabric: ib-400g\n  nodes: " + nodes + "\n"},
			{"    nodes: 1\n    parallel", "    nodes: " + nodes + "\n    parallel"},
			{"      tp: 4\n", "      tp: " + strconv.Itoa(c.tp) + "\n"},
			{"      block_size: 16\n", ""},
		} {
			old, repl := e[0], e[1]
			if !strings.Contains(body, old) {
				t.Fatalf("%s has no %q", fixture, old)
			}
			body = strings.Replace(body, old, repl, 1)
		}
		r := repos()
		r.Scenarios = t.TempDir()
		if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		m, err := Open("v.yaml", r)
		if err != nil {
			t.Fatalf("tp %d: %v", c.tp, err)
		}
		e, err := m.Engine()
		switch {
		case c.refused && (err == nil || !strings.Contains(err.Error(), "blis-vllm/issues/68") || !strings.Contains(err.Error(), "tensor_parallel_size")):
			t.Errorf("tp %d: Engine: %v, want the #68 refusal for a width the oracle did not run", c.tp, err)
		case !c.refused && (err != nil || e.BlockSize != 16):
			t.Errorf("tp %d: Engine: block_size %d (%v), want 16 unaffected", c.tp, e.BlockSize, err)
		}
	}
}
