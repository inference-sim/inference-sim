package kernelmodel

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-vllm/internal/substratedata"
)

// openGLM opens a committed GLM-5 fixture with its cache_dtype line replaced by line ("" removes
// it), on the given chip.
func openGLM(t *testing.T, fixture, hardware, line string) (*Model, error) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "kerneltests", "testdata", "aisimulate", fixture))
	if err != nil {
		t.Fatal(err)
	}
	re := regexp.MustCompile(`(?m)^(\s*)cache_dtype:.*\n`)
	if !re.Match(raw) {
		t.Fatalf("%s states no cache_dtype", fixture)
	}
	repl := ""
	if line != "" {
		repl = "${1}cache_dtype: " + line + "\n"
	}
	body := re.ReplaceAllString(string(raw), repl)
	body = strings.Replace(body, "hardware: "+strings.Fields(regexp.MustCompile(`hardware: \S+`).FindString(body))[1], "hardware: "+hardware, 1)
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return Open("v.yaml", r)
}

// GlmMoeDsaForCausalLM.verify_and_update_config changes an unstated (auto) cache dtype to
// fp8_e4m3 on CUDA capability family 100 (vllm/model_executor/models/config.py:43-55@5df63a72), so
// a GLM-5 deployment that states none runs, and is priced and sized with, a one-byte cache on a
// B200 and keeps the bf16 cache on a Hopper. A stated dtype is never changed.
func TestAnUnstatedCacheDTypeOfAGLM5IsFP8OnSM10x(t *testing.T) {
	const fixture = "glm-5-b200-fp8-sglang-tp8.yaml"
	for _, c := range []struct {
		name, hardware, stated, want string
		perBlock                     int64
	}{
		{"unstated on SM10x", "b200", "", "fp8_e4m3", 2875392},
		{"auto on SM10x", "b200", "auto", "fp8_e4m3", 2875392},
		{"unstated on SM10x, B300", "b300", "", "fp8_e4m3", 2875392},
		{"a stated fp8 stays fp8", "b200", "fp8", "fp8", 2875392},
		{"a stated bfloat16 stays bfloat16", "b200", "bfloat16", "bfloat16", 5750784},
		{"unstated on SM9x", "h200", "", "", 5750784},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, err := openGLM(t, fixture, c.hardware, c.stated)
			if err != nil {
				t.Fatal(err)
			}
			e, err := m.Engine()
			if err != nil {
				t.Fatal(err)
			}
			if e.CacheDType != c.want {
				t.Errorf("Engine cache_dtype %q, want %q", e.CacheDType, c.want)
			}
			b, err := m.KVBudget()
			if err != nil {
				t.Fatal(err)
			}
			if b.PerBlockBytes != c.perBlock {
				t.Errorf("%d bytes a block, want %d", b.PerBlockBytes, c.perBlock)
			}
			if c.stated == "" && c.want != "" && !strings.Contains(m.EngineSources()["cache_dtype"], "config.py:43-55@5df63a72") {
				t.Errorf("source %q, want the rule with its citation", m.EngineSources()["cache_dtype"])
			}
		})
	}
}

// The catalog graphs of GLM-5, 5.2, 5.2-fp8 and 5.3 state no global.architecture, so the engine
// names their architecture by the digest of the config.json they were derived from. Each digest is
// the one the catalog's config.json has, that config's architectures[0] is GlmMoeDsaForCausalLM, and
// no other catalog model has that architecture.
func TestGLMConfigHashesNameTheArchitecture(t *testing.T) {
	dir, err := substratedata.Catalog()
	if err != nil {
		t.Fatal(err)
	}
	data := repos().Data
	named := map[string]bool{}
	for _, name := range data.Models() {
		g, err := data.Model(name)
		if err != nil {
			t.Fatal(err)
		}
		raw, err := os.ReadFile(filepath.Join(dir, "models", name, "config.json"))
		if err != nil {
			continue
		}
		isGLM := regexp.MustCompile(`"architectures"\s*:\s*\[\s*"GlmMoeDsaForCausalLM"`).Match(raw)
		listed := glmMoeDsaConfigs[g.DerivedFrom.SHA256] == name
		if isGLM != listed {
			t.Errorf("%s: config.json architecture GlmMoeDsaForCausalLM is %t, glmMoeDsaConfigs lists it %t", name, isGLM, listed)
		}
		if listed {
			named[name] = true
			if got := architectureOf(g); got != "GlmMoeDsaForCausalLM" {
				t.Errorf("%s: architectureOf %q", name, got)
			}
		}
	}
	if len(named) != len(glmMoeDsaConfigs) {
		t.Errorf("%d of %d listed models found in the catalog", len(named), len(glmMoeDsaConfigs))
	}
	if architectureOf(&model.Graph{}) != "" {
		t.Error("a graph with no architecture and no listed digest was given one")
	}
}
