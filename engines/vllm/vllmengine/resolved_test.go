package vllmengine

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"testing"

	"gopkg.in/yaml.v3"

	latencykernel "github.com/inference-sim/blis-substrate/kernel"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// scenarioWithout writes a committed 0.32.0 fixture to a temporary directory with the named
// settings removed, as a regenerated deployment states none (X19), and returns its roots.
func scenarioWithout(t *testing.T, fixture string, drop ...string) map[string]string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "kerneltests", "testdata", "aisimulate", fixture))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	for _, k := range drop {
		re := regexp.MustCompile(`(?m)^\s*` + k + `:.*\n`)
		if !re.MatchString(body) {
			t.Fatalf("%s states no %s", fixture, k)
		}
		body = re.ReplaceAllString(body, "")
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "v.yaml")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return map[string]string{RootScenario: p}
}

func blockInt(t *testing.T, block yaml.Node, key string) int {
	t.Helper()
	var m map[string]any
	if err := block.Decode(&m); err != nil {
		t.Fatal(err)
	}
	v, ok := m[key]
	if !ok {
		t.Fatalf("the completed engine block states no %s", key)
	}
	n, err := strconv.Atoi(toString(v))
	if err != nil {
		t.Fatalf("%s = %v", key, v)
	}
	return n
}

func toString(v any) string {
	switch x := v.(type) {
	case int:
		return strconv.Itoa(x)
	case string:
		return x
	}
	return ""
}

// A deployment that states none of block_size, max_num_seqs, max_num_batched_tokens and
// gpu_memory_utilization builds, and the engine block the factory is given, Model.Engine and the kernel's
// own settings carry the same values, so the scheduler runs the batch the cost model prices.
func TestAnEngineBuildsFromAnUnstatedDeploymentWithOneSetOfValues(t *testing.T) {
	Register()
	const fixture = "gpt-oss-120b-h200-fp4-vllm-tp4.yaml"
	roots := scenarioWithout(t, fixture, "block_size", "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization")
	spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots, Pool: 0}
	p, err := open(spec, nil)
	if err != nil {
		t.Fatal(err)
	}
	block := p.complete(yaml.Node{})
	e, err := p.model.Engine()
	if err != nil {
		t.Fatal(err)
	}
	k, ok := p.model.Kernel().(*latencykernel.Kernel)
	if !ok {
		t.Fatalf("the kernel is a %T", p.model.Kernel())
	}
	for _, c := range []struct {
		key           string
		model, kernel int
	}{
		{"block_size", e.BlockSize, k.Engine().BlockSize},
		{"max_num_seqs", e.MaxNumSeqs, p.resolved.MaxNumSeqs},
		{"max_num_batched_tokens", e.MaxNumBatchedTokens, p.resolved.MaxNumBatchedTokens},
	} {
		if got := blockInt(t, block, c.key); got != c.model || got != c.kernel || got <= 0 {
			t.Errorf("%s: engine block %d, Model.Engine %d, kernel/resolved %d; want one positive value", c.key, got, c.model, c.kernel)
		}
	}
	if _, err := engine.New(spec); err != nil {
		t.Errorf("the engine does not build from an unstated deployment: %v", err)
	}
}

// A value the file states wins: complete never replaces it.
func TestAStatedEngineSettingIsNeverReplaced(t *testing.T) {
	Register()
	spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Pool: 0,
		Roots: scenarioWithout(t, "gpt-oss-120b-h200-fp4-vllm-tp4.yaml", "max_num_seqs", "max_num_batched_tokens")}
	p, err := open(spec, nil)
	if err != nil {
		t.Fatal(err)
	}
	var given yaml.Node
	if err := yaml.Unmarshal([]byte("block_size: 64\nmax_num_seqs: 7\nmax_num_batched_tokens: 999\nmax_model_len: 12345\n"), &given); err != nil {
		t.Fatal(err)
	}
	block := p.complete(given)
	for k, want := range map[string]int{"block_size": 64, "max_num_seqs": 7, "max_num_batched_tokens": 999, "max_model_len": 12345} {
		if got := blockInt(t, block, k); got != want {
			t.Errorf("%s = %d, want the stated %d", k, got, want)
		}
	}
}
