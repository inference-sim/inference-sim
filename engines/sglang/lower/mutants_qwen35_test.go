// Original to blis-sglang.

package lower

import (
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"gopkg.in/yaml.v3"
)

// qwen35With is the Qwen3.5 catalog graph with its text edited before it is parsed.
func qwen35With(t *testing.T, edit func(string) string) *model.Graph {
	t.Helper()
	raw, err := os.ReadFile("testdata/qwen3.5-397b-a17b-fp8.graph.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var g model.Graph
	if err := yaml.Unmarshal([]byte(edit(string(raw))), &g); err != nil {
		t.Fatal(err)
	}
	return &g
}

const recurrentNode = `  - op: RecurrentUpdate
    recurrent_kind: gdn
    n_heads: 64
    state_size: 128
    conv_kernel: 4
    intermediate_size: 12288
    state_dtype: fp32
`

func lowerQwen35(t *testing.T, g *model.Graph, block string) (float64, error) {
	t.Helper()
	d := pool(t, 8, block)
	low, err := Pool(PoolInput{Deployment: d, Model: g, Chip: h200(t)})
	if err != nil {
		return 0, err
	}
	return low.Plan.Cache.StateElementBytes, nil
}

// The state's width is the one the graph's recurrent layers state, and a graph whose recurrent layers
// state different widths is refused by name; a layer of another kind states none.
func TestHybridStateWidthComesFromTheRecurrentLayers(t *testing.T) {
	const block = "mem_fraction_static: 0.8\ndisable_radix_cache: true\n"
	if !strings.Contains(recurrentNode, "state_dtype: fp32") {
		t.Fatal("the fixture's recurrent node moved")
	}
	bf16 := qwen35With(t, func(s string) string {
		return strings.Replace(s, recurrentNode, strings.Replace(recurrentNode, "fp32", "bf16", 1), 1)
	})
	if w, err := lowerQwen35(t, bf16, block); err != nil || w != 2 {
		t.Errorf("a graph stating bf16 recurrent state: width %v, error %v, want 2", w, err)
	}
	// An override flag still wins over the graph.
	if w, err := lowerQwen35(t, bf16, block+"mamba_ssm_dtype: float32\n"); err != nil || w != 4 {
		t.Errorf("mamba_ssm_dtype float32 over a bf16 graph: width %v, error %v, want 4", w, err)
	}
	same := qwen35With(t, func(s string) string { return strings.Replace(s, recurrentNode, recurrentNode+recurrentNode, 1) })
	if w, err := lowerQwen35(t, same, block); err != nil || w != 4 {
		t.Errorf("two recurrent layers stating one width: width %v, error %v, want 4", w, err)
	}
	mixed := qwen35With(t, func(s string) string {
		return strings.Replace(s, recurrentNode, recurrentNode+strings.Replace(recurrentNode, "fp32", "bf16", 1), 1)
	})
	_, err := lowerQwen35(t, mixed, block)
	var nl *NotLowered
	if !errors.As(err, &nl) || !strings.Contains(nl.Name, "different state dtypes") {
		t.Errorf("recurrent layers stating different widths: %v, want a refusal naming them", err)
	}
	// A graph that states no width for its recurrent layer gets the float32 default.
	unstated := qwen35With(t, func(s string) string {
		return strings.Replace(s, recurrentNode, strings.Replace(recurrentNode, "    state_dtype: fp32\n", "", 1), 1)
	})
	if w, err := lowerQwen35(t, unstated, block); err != nil || w != 4 {
		t.Errorf("a recurrent layer stating no width: width %v, error %v, want the float32 default 4", w, err)
	}
}
