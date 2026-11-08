package latencykernel

import (
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/shapetest"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
)

// SUB-10: the cost model runs each unusual model shape or refuses it with a reason, and never
// returns a non-finite or zero size or price where a positive value is required. The shapes
// are priced on every catalog chip, and also in a build that cannot price them, to show a
// refusal is reported rather than hidden.
func TestTheCostModelRunsOrRefusesEveryUnusualShape(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, chip := range d.Chips() {
		chip := chip
		t.Run(chip, func(t *testing.T) {
			shapetest.Run(t, func(s shapetest.Shape) shapetest.Result {
				return priceShape(t, d, s, chip)
			})
		})
	}
}

func priceShape(t *testing.T, d *load.Data, s shapetest.Shape, chip string) shapetest.Result {
	t.Helper()
	cc, err := d.Chip(chip)
	if err != nil {
		t.Fatal(err)
	}
	packaging := "  gpus_per_node: 8\n"
	if cc.GPUsPerNode > 0 {
		packaging = ""
	}
	// The model is named in the catalog so the scenario resolves; its graph is then replaced
	// by the shape, so the cost model prices the shape on the chip's real coefficients.
	body := fmt.Sprintf("kind: Scenario\nname: p\nmodel: qwen3-14b\ncoefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric]\nengine_version: \"0.32.0\"\ncluster:\n  hardware: %s\n  fabric: ib-400g\n  nodes: 1\n%s---\nkind: Deployment\nname: d\npools:\n  - role: colocated\n    nodes: 1\n    parallel: {tp: 8, pp: 1, dp: 1}\n", chip, packaging)
	r, name := openRepos(t, d, body)
	r.Rules = func(string) rules.EngineRules { return windowedRules{} }
	r.EngineSettings = func(*deployment.Deployment, int) (enginesettings.Settings, error) {
		return enginesettings.Settings{BlockSize: 16, MaxNumSeqs: 64, MaxNumBatchedTokens: 8192, MaxModelLen: 8192}, nil
	}
	in, err := OpenInputs(name, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	g := *s.Graph
	g.Name = "qwen3-14b"
	in.Model = &g
	k, err := New(in)
	if err != nil {
		return shapetest.Result{Refusal: err}
	}
	return shapetest.Result{Kernel: k}
}

// A refusal carries the cost model's own words, which name what it could not price.
func TestARefusedShapeStatesItsReason(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range shapetest.Shapes() {
		if r := priceShape(t, d, s, "h100"); r.Refusal != nil && strings.TrimSpace(r.Refusal.Error()) == "" {
			t.Errorf("%s: refused with no reason", s.Name)
		}
	}
}

// windowedRules is the inert rules plus the one engine rule a windowed shape needs to hold any
// cache at all: the pages a request keeps in a window cache, which is blis-vllm's
// rules/pack Pack.WindowPagesPerRequest (cdiv(min(window - 1 + extraRetained + inFlight,
// maxModelLen), block) + 1, vllm/v1/kv_cache_interface.py max_admission_blocks_per_request).
// The inert rules return zero, which prices a window as holding nothing; that is right for the
// tests that do not look at windows and wrong for this one.
type windowedRules struct{ inertRules }

func (windowedRules) WindowPagesPerRequest(c model.Cache, engineBlock, inFlight, extraRetained, maxModelLen int) int {
	block := c.BlockSize
	if block < 1 {
		block = engineBlock
	}
	if c.Kind != model.CacheWindow || block < 1 {
		return 0
	}
	tokens := c.Window - 1 + extraRetained + inFlight
	if maxModelLen > 0 && tokens > maxModelLen {
		tokens = maxModelLen
	}
	return (tokens+block-1)/block + 1
}
