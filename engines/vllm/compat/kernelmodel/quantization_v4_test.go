package kernelmodel

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// quantizationOracle is oracle/quantization/resolved.json: what vLLM's own ModelConfig
// (vllm/config/model.py:1370-1412@5df63a72, DeepseekV4FP8Config.override_quantization_method at
// vllm/models/deepseek_v4/quant_config.py:140-157@5df63a72) resolves `quantization` to on the
// catalog's deepseek-v4-pro checkpoint, for each value a deployment could state.
type quantizationOracle struct {
	Commit string `json:"vllm"`
	Model  string `json:"model"`
	Cases  []struct {
		Stated   *string `json:"stated"`
		Resolved *string `json:"resolved"`
	} `json:"cases"`
	Control []struct {
		Stated   *string `json:"stated"`
		Resolved *string `json:"resolved"`
	} `json:"control_without_quantization_config"`
}

func loadQuantizationOracle(t *testing.T) quantizationOracle {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "oracle", "quantization", "resolved.json"))
	if err != nil {
		t.Fatal(err)
	}
	var o quantizationOracle
	if err := json.Unmarshal(raw, &o); err != nil {
		t.Fatal(err)
	}
	if o.Commit != "5df63a7203cdb9328d824a27b682e77c7b9928be" || o.Model != "deepseek-v4-pro" {
		t.Fatalf("the oracle file is for %s at %s", o.Model, o.Commit)
	}
	return o
}

// openV4Quantization opens the committed deepseek-v4-pro fixture on an h200 with its
// quantization line set to stated.
func openV4Quantization(t *testing.T, stated string) *Model {
	t.Helper()
	return openQuantization(t, filepath.Join("..", "..", "vllmengine", "testdata", "d011-032.yaml"), stated)
}

// openQuantization opens a committed fixture with its quantization line set to stated ("" removes it).
func openQuantization(t *testing.T, path, stated string) *Model {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	re := regexp.MustCompile(`(?m)^(\s*)quantization:.*\n`)
	if !re.Match(raw) {
		t.Fatal("the fixture states no quantization")
	}
	repl := ""
	if stated != "" {
		repl = "${1}quantization: " + stated + "\n"
	}
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), re.ReplaceAll(raw, []byte(repl)), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Open("v.yaml", r)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// planOf is what the quantization can change in a priced plan: the per-rank fixed bytes, the
// per-sequence bytes at a block, and the step time of a mixed batch (a prefill chunk, a long
// decode and a short one).
func planOf(m *Model) string {
	k := m.Kernel()
	step := m.StepTimeOfShapes([]ReqShape{
		{Scheduled: 512, Computed: 0, PromptLen: 512},
		{Scheduled: 3, Computed: 1500, PromptLen: 1000},
		{Scheduled: 3, Computed: 90, PromptLen: 80},
	})
	return fmt.Sprintf("fixed %+v, variable(256) %d, step %d", k.FixedBytes(), k.SequenceVariableBytes(256), step)
}

func overridesOf(m *Model, field string) []kernel.Override {
	var out []kernel.Override
	for _, o := range m.Kernel().Resolved().Overrides {
		if o.Field == field {
			out = append(out, o)
		}
	}
	return out
}

// On a DeepSeek-V4 FP8 checkpoint, vLLM serves every value a deployment states for `quantization`
// with DeepseekV4FP8Config: its override claims the checkpoint (model_type deepseek_v4, quant_method
// fp8) whatever was named, and the same value on a checkpoint with no quantization_config stays as
// stated (the oracle's control). The kernel prices the checkpoint's own formats whatever is
// stated, so an unstated quantization and `deepseek_v4_fp8` (the value the dashboard's points
// state on this checkpoint) must give one plan, with no override recorded, and every other stated
// value must give that same plan and record the override vLLM makes.
func TestV4FP8QuantizationUnsetAndStatedGiveOnePlan(t *testing.T) {
	o := loadQuantizationOracle(t)
	if len(o.Cases) < 3 {
		t.Fatalf("%d oracle cases", len(o.Cases))
	}
	for _, c := range o.Control {
		if c.Stated != nil && (c.Resolved == nil || *c.Resolved != *c.Stated) {
			t.Fatalf("the control (no quantization_config) resolves %v to %v; the oracle would prove nothing", *c.Stated, c.Resolved)
		}
	}
	const class = "deepseek_v4_fp8"
	base := openV4Quantization(t, "")
	want := planOf(base)
	if ov := overridesOf(base, "quantization"); len(ov) != 0 {
		t.Errorf("an unstated quantization recorded %+v", ov)
	}
	sawClass := false
	for _, c := range o.Cases {
		if c.Resolved == nil || *c.Resolved != class {
			t.Fatalf("vLLM resolved %v to %v; the kernel serves the checkpoint as %s", c.Stated, c.Resolved, class)
		}
		stated := ""
		if c.Stated != nil {
			stated = *c.Stated
		}
		t.Run("stated "+stated, func(t *testing.T) {
			m := openV4Quantization(t, stated)
			if got := planOf(m); got != want {
				t.Errorf("plan differs from the unstated one:\n got %s\nwant %s", got, want)
			}
			ov := overridesOf(m, "quantization")
			switch {
			case stated == "" || stated == *c.Resolved:
				if len(ov) != 0 {
					t.Errorf("recorded %+v where the stated value is vLLM's own resolution", ov)
				}
			case len(ov) != 1 || ov[0].Requested != stated || ov[0].Resolved != *c.Resolved:
				t.Errorf("overrides %+v, want one from %s to vLLM's %s", ov, stated, *c.Resolved)
			}
		})
		sawClass = sawClass || stated == class
	}
	if !sawClass {
		t.Errorf("the oracle file has no %s case", class)
	}
}

// The comparison above can fail: on a checkpoint that is not a DeepSeek-V4 FP8 one (llama-3.1-70b,
// bfloat16), vLLM leaves the stated value as stated (the oracle's control), and the kernel's plan
// for fp8 is then not the unstated one.
func TestQuantizationChangesThePlanOffTheV4Checkpoint(t *testing.T) {
	fixture := filepath.Join("..", "..", "kerneltests", "testdata", "aisimulate", "llama-3.1-70b-instruct-h200-fp8-vllm-tp8.yaml")
	if planOf(openQuantization(t, fixture, "")) == planOf(openQuantization(t, fixture, "fp8")) {
		t.Error("planOf does not see the quantization, so the V4 comparison proves nothing")
	}
}
