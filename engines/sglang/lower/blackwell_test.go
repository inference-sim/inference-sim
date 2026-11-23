// SPDX-License-Identifier: Apache-2.0

package lower

import (
	"encoding/json"
	"reflect"
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/hardware"

	"github.com/inference-sim/inference-sim/engines/sglang/config"
	"github.com/inference-sim/inference-sim/engines/sglang/internal/testgz"
)

func chipFile(t *testing.T, name string) *hardware.Chip {
	t.Helper()
	var c hardware.Chip
	readYAML(t, "testdata/"+name+".yaml", &c)
	c.Name = name
	return &c
}

// B200 (compute capability 10.0) and B300 (10.3) are SM100 to SGLang's resolution, which reads the
// major, and so are the GB200 NVL72 (10.0) and GB300 (10.3) parts; their nvidia-smi memory is the
// oracle's figure for each (oracle/serverargs/cases.json "chips").
func TestBlackwellChipsAreLowered(t *testing.T) {
	for _, tc := range []struct {
		chip string
		cc   [2]int
		mib  float64
	}{
		{"b200", [2]int{10, 0}, 183359},
		{"b300", [2]int{10, 3}, 275040},
		{"gb200-nvl72", [2]int{10, 0}, 189471},
		{"gb300", [2]int{10, 3}, 284208},
	} {
		chip := &hardware.Chip{Name: tc.chip, ComputeCapability: strconv.Itoa(tc.cc[0]) + "." + strconv.Itoa(tc.cc[1])}
		c, err := ChipOf(chip)
		if err != nil {
			t.Fatalf("%s: %v", tc.chip, err)
		}
		if c.Capability != tc.cc || c.MemoryMiB != tc.mib {
			t.Errorf("%s: capability %v memory %v MiB; want %v and %v", tc.chip, c.Capability, c.MemoryMiB, tc.cc, tc.mib)
		}
	}
}

// The facts the lowering hands config.Resolve for a catalog DeepSeek-R1 or GLM-5 checkpoint are the
// facts SGLang derives for the same checkpoint (the cases of oracle/serverargs/cases.json).
func TestBlackwellModelFactsMatchTheOracle(t *testing.T) {
	a := readAnswersByCase(t)
	for _, tc := range []struct{ graph, oracleCase string }{
		{"deepseek-r1-0528", "dsr1-fp8-b200-tp8"},
		{"deepseek-r1-0528-fp4-v2", "dsr1-nvfp4v2-b200-tp8"},
		{"glm-5", "glm5-fp8-b200-tp8"},
		{"glm-5-nvfp4", "glm5-nvfp4-b200-tp8"},
	} {
		want, ok := a[tc.oracleCase]
		if !ok {
			t.Fatalf("the oracle has no case %s", tc.oracleCase)
		}
		got, err := ModelFactsOf(graph(t, tc.graph))
		if err != nil {
			t.Fatal(err)
		}
		got.HiddenSize, want.HiddenSize = nil, nil
		// The oracle records an empty list for a checkpoint that lists no mixed-precision algorithms;
		// the lowering leaves it unset. Both state that there are none.
		if len(got.MixedPrecisionMoEQuantAlgos) == 0 && len(want.MixedPrecisionMoEQuantAlgos) == 0 {
			got.MixedPrecisionMoEQuantAlgos, want.MixedPrecisionMoEQuantAlgos = nil, nil
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s: facts %+v, SGLang derives %+v", tc.graph, got, want)
		}
	}
}

// DeepSeek-R1 on a B200 or B300 at SGLang's defaults: trtllm_mla, FlashInfer's TRT-LLM MoE runner
// (fp8 from the checkpoint's declaration, nvfp4 from none), and no prefill CUDA graph. The runner's
// kernel is not named (the substrate prices it from the experts' format), so the pool lowers.
func TestBlackwellDeepSeekResolvesAndLowersTheTRTLLMRunner(t *testing.T) {
	for _, tc := range []struct{ graph, chip string }{
		{"deepseek-r1-0528", "b200"}, {"deepseek-r1-0528", "b300"}, {"deepseek-r1-0528-fp4-v2", "b200"},
	} {
		in := PoolInput{Deployment: pool(t, 8, "mem_fraction_static: 0.85\n"), Model: graph(t, tc.graph), Chip: chipFile(t, tc.chip)}
		res, err := Resolve(in)
		if err != nil {
			t.Fatalf("%s on %s: %v", tc.graph, tc.chip, err)
		}
		a := res.Args
		if a.AttentionBackend == nil || *a.AttentionBackend != "trtllm_mla" || a.MoeRunnerBackend != "flashinfer_trtllm" ||
			a.CudaGraphConfig.Prefill.Backend != "disabled" {
			t.Errorf("%s on %s: attention %v, runner %q, prefill graph %q", tc.graph, tc.chip, a.AttentionBackend, a.MoeRunnerBackend, a.CudaGraphConfig.Prefill.Backend)
		}
		// The runner is lowered on block-fp8 experts (blis-sglang#135) and on nvfp4 experts (nvfp4_test.go).
		if _, err = Pool(in); err != nil {
			t.Errorf("%s on %s: %v", tc.graph, tc.chip, err)
		}
	}
}

// readAnswersByCase is the model facts of each case of the oracle's answers, by case name.
func readAnswersByCase(t *testing.T) map[string]config.ModelFacts {
	t.Helper()
	b, err := testgz.ReadFile("../config/testdata/serverargs.json")
	if err != nil {
		t.Fatal(err)
	}
	var d struct {
		Cases []struct {
			Name       string            `json:"name"`
			ModelFacts config.ModelFacts `json:"model_facts"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(b, &d); err != nil {
		t.Fatal(err)
	}
	out := map[string]config.ModelFacts{}
	for _, c := range d.Cases {
		out[c.Name] = c.ModelFacts
	}
	return out
}
