// SPDX-License-Identifier: Apache-2.0

package lower

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/hardware"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
)

var denseGraphs = []string{
	"llama-3.1-8b-instruct", "llama-3.1-70b-instruct", "qwen2.5-7b-instruct", "qwen3-14b", "mistral-nemo-instruct-2407",
}

// The facts the lowering hands config.Resolve for a dense decoder-only catalog model are the facts
// SGLang derives for the same checkpoint (the cases of oracle/serverargs/cases.json). SGLang names
// no override for these architectures, so the oracle's resolution on them is the check (SGL-1).
func TestDenseModelFactsMatchTheOracle(t *testing.T) {
	a := readAnswersByCase(t)
	for _, tc := range []struct{ graph, oracleCase string }{
		{"llama-3.1-8b-instruct", "llama31-8b-h100-defaults"},
		{"qwen2.5-7b-instruct", "qwen25-7b-h100-defaults"},
		{"qwen3-14b", "qwen3-14b-h100-defaults"},
		{"mistral-nemo-instruct-2407", "nemo-h100-defaults"},
	} {
		want, ok := a[tc.oracleCase]
		if !ok {
			t.Fatalf("the oracle has no case %s", tc.oracleCase)
		}
		got, err := ModelFactsOf(graph(t, tc.graph))
		if err != nil {
			t.Fatalf("%s: %v", tc.graph, err)
		}
		got.HiddenSize, want.HiddenSize = nil, nil
		got.MixedPrecisionMoEQuantAlgos, want.MixedPrecisionMoEQuantAlgos = nil, nil
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s: facts %+v, SGLang derives %+v", tc.graph, got, want)
		}
	}
}

func bareChip(name, cc string) *hardware.Chip {
	return &hardware.Chip{Name: name, ComputeCapability: cc}
}

// A dense bf16 model resolves and lowers on every chip the engine knows the memory of, at SGLang's
// defaults and across tensor-parallel widths.
func TestDenseModelsLowerOnEveryChip(t *testing.T) {
	for _, g := range denseGraphs {
		for _, c := range []struct{ name, cc string }{
			{"h100", "9.0"}, {"h200", "9.0"}, {"b200", "10.0"}, {"b300", "10.3"}, {"gb200-nvl72", "10.0"}, {"gb300", "10.3"},
		} {
			for _, tp := range []int{1, 2, 4, 8} {
				in := PoolInput{Deployment: pool(t, tp, "{}\n"), Model: graph(t, g), Chip: bareChip(c.name, c.cc)}
				if _, err := Pool(in); err != nil {
					t.Errorf("%s on %s at tp %d: %v", g, c.name, tp, err)
				}
			}
		}
	}
}

// A dense graph whose nodes are quantised (the ModelOpt fp8 and fp4 Llama-3.3 checkpoints state a
// weight dtype per node under a bf16 global) carries the quantisation method, which the oracle's
// cases do not cover; it is refused naming itself, never lowered as bf16 (SGL-1).
func TestMixedPrecisionDenseGraphIsRefused(t *testing.T) {
	for _, g := range []string{"llama-3.3-70b-instruct-fp8", "llama-3.3-70b-instruct-fp4"} {
		_, err := Pool(PoolInput{Deployment: pool(t, 4, "{}\n"), Model: graph(t, g), Chip: bareChip("h200", "9.0")})
		var nl *NotLowered
		if !errors.As(err, &nl) || !strings.Contains(err.Error(), "weight_dtype") {
			t.Errorf("%s: got %v, want a NotLowered naming the per-node weight_dtype", g, err)
		}
	}
}

// A dense graph whose global weights are fp8 has no block scale to state, so it is refused too.
func TestFP8DenseGraphIsRefused(t *testing.T) {
	g := graph(t, "llama-3.1-8b-instruct")
	g.Global.WeightDType = model.DTypeFP8
	_, err := Pool(PoolInput{Deployment: pool(t, 1, "{}\n"), Model: g, Chip: bareChip("h200", "9.0")})
	var nl *NotLowered
	if !errors.As(err, &nl) {
		t.Errorf("an fp8 dense graph: got %v, want NotLowered", err)
	}
}
