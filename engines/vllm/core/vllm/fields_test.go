// Original to blis-vllm.

package vllm

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/spec/fields"

	blisvllm "github.com/inference-sim/blis-vllm"
)

// The engine reports the list its release publishes: the one fields.json holds, for the engine
// kind and the release this build models, from an engine and from a data-parallel group alike
// (design VLM-3). The contract's own law (enginetest.CheckFields) holds for both.
func TestTheEngineReportsTheFieldListItPublishes(t *testing.T) {
	want, err := fields.Parse(blisvllm.FieldListJSON)
	if err != nil {
		t.Fatal(err)
	}
	for name, extra := range map[string]string{
		"one engine":            "",
		"a data-parallel group": "data_parallel_size: 2\n",
		"a speculating engine":  "speculative_config: {method: eagle3, num_speculative_tokens: 3}\n",
	} {
		t.Run(name, func(t *testing.T) {
			e, err := build(t, extra)()
			if err != nil {
				t.Fatal(err)
			}
			got := e.Fields()
			if !reflect.DeepEqual(got, *want) {
				t.Errorf("Fields() is not fields.json: engine %+v, %d fields, want %d", got.Engine, len(got.Fields), len(want.Fields))
			}
			if got.Engine.Kind != blisvllm.Kind || got.Engine.Version != blisvllm.Upstream.Version {
				t.Errorf("the list names %+v, and this build models %s %s", got.Engine, blisvllm.Kind, blisvllm.Upstream.Version)
			}
			if err := enginetest.CheckFields(e); err != nil {
				t.Error(err)
			}
		})
	}
}
