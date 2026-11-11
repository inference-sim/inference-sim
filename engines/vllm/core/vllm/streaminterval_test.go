package vllm

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
)

// stream_interval is proven inert for the engine (oracle/inertness/stream_interval.json): it sets how many
// tokens the front end's OutputProcessor groups into one delivery, and nothing in the scheduler, the KV
// cache or the pricing reads it. The engine accepts it at every value, and its resolved settings and its
// step-by-step behaviour are identical at the default and at the values recorded runs state. The delivery
// grouping is the simulator's (inference-sim#2035).
func TestStreamIntervalChangesNothingTheEngineDecides(t *testing.T) {
	base, err := resolve(mustConfig(t, ""))
	if err != nil {
		t.Fatal(err)
	}
	arr := arrivals(req("a", 40, 12), req("b", 33, 9), req("c", 20, 7), req("d", 90, 5))
	want := enginetest.TestEngine(t, build(t, ""), arr, enginetest.Options{})
	for _, v := range []int{1, 2, 10, 20, 32} {
		extra := fmt.Sprintf("stream_interval: %d\n", v)
		st, err := resolve(mustConfig(t, extra))
		if err != nil {
			t.Fatalf("stream_interval %d is refused: %v", v, err)
		}
		if !reflect.DeepEqual(st, base) {
			t.Errorf("stream_interval %d changes the resolved settings:\n got %+v\nwant %+v", v, st, base)
		}
		got := enginetest.TestEngine(t, build(t, extra), arr, enginetest.Options{})
		if !reflect.DeepEqual(got.Steps, want.Steps) || !reflect.DeepEqual(got.Finished, want.Finished) {
			t.Errorf("stream_interval %d changes the engine's steps", v)
		}
	}
}
