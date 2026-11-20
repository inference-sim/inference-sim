package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
)

// stand-ins for the vLLM engine's Engine and DPGroup, which report their KV cache size by a method that
// returns core/kv.Capacity: a struct with Tokens, Concurrency and BlocksPerRequest.
type fakeCapacity struct {
	Tokens           int64
	Concurrency      float64
	BlocksPerRequest int64
}
type reportsCapacity struct {
	engine.Engine
	c fakeCapacity
}

func (r reportsCapacity) KVCacheCapacity() fakeCapacity { return r.c }

type reportsCapacityPtr struct {
	engine.Engine
	c *fakeCapacity
}

func (r reportsCapacityPtr) KVCacheCapacity() *fakeCapacity { return r.c }

type reportsWrongShape struct{ engine.Engine }

func (reportsWrongShape) KVCacheCapacity() int64 { return 42 }

// The Result reports the engine's own figure when it states one: vLLM's start-up figure for the
// 208,811-token V4-Pro example of inference-sim#1969 (19,848 blocks / 219 blocks per request x
// 2,304), not the pool's blocks x block size.
func TestReportedKVTokensReadsTheEnginesOwnFigure(t *testing.T) {
	c := fakeCapacity{Tokens: 208811, Concurrency: 90.63013698630137, BlocksPerRequest: 219}
	if got, ok := reportedKVTokens(reportsCapacity{c: c}); !ok || got != 208811 {
		t.Errorf("reportedKVTokens = %d, %v; want 208811, true", got, ok)
	}
	if got, ok := reportedKVTokens(reportsCapacityPtr{c: &c}); !ok || got != 208811 {
		t.Errorf("through a pointer: %d, %v; want 208811, true", got, ok)
	}
}

// An engine that reports nothing, a figure of zero, a nil engine or a method of another shape
// leaves the caller on the pool's blocks x block size.
func TestReportedKVTokensIsAbsentWhenTheEngineStatesNone(t *testing.T) {
	cases := map[string]engine.Engine{
		"nil":             nil,
		"no method":       struct{ engine.Engine }{},
		"zero tokens":     reportsCapacity{c: fakeCapacity{}},
		"nil pointer":     reportsCapacityPtr{},
		"another shape":   reportsWrongShape{},
		"negative tokens": reportsCapacity{c: fakeCapacity{Tokens: -1}},
	}
	for name, e := range cases {
		if got, ok := reportedKVTokens(e); ok || got != 0 {
			t.Errorf("%s: reportedKVTokens = %d, %v; want 0, false", name, got, ok)
		}
	}
}
