// Original to blis-vllm.

package vllm

import (
	"sync"

	"github.com/inference-sim/blis-schemas/engine"
)

var registerOnce sync.Once

// Register makes the "vllm" kind available in blis-schemas' engine registry, with the given
// factory's pricer and compatibility engine. It is the one place this module registers itself,
// called by the simulator's one build site rather than from an init function, so importing this
// package registers nothing behind the caller's back. Later calls are no-ops.
func Register(f Factory) {
	registerOnce.Do(func() { engine.Register(f) })
}
