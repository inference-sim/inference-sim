// Original to blis-vllm.

package kernelmodel

import (
	"fmt"

	vllmrules "github.com/inference-sim/inference-sim/engines/vllm/core/rulepack"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/scenario"
)

// engineKind is the engine kind blis-vllm registers, as a scenario's engine.kind names it.
const engineKind = "vllm"

// EngineRelease is the vLLM release the build runs for a scenario, which a scenario never
// states (LIF-6): the one the engine's rules are written for. A scenario naming another engine
// kind is refused: it is not this engine's to run (SIM-1).
func EngineRelease(sc *scenario.Scenario) (string, error) {
	if e := sc.Engine; e != nil {
		if e.Kind != engineKind {
			return "", fmt.Errorf("engine.kind is %q: blis-vllm runs engine kind %q", e.Kind, engineKind)
		}
	}
	if r := sc.EngineRelease(); r != "" {
		return r, nil
	}
	if vs := vllmrules.Versions(); len(vs) == 1 {
		return vs[0], nil
	}
	return "", nil
}
