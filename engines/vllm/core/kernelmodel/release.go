// Original to blis-vllm.

package kernelmodel

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
)

// engineKind is the engine kind blis-vllm registers, as a scenario's engine.kind names it.
const engineKind = "vllm"

// EngineRelease is the vLLM release a scenario names: engine.version, or the deprecated
// engine_version when engine is not stated (contracts Scenario.EngineRelease). Until M4 removes
// engine_version (plan §3.3), a scenario may state it alone, or beside engine with the same
// release; one stating both, unequal, is refused, as contracts' scenario validation refuses it.
// A scenario naming another engine kind is refused: it is not this engine's to run (SIM-1).
func EngineRelease(sc *scenario.Scenario) (string, error) {
	if e := sc.Engine; e != nil {
		if e.Kind != engineKind {
			return "", fmt.Errorf("engine.kind is %q: blis-vllm runs engine kind %q", e.Kind, engineKind)
		}
		if sc.EngineVersion != "" && sc.EngineVersion != e.Version {
			return "", fmt.Errorf("engine_version is %q but engine.version is %q: engine_version is "+
				"deprecated; state the release once, in engine.version", sc.EngineVersion, e.Version)
		}
	}
	return sc.EngineRelease(), nil
}
