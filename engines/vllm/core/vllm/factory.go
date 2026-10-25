// Original to blis-vllm.

package vllm

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/engine"
)

// Kind is the engine kind this module registers with blis-schemas' factory registry.
const Kind = "vllm"

// Version is the vLLM release this engine models.
const Version = "0.32.0"

// CompatVersions are the releases of the scenario files written before this engine existed.
// A scenario stating one runs inference-sim's extracted engine as an explicit compatibility
// behaviour (inference-sim#2021, decision D7; removed by inference-sim#2027), which this
// factory does not build: the simulator wires it in through Compat.
var CompatVersions = []string{"0.29.0", "0.31.0"}

// Factory builds vLLM engines from an opaque Spec, for blis-schemas' registry.
type Factory struct {
	// Pricer prices every engine's steps. Nil refuses to build: an engine must not run at a
	// price nobody chose.
	Pricer func(s engine.Spec) (Pricer, error)
	// Compat builds the compatibility engine for a CompatVersions release; nil refuses those
	// releases with the reason.
	Compat func(s engine.Spec) (engine.Engine, error)
}

// Kind implements engine.Factory.
func (Factory) Kind() string { return Kind }

// New implements engine.Factory: it builds the v0.32.0 engine from the Spec's engine block, a
// compatibility engine for an older release a scenario states, and refuses any other release
// (#2021 B6: one build models one upstream release, and a version is a check, not a selector).
func (f Factory) New(s engine.Spec) (engine.Engine, error) {
	if s.Kind != Kind {
		return nil, fmt.Errorf("vllm: spec kind %q is not %q", s.Kind, Kind)
	}
	switch {
	case s.UpstreamVersion == Version:
	case isCompat(s.UpstreamVersion):
		if f.Compat == nil {
			return nil, fmt.Errorf("vllm: engine_version %s selects the compatibility engine for scenarios "+
				"written before vLLM %s (inference-sim#2021 D7, inference-sim#2027), which this factory "+
				"does not build: set Factory.Compat", s.UpstreamVersion, Version)
		}
		return f.Compat(s)
	default:
		return nil, fmt.Errorf("vllm: this build models vLLM %s; engine_version %q is refused (compatibility "+
			"releases: %v)", Version, s.UpstreamVersion, CompatVersions)
	}
	cfg, err := decodeConfig(s.Config)
	if err != nil {
		return nil, err
	}
	st, err := resolve(cfg)
	if err != nil {
		return nil, err
	}
	if f.Pricer == nil {
		return nil, fmt.Errorf("vllm: Factory.Pricer is nil: the caller prices steps (the latency kernel, " +
			"or fixed durations in a behavioural test)")
	}
	p, err := f.Pricer(s)
	if err != nil {
		return nil, fmt.Errorf("vllm: pricer: %w", err)
	}
	return newEngine(st, p)
}

func isCompat(v string) bool {
	for _, c := range CompatVersions {
		if v == c {
			return true
		}
	}
	return false
}

var _ engine.Factory = Factory{}
