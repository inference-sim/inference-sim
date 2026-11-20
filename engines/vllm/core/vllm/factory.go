// Original to blis-vllm.

package vllm

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine"

	"github.com/inference-sim/blis-vllm/internal/upstream"
)

// Kind is the engine kind this module registers with blis-schemas' factory registry.
const Kind = "vllm"

// Version is the vLLM release this engine models (internal/upstream).
const Version = upstream.Version

// Factory builds vLLM engines from an opaque Spec, for blis-schemas' registry.
type Factory struct {
	// Pricer prices every engine's steps. Nil refuses to build: an engine must not run at a
	// price nobody chose.
	Pricer func(s engine.Spec) (Pricer, error)
	// Prepare, when set, completes a v0.33.0 spec before its engine block is decoded: a
	// production caller fills in what the block does not state and the cost model knows (the KV
	// block count, the pool's pipeline_parallel_size, the model runner), or the whole block from
	// the scenario. Nil decodes the spec as given.
	Prepare func(s engine.Spec) (engine.Spec, error)
	// Open, when set, replaces Prepare and Pricer for a v0.33.0 spec: it completes the spec as
	// Prepare does and returns the function that makes the spec's pricer, which New calls once the
	// engine block has resolved. A caller whose two steps read the same data (vllmengine opens a
	// pool, and with it the catalog and registry) opens it once here, so one build has one
	// report and pays for one open. New reads Prepare and Pricer only when Open is nil.
	Open func(s engine.Spec) (engine.Spec, func() (Pricer, error), error)
}

// Kind implements engine.Factory.
func (Factory) Kind() string { return Kind }

// New implements engine.Factory: it builds the v0.33.0 engine from the Spec's engine block and
// refuses any other release (#2021 B6: one build models one upstream release, and a version is a
// check, not a selector).
func (f Factory) New(s engine.Spec) (engine.Engine, error) {
	if s.Kind != Kind {
		return nil, fmt.Errorf("vllm: spec kind %q is not %q", s.Kind, Kind)
	}
	if s.UpstreamVersion != Version {
		return nil, fmt.Errorf("vllm: this build models vLLM %s; engine_version %q is refused",
			Version, s.UpstreamVersion)
	}
	pricer := func() (Pricer, error) {
		if f.Pricer == nil {
			return nil, fmt.Errorf("vllm: Factory.Pricer is nil: the caller prices steps (the latency kernel, " +
				"or fixed durations in a behavioural test)")
		}
		p, err := f.Pricer(s)
		if err != nil {
			return nil, fmt.Errorf("vllm: pricer: %w", err)
		}
		return p, nil
	}
	switch {
	case f.Open != nil:
		var err error
		if s, pricer, err = f.Open(s); err != nil {
			return nil, err
		}
	case f.Prepare != nil:
		var err error
		if s, err = f.Prepare(s); err != nil {
			return nil, err
		}
	}
	cfg, err := decodeConfig(s.Config)
	if err != nil {
		return nil, err
	}
	st, err := resolve(cfg)
	if err != nil {
		return nil, err
	}
	p, err := pricer()
	if err != nil {
		return nil, err
	}
	if st.dp.ranks > 1 {
		return newDPGroup(st, p)
	}
	return newEngine(st, p)
}

var _ engine.Factory = Factory{}
