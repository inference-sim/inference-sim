// Original to blis-sglang.

// Package sglangengine is the factory of the "sglang" engine kind: it builds core/sglang's Engine
// from a configuration. Everything that decides how the engine behaves is in core/sglang; this
// package only connects it to a simulator's registry (blis-substrate contracts/engine/spec.go).
package sglangengine

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine"

	blissglang "github.com/inference-sim/blis-sglang"
	"github.com/inference-sim/blis-sglang/core/sched"
	"github.com/inference-sim/blis-sglang/core/sglang"
)

// Kind is the engine kind this package registers.
const Kind = "sglang"

// Input is what an engine is built from once its configuration is resolved: the scheduler's
// settings as SGLang resolved them and the cost model that prices its passes.
type Input struct {
	Scheduler sched.Config
	Pricer    sglang.Pricer
}

// New builds the engine from resolved settings.
func New(in Input) (engine.Engine, error) {
	list, err := blissglang.Fields()
	if err != nil {
		return nil, err
	}
	return sglang.New(in.Scheduler, in.Pricer, list)
}

// Factory builds engines of kind "sglang" from a deployment document's engine block.
func Factory() engine.Factory { return factory{} }

type factory struct{}

func (factory) Kind() string { return Kind }

// New refuses a release other than the one this build models, and names what a document-built
// engine still needs: SGLang's ServerArgs resolution (config/, S08, inference-sim/blis-sglang#50)
// and the pool and cost model lowering (S17, inference-sim/blis-sglang#49). Until they are merged
// an engine is built with New(Input).
func (factory) New(s engine.Spec) (engine.Engine, error) {
	if s.UpstreamVersion != "" && s.UpstreamVersion != blissglang.Upstream.Version {
		return nil, fmt.Errorf("sglangengine: this build models SGLang %s, not %s", blissglang.Upstream.Version, s.UpstreamVersion)
	}
	return nil, fmt.Errorf("sglangengine: building an engine from a document needs ServerArgs resolution (inference-sim/blis-sglang#50) and the pool and cost lowering (inference-sim/blis-sglang#49), which are not merged; use sglangengine.New with resolved settings")
}

// Register makes the factory available under its kind.
func Register() { engine.Register(Factory()) }
