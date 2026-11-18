// Original to blis-sglang.

// Package sglangengine is the factory of the "sglang" engine kind: it builds core/sglang's Engine
// from a configuration. Everything that decides how the engine behaves is in core/sglang; this
// package only connects it to a simulator's registry (blis-substrate contracts/engine/spec.go).
package sglangengine

import (
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
