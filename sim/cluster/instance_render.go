package cluster

import "github.com/inference-sim/blis-substrate/contracts/engine"

// ServesRender reports whether this instance's model server answers the router's render
// endpoints. The engine says so (engine.RenderServing); the simulator does not decide it (X3).
// An instance with no engine, or an engine that does not say, serves them: that is what the
// simulated server did before engines could state it.
func (i *InstanceSimulator) ServesRender() bool {
	if i.drv == nil || i.drv.Engine() == nil {
		return true
	}
	if r, ok := i.drv.Engine().(engine.RenderServing); ok {
		return r.ServesRender()
	}
	return true
}
