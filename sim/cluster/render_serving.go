package cluster

import "github.com/inference-sim/inference-sim/substrate/contracts/engine"

// ServesRender is whether the instance's model server answers the router's render endpoints,
// as its engine states it (engine.RenderServing, inference-sim#2129). An engine that does not
// state it is taken to serve them.
func (i *InstanceSimulator) ServesRender() bool {
	if i.drv == nil {
		return true
	}
	if r, ok := i.drv.Engine().(engine.RenderServing); ok {
		return r.ServesRender()
	}
	return true
}
