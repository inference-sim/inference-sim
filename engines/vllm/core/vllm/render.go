// Original to blis-vllm.

package vllm

// ServesRender implements engine.RenderServing: whether the model server answers the router's
// render endpoints. vLLM answers them only when the server runs with --enable-scale-out
// (engine.enable_scale_out); otherwise each returns HTTP 403 (inference-sim#2129).
func (e *Engine) ServesRender() bool { return e.servesRender }

// ServesRender implements engine.RenderServing for the group: one server fronts all ranks.
func (g *DPGroup) ServesRender() bool { return g.servesRender }
