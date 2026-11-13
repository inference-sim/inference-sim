// Original to blis-vllm.

package vllm

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// Frontend is vLLM's API-server settings that decide how requests queue before the engine admits
// them. blis-schemas states neither yet (blis-vllm#85, #105), so the host that builds the engine
// gives them with SetFrontend and an unset engine has vLLM's defaults.
//
// vLLM v0.32.0 runs the tokenizer and the chat template on the renderer's thread pool,
// ThreadPoolExecutor(max_workers=renderer_num_workers), default 1 (vllm/renderers/base.py:139-158,
// vllm/renderers/hf.py:965, vllm/config/model.py:408). Each API-server process has its own pool, and
// the process count is api_server_count (vllm/entrypoints/cli/serve.py:107-137).
type Frontend struct {
	// RendererNumWorkers is renderer_num_workers; 0 gives no queue, a request waiting only its
	// delay.
	RendererNumWorkers int
	// APIServerCount is api_server_count; 0 is the default for the pool: the data-parallel size
	// behind the internal balancer, else 1.
	APIServerCount int
}

// DefaultFrontend is vLLM's default: one render thread, the per-mode API-server count.
func DefaultFrontend() Frontend { return Frontend{RendererNumWorkers: 1} }

// Validate refuses a negative count.
func (f Frontend) Validate() error {
	if f.RendererNumWorkers < 0 || f.APIServerCount < 0 {
		return fmt.Errorf("vllm: renderer_num_workers %d and api_server_count %d: neither is negative",
			f.RendererNumWorkers, f.APIServerCount)
	}
	return nil
}

// queue is the admission queue the settings state for an endpoint of dp ranks behind one balancer:
// an unstated api_server_count is the data-parallel size (serve.py:123-130).
func (f Frontend) queue(dp int) plan.AdmissionQueue {
	servers := f.APIServerCount
	if servers == 0 {
		servers = max(dp, 1)
	}
	return plan.AdmissionQueue{Servers: servers, WorkersPerServer: f.RendererNumWorkers}
}

// SetFrontend states the API-server settings the engine's plan reports as its admission queue.
func (e *Engine) SetFrontend(f Frontend) error {
	if err := f.Validate(); err != nil {
		return err
	}
	e.frontend, e.dp = f, 1
	return nil
}

// SetFrontend states the API-server settings for the whole group: its endpoint has one front end
// ahead of the balancer, so an unstated api_server_count is the group's size.
func (g *DPGroup) SetFrontend(f Frontend) error {
	if err := f.Validate(); err != nil {
		return err
	}
	g.ranks[0].e.frontend, g.ranks[0].e.dp = f, len(g.ranks)
	return nil
}

// Frontender is what a host asks of an engine to state its API-server settings.
type Frontender interface {
	SetFrontend(Frontend) error
}

var (
	_ Frontender = (*Engine)(nil)
	_ Frontender = (*DPGroup)(nil)
)
