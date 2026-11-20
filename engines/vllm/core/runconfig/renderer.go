// Moved from inference-sim/sim/engine/renderer.go at ca532abca209af41425c007974a5f3ef336f5981.

package runconfig

import "fmt"

// FrontendConfig is the API-server front end that renders and tokenizes each request before
// the engine sees it (INV-R1). Its zero value is inert: no render queue, and a request's
// admission overhead is a per-request delay with no contention, byte-identical to a build
// without it (INV-6).
//
// vLLM v0.31.0 renders the chat template and tokenizes on the API server through the
// renderer's thread pool (BaseRenderer._executor, ThreadPoolExecutor(max_workers=
// renderer_num_workers), vllm/renderers/base.py:117-133; the template step of the V4
// renderer on the same pool, vllm/renderers/deepseek_v4.py:32-37). The pool belongs to one
// API-server process, so --api-server-count N runs N independent pools.
type FrontendConfig struct {
	// RendererNumWorkers is vLLM's renderer_num_workers (vllm/config/model.py:378, default 1):
	// the render threads of one API server. 0 disables the render queue.
	RendererNumWorkers int
	// APIServerCount is vLLM's --api-server-count for one endpoint. 0 is vLLM's per-mode
	// default (vllm/entrypoints/cli/serve.py:105-127): the data-parallel size behind an
	// internal-balancer replica endpoint, else 1.
	APIServerCount int
}

// Enabled reports whether requests queue for a render thread.
func (c FrontendConfig) Enabled() bool { return c.RendererNumWorkers > 0 }

// Validate rejects negative counts.
func (c FrontendConfig) Validate() error {
	if c.RendererNumWorkers < 0 {
		return fmt.Errorf("renderer_num_workers must be >= 0, got %d", c.RendererNumWorkers)
	}
	if c.APIServerCount < 0 {
		return fmt.Errorf("api_server_count must be >= 0, got %d", c.APIServerCount)
	}
	return nil
}

// ReplicaAPIServers is how many API servers render a data-parallel replica's requests before
// its balancer picks a rank: the stated --api-server-count, else vLLM's default for a replica
// endpoint behind the internal or hybrid balancer, its data-parallel size
// (vllm/entrypoints/cli/serve.py:107-129@4ba787ec). Moved from inference-sim's cluster
// (sim/cluster/dp_replica.go at 972b4abebc9d84130692e349b75da66c51781e16).
func ReplicaAPIServers(stated, dp int) int {
	if stated == 0 {
		return dp
	}
	return stated
}
