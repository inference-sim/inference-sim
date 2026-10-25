package sim

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

// Renderer is one endpoint's render stage (INV-R1): apiServers API servers, each a FIFO
// queue served by its own workers render threads. A request's render takes the latency
// model's QueueingTime, which the kernel prices (AdmissionOverhead); the Renderer only
// decides when that work can start.
//
// Requests reach the API servers round-robin in arrival order. vLLM's servers share one
// listening socket and the OS hands each connection to one of them; round-robin is BLIS's
// deterministic stand-in for that assignment.
//
// Within one server a request takes the earliest-free thread, in arrival order (FIFO, an
// M/G/c queue). vLLM submits a request's template and tokenize steps to the pool as two jobs,
// so another request's job can run between them; the template step is ~0.2% of the tokenize
// step (diagnosis bench_frontend), so whole-request FIFO is the model.
type Renderer struct {
	// freeAt[s][w] is when server s's thread w finishes its last render.
	freeAt [][]int64
	next   int
	// rendered is the number of requests admitted to a thread; waitTicks their summed wait.
	rendered  int64
	waitTicks int64
}

// NewRenderer builds a render stage of apiServers servers with workers threads each.
func NewRenderer(apiServers, workers int) (*Renderer, error) {
	if apiServers < 1 || workers < 1 {
		return nil, fmt.Errorf("NewRenderer: need >= 1 API server and >= 1 render thread, got %d and %d",
			apiServers, workers)
	}
	r := &Renderer{freeAt: make([][]int64, apiServers)}
	for s := range r.freeAt {
		r.freeAt[s] = make([]int64, workers)
	}
	return r, nil
}

// Admit queues a request that reached the endpoint at arrival and whose render takes service
// ticks, and returns when its render finishes: max(arrival, the thread's free time) + service.
// Callers admit in arrival order (the event loop's), which makes each server FIFO.
func (r *Renderer) Admit(arrival, service int64) int64 {
	threads := r.freeAt[r.next]
	r.next = (r.next + 1) % len(r.freeAt)
	w := 0
	for k := 1; k < len(threads); k++ {
		if threads[k] < threads[w] {
			w = k
		}
	}
	start := max(arrival, threads[w])
	done := start + service
	threads[w] = done
	r.rendered++
	r.waitTicks += start - arrival
	return done
}

// Rendered is the number of requests admitted so far.
func (r *Renderer) Rendered() int64 { return r.rendered }

// WaitTicks is the summed time requests waited for a render thread.
func (r *Renderer) WaitTicks() int64 { return r.waitTicks }

// ShareRenderer gives the ranks of one endpoint a single render stage of apiServers servers:
// the replica behind vLLM's internal DP balancer, whose API servers render every request
// before the balancer picks a rank. A no-op when the ranks' front end is inert.
func ShareRenderer(ranks []*EngineCore, apiServers int) error {
	if len(ranks) == 0 || ranks[0].renderer == nil {
		return nil
	}
	workers := len(ranks[0].renderer.freeAt[0])
	r, err := NewRenderer(apiServers, workers)
	if err != nil {
		return err
	}
	for _, s := range ranks {
		if s.renderer == nil || len(s.renderer.freeAt[0]) != workers {
			return fmt.Errorf("ShareRenderer: the ranks of one endpoint must all render, with the same thread count")
		}
		s.renderer = r
	}
	return nil
}

// Renderer is the simulator's render stage, nil when its front end is inert.
func (sim *EngineCore) Renderer() *Renderer { return sim.renderer }
