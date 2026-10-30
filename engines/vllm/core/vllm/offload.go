// Original to blis-vllm.

package vllm

import (
	"fmt"
	"sort"

	"github.com/inference-sim/blis-schemas/engine"

	"github.com/inference-sim/blis-vllm/core/connector/offloading"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// KV offload through engine.Engine, with vLLM's OffloadingConnector (core/connector/offloading)
// on the scheduler and its CPU tier.
//
// Each schedule's connector metadata lists the step's load and store jobs. The engine reports
// each job as a Transfer among the batch's TransfersStarted (a load into the device cache, a
// store out of it, both Tier "cpu"), and the simulator prices it and reports its end with
// TransferDone. The outcome reaches the scheduler with the next applied output, as vLLM's
// workers report completed jobs in the next step's kv_connector_worker_meta and a completed
// load's request in finished_recving (offloading/worker.py:273-309@5df63a72). Nothing waits on
// a store: vLLM's worker submits it at the start of the next step and runs it beside the
// forward passes (prepare_store_kv, worker.py:240-249), so a store in flight holds no step open.
// The engine only steps again for it, as vLLM's loop does (has_pending_push_work keeps
// has_requests true, vllm/v1/core/sched/scheduler.py:2823-2838), when its completion is due.
//
// Two of the worker's protocol points are the simulator's to time: jobs_to_flush, the in-flight
// jobs a step's metadata tells the worker to finish before the step runs (a preempted
// request's, or one reading a block the step reallocated; handle_preemptions,
// worker.py:198-224), are reported done with that step's output, whatever the simulator's
// price, since vLLM's worker waits for them there; and every worker of the engine's world size
// reports a job in the same step (the one-transfer-per-job price is the slowest worker's).

// offloadConfig is the engine block's kv_offload: the offload connector and its CPU tier, as
// kv_connector_extra_config states them for the OffloadingConnector
// (vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py:98-134,
// vllm/v1/kv_offload/cpu/spec.py:96-139@5df63a72).
type offloadConfig struct {
	// Connector is kv_connector: OffloadingConnector.
	Connector string `yaml:"connector"`
	// Spec is spec_name: CPUOffloadingSpec (vLLM's default) or TieringOffloadingSpec with no
	// secondary tiers.
	Spec string `yaml:"spec,omitempty"`
	// CPUBytes is cpu_bytes_to_use, required (cpu/spec.py:97-101).
	CPUBytes int64 `yaml:"cpu_bytes"`
	// BlockSize is block_size, the CPU chunk's tokens, a multiple of the GPU block size; 0 is one
	// GPU block a chunk (config.py:98-134).
	BlockSize int `yaml:"block_size,omitempty"`
	// EvictionPolicy is eviction_policy, lru unstated (cpu/spec.py:131).
	EvictionPolicy string `yaml:"eviction_policy,omitempty"`
	// OffloadPromptOnly is offload_prompt_only, true unstated (vllm/v1/kv_offload/base.py:617-623).
	OffloadPromptOnly *bool `yaml:"offload_prompt_only,omitempty"`
	// BytesPerBlock is one worker's KV bytes for one GPU block over the offloaded groups
	// (worker_kv_bytes_per_block, config.py:136-151): the caller's memory model states it, as it
	// states num_gpu_blocks. With PageSize, the host's mmap.PAGESIZE (4096 unstated), and Copies
	// it sizes the CPU tier in chunks (cpu/spec.py:103-121).
	BytesPerBlock int64 `yaml:"bytes_per_block"`
	PageSize      int64 `yaml:"page_size,omitempty"`
	// Workers is the engine's world size, the connector's num_workers (offloading/scheduler.py:335):
	// every job is pending on each worker. 1 unstated.
	Workers int `yaml:"workers,omitempty"`
	// Copies is how many workers' copies of a block the CPU tier holds: the world size, or 1 for
	// the replicated latent layout (num_copies, cpu/spec.py:106-109; offloading/config.py:158-171).
	// Workers unstated. The oracle runs num_copies = world size only (its CPU platform has no
	// shared region, cpu/spec.py:161-166); 1 changes only the chunk count.
	Copies int `yaml:"copies,omitempty"`
}

// offload is an engine's KV offload side: its connector, the jobs in flight as transfers, and
// the outcome the next applied output reports.
type offload struct {
	conn      *offloading.Scheduler
	blockSize int
	next      engine.TransferID
	// early are the transfers of flushed jobs already reported complete: their TransferDone is
	// accepted and changes nothing.
	early map[engine.TransferID]bool
	// jobs are the transfers in flight, by transfer, with the job each is; flushed are the jobs a
	// schedule told the worker to finish, which the next applied output reports done.
	jobs    map[engine.TransferID]int
	ids     map[int]engine.TransferID
	flushed map[int]bool
	// completed are the jobs completed since the last applied output, and recv the requests whose
	// load completed.
	completed []int
	recv      map[string]bool
	// loadReq are the loads in flight, by job, with their request.
	loadReq map[int]string
	// lastFlush are the last schedule's jobs_to_flush, for a test.
	lastFlush []int
}

// buildOffload installs the OffloadingConnector on the scheduler, refusing what it does not
// model. Each KV cache group is offloaded at its block size, a sliding window with the
// extra_retained_tokens the drafter's prefill lookahead sets (get_kv_cache_configs,
// vllm/v1/core/kv_cache_utils.py:2582-2589@5df63a72).
func buildOffload(s *sched.Scheduler, c *offloadConfig, spec sched.Spec, prefixCaching bool) (*offload, error) {
	if c.Connector != "OffloadingConnector" {
		return nil, fmt.Errorf("vllm: kv_offload.connector %q is not modelled through the engine (modelled: "+
			"OffloadingConnector; registry/connectors.yaml)", c.Connector)
	}
	if c.CPUBytes <= 0 {
		return nil, fmt.Errorf("vllm: kv_offload.cpu_bytes must be stated: vLLM requires cpu_bytes_to_use " +
			"(vllm/v1/kv_offload/cpu/spec.py:97-101@5df63a72)")
	}
	if c.BytesPerBlock <= 0 {
		return nil, fmt.Errorf("vllm: kv_offload.bytes_per_block must be stated: one worker's KV bytes a GPU block, " +
			"which sizes the CPU tier in chunks (offloading/config.py:136-151@5df63a72); the caller's memory model states it")
	}
	if err := offloading.Check(offloading.Options{SpecName: c.Spec, EvictionPolicy: c.EvictionPolicy}); err != nil {
		return nil, fmt.Errorf("vllm: %w", err)
	}
	bs := s.BlockSize()
	bpc := 1
	if c.BlockSize != 0 {
		if c.BlockSize < 0 || c.BlockSize%bs != 0 {
			return nil, fmt.Errorf("vllm: kv_offload.block_size %d must be a multiple of the GPU block size %d "+
				"(offloading/config.py:121-134@5df63a72)", c.BlockSize, bs)
		}
		bpc = c.BlockSize / bs
	}
	var groups []offloading.Group
	extra := max(0, spec.PrefillLookahead-1)
	for g := 0; g < s.KV.NumGroups(); g++ {
		spec := s.KV.GroupSpec(g)
		gb := s.KV.GroupBlockSize(g)
		gr := offloading.Group{BlockSize: gb}
		if spec.Window > 0 {
			gr.Window, gr.ExtraRetained = spec.Window, extra
		}
		groups = append(groups, gr)
	}
	page := c.PageSize
	if page == 0 {
		page = 4096
	}
	promptOnly := c.OffloadPromptOnly == nil || *c.OffloadPromptOnly
	world := max(c.Workers, 1)
	copies := c.Copies
	if copies == 0 {
		copies = world
	}
	if copies != 1 && copies != world {
		return nil, fmt.Errorf("vllm: kv_offload.copies %d: the CPU tier holds one copy of a block per worker (%d) or "+
			"one for the replicated latent layout (cpu/spec.py:106-109@5df63a72)", copies, world)
	}
	chunks := offloading.NumChunks(c.CPUBytes, c.BytesPerBlock, copies, bpc, page)
	if chunks < 1 {
		return nil, fmt.Errorf("vllm: kv_offload.cpu_bytes %d holds no chunk of %d block(s) of %d bytes for %d "+
			"worker copies (cpu/spec.py:107-121@5df63a72)", c.CPUBytes, bpc, c.BytesPerBlock, copies)
	}
	conn, err := offloading.New(offloading.Config{Groups: groups, BlocksPerChunk: bpc, TokensPerHash: s.KV.HashBlockSize(),
		NumChunks: chunks, OffloadPromptOnly: promptOnly, PrefixCaching: prefixCaching, Workers: world})
	if err != nil {
		return nil, fmt.Errorf("vllm: %w", err)
	}
	o := &offload{conn: conn, blockSize: bs, early: map[engine.TransferID]bool{}, jobs: map[engine.TransferID]int{}, ids: map[int]engine.TransferID{},
		flushed: map[int]bool{}, recv: map[string]bool{}, loadReq: map[int]string{}}
	// kv_both: the CPU tier is both stored to and loaded from.
	if err := s.SetConnector(conn, sched.ConnectorConfig{Consumer: true, Producer: true}); err != nil {
		return nil, fmt.Errorf("vllm: %w", err)
	}
	return o, nil
}

// started reports the jobs a schedule's metadata lists as transfers: the step's loads, then its
// stores, each in job order. The jobs it tells the worker to flush complete with the step.
func (o *offload) started(cb any) []engine.Transfer {
	if o == nil {
		return nil
	}
	m, ok := cb.(offloading.Meta)
	if !ok {
		return nil
	}
	var out []engine.Transfer
	add := func(j offloading.Job, kind engine.TransferKind) {
		o.next++
		o.jobs[o.next], o.ids[j.ID] = j.ID, o.next
		// Tokens are the widest group's blocks: every group is at one block size (offloading.New),
		// and a full-attention group moves every block of the range, a window only its own.
		widest := 0
		for _, n := range j.GroupSizes {
			widest = max(widest, n)
		}
		out = append(out, engine.Transfer{ID: o.next, Request: j.Req, Kind: kind, Tier: "cpu",
			Tokens: widest * o.blockSize, Blocks: len(j.GPU)})
	}
	for _, j := range m.Loads {
		o.loadReq[j.ID] = j.Req
		add(j, engine.TransferLoad)
	}
	for _, j := range m.Stores {
		add(j, engine.TransferStore)
	}
	for _, j := range m.Flush {
		o.flushed[j] = true
	}
	o.lastFlush = append(o.lastFlush[:0], m.Flush...)
	return out
}

// done records a transfer's end for the next applied output. A failed transfer is refused: vLLM
// v0.32.0's offloading worker asserts every job succeeds (worker.py:293-294@5df63a72).
func (o *offload) done(id engine.TransferID, ok bool) error {
	if o == nil {
		return engine.ErrUnknownTransfer
	}
	if o.early[id] {
		delete(o.early, id)
		return nil
	}
	j, known := o.jobs[id]
	if !known {
		return engine.ErrUnknownTransfer
	}
	if !ok {
		return fmt.Errorf("vllm: offload transfer %d failed, which vLLM's offloading worker does not support "+
			"(it asserts success, offloading/worker.py:293-294@5df63a72; %s)", id, offloading.Issue)
	}
	o.complete(j)
	return nil
}

func (o *offload) complete(j int) {
	id := o.ids[j]
	delete(o.jobs, id)
	delete(o.ids, j)
	delete(o.flushed, j)
	o.completed = append(o.completed, j)
	if r, isLoad := o.loadReq[j]; isLoad {
		delete(o.loadReq, j)
		o.recv[r] = true
	}
}

// take is the outcome the workers report with the applied output, once: the jobs completed since
// the last, flushed ones included, each by every worker, in job order, and the requests whose
// load completed, by request ID (the transport's order, oracle/mocks.json).
func (o *offload) take() (jobs []offloading.Completion, recv []string) {
	if o == nil {
		return nil, nil
	}
	for j := range o.flushed {
		if id, inFlight := o.ids[j]; inFlight {
			o.early[id] = true
			o.complete(j)
		}
	}
	o.flushed = map[int]bool{}
	sort.Ints(o.completed)
	for _, j := range o.completed {
		jobs = append(jobs, offloading.Completion{Job: j, Workers: o.conn.Workers()})
	}
	for r := range o.recv {
		recv = append(recv, r)
	}
	sort.Strings(recv)
	o.completed, o.recv = nil, map[string]bool{}
	return jobs, recv
}
