package vllm

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"
)

// cpuTier is an OffloadingConnector beside the base block's 64 GPU blocks of 16 tokens: a CPU
// tier of 200 chunks of one 4 KiB block each.
const cpuTier = "kv_offload: {connector: OffloadingConnector, cpu_bytes: 819200, bytes_per_block: 4096}\n"

// tightGPU leaves the GPU prefix cache too small to keep one request's prompt past the next's.
const tightGPU = "num_gpu_blocks: 12\nmax_model_len: 160\n"

// An offloading engine keeps the engine contract's laws (blis-schemas enginetest), stores and
// loads included, at one batch in flight, with several workers, a hybrid model's groups, a chunk
// of two blocks and prompt and decode tokens offloaded.
func TestOffloadConformance(t *testing.T) {
	for _, c := range []struct {
		name, extra string
		loads       bool
	}{
		{"cpu tier", cpuTier, false},
		{"tight gpu", cpuTier + tightGPU, true},
		{"tp4 workers", "kv_offload: {connector: OffloadingConnector, cpu_bytes: 3276800, bytes_per_block: 4096, workers: 4}\n" + tightGPU, true},
		{"chunk of two blocks, decode stored", "kv_offload: {connector: OffloadingConnector, cpu_bytes: 819200, " +
			"bytes_per_block: 4096, block_size: 32, offload_prompt_only: false}\n" + tightGPU, true},
		{"tiny cpu tier", "kv_offload: {connector: OffloadingConnector, cpu_bytes: 16384, bytes_per_block: 4096}\n" + tightGPU, false},
		{"hybrid", cpuTier + "num_gpu_blocks: 24\nmax_model_len: 160\nkv_layers:\n" +
			"- {count: 1, spec: full, num_kv_heads: 1, head_size: 64}\n" +
			"- {count: 1, spec: sliding_window, sliding_window: 32, num_kv_heads: 1, head_size: 64}\n", true},
		// The batch queue: async scheduling (two batches in flight), PP2 on the V2 runner (three),
		// and a drafter beside it.
		{"async", cpuTier + tightGPU + "async_scheduling: true\n", true},
		{"pp2 async v2", "kv_offload: {connector: OffloadingConnector, cpu_bytes: 1638400, bytes_per_block: 4096, workers: 2}\n" +
			tightGPU + "async_scheduling: true\nuse_v2_model_runner: true\npipeline_parallel_size: 2\n", true},
		{"async eagle3", cpuTier + tightGPU + "async_scheduling: true\nspeculative: {method: eagle3, num_spec_tokens: 3}\n", true},
		{"sync eagle3", cpuTier + tightGPU + "speculative: {method: eagle3, num_spec_tokens: 3}\n", true},
	} {
		t.Run(c.name, func(t *testing.T) {
			// Three prompts, each returning after others of new tokens have passed through the
			// tight GPU cache, so the CPU tier serves them back.
			var arr []enginetest.Arrival
			at := engine.Time(0)
			for i, id := range []string{"a", "b", "a2", "c", "b2", "d", "a3", "e"} {
				base, n := 100000*(i+1), 100+4*i // a fresh prompt
				switch id[0] {
				case 'a':
					base, n = 1000, 96
				case 'b':
					base, n = 2000, 64
				}
				r := req(id, 0, 3)
				for k := 0; k < n; k++ {
					r.Prompt = append(r.Prompt, int32(base+k))
				}
				r.Prompt = append(r.Prompt, int32(i)) // the last token differs; the prefix repeats
				arr = append(arr, enginetest.Arrival{At: at, Req: r})
				at += 3000
			}
			kinds := map[engine.TransferKind]int{}
			tr := enginetest.TestEngine(t, build(t, c.extra), arr, enginetest.Options{
				TransferTime: func(tr engine.Transfer) engine.Duration {
					kinds[tr.Kind]++
					return engine.Duration(50 + 10*tr.Blocks)
				},
			})
			if kinds[engine.TransferStore] == 0 || (c.loads && kinds[engine.TransferLoad] == 0) {
				t.Errorf("transfers %v: the CPU tier was not exercised", kinds)
			}
			t.Logf("transfers started over both runs: %v", kinds)
			for id, f := range tr.Finished {
				if f.Reason != engine.FinishStop {
					t.Errorf("%s finished %+v, want stop", id, f)
				}
			}
		})
	}
}

// A prompt stored to the CPU tier and forgotten by the GPU's prefix cache loads back for a later
// request with that prefix: one TransferLoad of the prefix's tokens, Tier "cpu", and the request
// then computes only the rest. Stores are reported as TransferStore and never hold a step open:
// the engine schedules the next batch while they are in flight.
func TestOffloadHitLoadsBackWithoutRecomputing(t *testing.T) {
	e := mustEngine(t, cpuTier+tightGPU)
	a := req("a", 96, 2)
	if err := e.Add(a, 0); err != nil {
		t.Fatal(err)
	}
	results, started, now := runOne(t, e, 5000)
	stores := 0
	for _, tr := range started {
		if tr.Tier != "cpu" {
			t.Fatalf("transfer %+v names tier %q, want cpu", tr, tr.Tier)
		}
		if tr.Kind == engine.TransferStore {
			stores++
		}
	}
	if stores == 0 || len(results) == 0 {
		t.Fatalf("a's prompt was not stored: transfers %+v", started)
	}
	// A request of other tokens evicts a's prompt from the GPU's prefix cache.
	if err := e.Add(req("xx", 140, 2), now); err != nil {
		t.Fatal(err)
	}
	_, _, now = runOne(t, e, 10)
	b := req("b", 0, 3)
	b.Prompt = append(append([]int32(nil), a.Prompt...), 7, 8, 9, 10)
	if err := e.Add(b, now); err != nil {
		t.Fatal(err)
	}
	var computed []int
	var load []engine.Transfer
	for steps := 0; steps < 1000 && (e.s.HasRequests() || len(load) == 0); steps++ {
		batch, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		for _, tr := range batch.TransfersStarted {
			if tr.Kind == engine.TransferLoad {
				load = append(load, tr)
			}
			if err := e.TransferDone(tr.ID, true, now+1); err != nil {
				t.Fatal(err)
			}
		}
		now += 2
		if batch.Empty() {
			continue
		}
		for _, sc := range batch.Reqs {
			if sc.ID == "b" && sc.Tokens > 1 {
				computed = append(computed, sc.Computed, sc.Tokens)
			}
		}
		if _, err := e.Complete(batch.ID, engine.Outcome{}, now); err != nil {
			t.Fatal(err)
		}
	}
	// The GPU may still hold the prompt's first blocks (a local hit); the CPU tier supplies the
	// rest of the 96 offloaded tokens.
	if len(load) != 1 || load[0].Request != "b" || load[0].Tokens == 0 || load[0].Tokens%16 != 0 || load[0].Tier != "cpu" {
		t.Fatalf("loads %+v, want one cpu load of b's offloaded prompt blocks", load)
	}
	if len(computed) != 2 || computed[0] != 96 || computed[1] != 4 {
		t.Fatalf("b's prefill ran (computed, tokens) %v, want (96, 4): the loaded prefix not recomputed", computed)
	}
}

// A store in flight holds no step open: with its completion far off, the engine still schedules
// and completes the request's decode, and the request finishes before the store does.
func TestOffloadStoreHoldsNoStep(t *testing.T) {
	e := mustEngine(t, cpuTier)
	if err := e.Add(req("a", 64, 4), 0); err != nil {
		t.Fatal(err)
	}
	var stores []engine.Transfer
	now := engine.Time(0)
	finished := false
	for steps := 0; steps < 50 && !finished; steps++ {
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		for _, tr := range b.TransfersStarted {
			if tr.Kind == engine.TransferStore {
				stores = append(stores, tr)
			}
		}
		if b.Empty() {
			t.Fatalf("step %d scheduled nothing while a store was in flight", steps)
		}
		now += e.StepTime(b)
		res, err := e.Complete(b.ID, engine.Outcome{}, now)
		if err != nil {
			t.Fatal(err)
		}
		finished = len(res.Finished) > 0
	}
	if !finished || len(stores) == 0 {
		t.Fatalf("finished %v with stores %+v in flight, want a finished with its prompt's store outstanding", finished, stores)
	}
	// The engine steps again for the store's completion only once it is reported.
	if w, ok := e.NextWake(now); ok {
		t.Fatalf("the engine wakes at %d for a store nobody reported done", w)
	}
	if err := e.TransferDone(stores[0].ID, true, now+1_000_000); err != nil {
		t.Fatal(err)
	}
	if err := e.TransferDone(stores[0].ID, true, now+1_000_001); err != engine.ErrUnknownTransfer {
		t.Fatalf("a second TransferDone of the store: %v, want ErrUnknownTransfer", err)
	}
}

// What the offloading engine refuses, each naming the vLLM code or the issue.
func TestOffloadRefusals(t *testing.T) {
	for _, c := range []struct{ name, extra string }{
		{"unknown connector", "kv_offload: {connector: LMCacheConnectorV1, cpu_bytes: 819200, bytes_per_block: 4096}\n"},
		{"no cpu bytes", "kv_offload: {connector: OffloadingConnector, bytes_per_block: 4096}\n"},
		{"no bytes a block", "kv_offload: {connector: OffloadingConnector, cpu_bytes: 819200}\n"},
		{"arc", "kv_offload: {connector: OffloadingConnector, cpu_bytes: 819200, bytes_per_block: 4096, eviction_policy: arc}\n"},
		{"chunk not a multiple", "kv_offload: {connector: OffloadingConnector, cpu_bytes: 819200, bytes_per_block: 4096, block_size: 24}\n"},
		{"smaller than a chunk", "kv_offload: {connector: OffloadingConnector, cpu_bytes: 100, bytes_per_block: 4096}\n"},
		{"with nixl", cpuTier + producer},
		{"an MTP drafter's group", cpuTier + "speculative: {method: mtp, num_spec_tokens: 1}\nmtp_nextn_predict_layers: 1\nkv_layers:\n" +
			"- {count: 1, spec: full, num_kv_heads: 1, head_size: 64}\n" +
			"- {count: 1, spec: sliding_window, sliding_window: 32, num_kv_heads: 1, head_size: 64}\n"},
	} {
		if _, err := build(t, c.extra)(); err == nil {
			t.Errorf("%s: built, want refused", c.name)
		}
	}
}

// A failed offload transfer is refused: vLLM's offloading worker asserts every job succeeds.
func TestOffloadFailedTransferIsRefused(t *testing.T) {
	e := mustEngine(t, cpuTier)
	if err := e.Add(req("a", 64, 2), 0); err != nil {
		t.Fatal(err)
	}
	// The prefill's step stores the prompt's computed blocks (build_connector_meta at schedule).
	b, err := e.Schedule(0)
	if err != nil || len(b.TransfersStarted) == 0 {
		t.Fatalf("schedule: %v, transfers %v, want the prompt's store", err, b.TransfersStarted)
	}
	if err := e.TransferDone(b.TransfersStarted[0].ID, false, 200); err == nil {
		t.Fatal("a failed offload store was accepted")
	}
}

// The CPU tier holds a copy of each block per worker unless the layout is replicated
// (cpu/spec.py:106-121): 3276800 bytes of 4096-byte blocks are 200 chunks for four workers, 800
// for one copy.
func TestOffloadSizesTheCPUTierPerWorkerCopy(t *testing.T) {
	for _, c := range []struct {
		extra  string
		chunks int
	}{
		{"kv_offload: {connector: OffloadingConnector, cpu_bytes: 3276800, bytes_per_block: 4096, workers: 4}\n", 200},
		{"kv_offload: {connector: OffloadingConnector, cpu_bytes: 3276800, bytes_per_block: 4096, workers: 4, copies: 1}\n", 800},
		{"kv_offload: {connector: OffloadingConnector, cpu_bytes: 3276800, bytes_per_block: 4096, block_size: 32}\n", 400},
		{"kv_offload: {connector: OffloadingConnector, cpu_bytes: 3276800, bytes_per_block: 5000, page_size: 16384}\n", 200},
	} {
		e := mustEngine(t, c.extra)
		if _, free, _ := e.off.conn.CPU(); free != c.chunks {
			t.Errorf("%s: %d CPU chunks, want %d", c.extra, free, c.chunks)
		}
	}
}

// A job the scheduler tells the worker to flush (a preempted request's, or one reading a block the
// step reallocated) is done when that step's output returns, as vLLM's worker waits for it
// before the step runs (handle_preemptions, offloading/worker.py:198-224@5df63a72): it never waits
// for a TransferDone, and the one that arrives later is accepted once and changes nothing.
func TestOffloadFlushedJobsCompleteWithTheStep(t *testing.T) {
	e := mustEngine(t, cpuTier+"num_gpu_blocks: 9\nmax_model_len: 128\n")
	for _, id := range []string{"a", "bb", "ccc", "dddd"} {
		if err := e.Add(req(id, 40, 30), 0); err != nil {
			t.Fatal(err)
		}
	}
	var late []engine.TransferID
	flushes := 0
	now := engine.Time(0)
	for steps := 0; steps < 2000 && e.s.HasUnfinished(); steps++ {
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		for _, tr := range b.TransfersStarted {
			late = append(late, tr.ID) // the simulator never reports them before the end
		}
		var flush []int
		if m, ok := e.lastMeta(); ok {
			flush = m
		}
		if b.Empty() {
			now++
			continue
		}
		now += e.StepTime(b)
		if _, err := e.Complete(b.ID, engine.Outcome{}, now); err != nil {
			t.Fatal(err)
		}
		inFlight := map[int]bool{}
		for _, j := range e.off.conn.Jobs() {
			inFlight[j] = true
		}
		for _, j := range flush {
			flushes++
			if inFlight[j] {
				t.Fatalf("flushed job %d is still in flight after its step's output", j)
			}
		}
	}
	if flushes == 0 {
		t.Fatal("no job was flushed: the scenario did not preempt beside a store in flight")
	}
	for _, id := range late {
		if err := e.TransferDone(id, true, now); err != nil {
			t.Fatalf("TransferDone(%d) after its job was flushed: %v", id, err)
		}
	}
	// The completions reach the scheduler with the next applied output: a step that runs no
	// model, as vLLM's loop takes while has_pending_push_work holds has_requests true.
	if !e.s.HasRequests() {
		t.Fatal("the scheduler has no requests while offload jobs are in flight")
	}
	if b, err := e.Schedule(now + 1); err != nil || !b.Empty() {
		t.Fatalf("the completions' step: %+v, %v", b, err)
	}
	if len(e.off.conn.Jobs()) != 0 {
		t.Fatalf("jobs %v still in flight after every transfer was reported", e.off.conn.Jobs())
	}
}

// lastMeta is the last schedule's jobs_to_flush.
func (e *Engine) lastMeta() ([]int, bool) {
	if e.off == nil {
		return nil, false
	}
	return append([]int(nil), e.off.lastFlush...), true
}

// groupPricer records the group sizes the engine prices each offload transfer by.
type groupPricer struct {
	FixedPricer
	got map[engine.TransferID][]int
}

func (g *groupPricer) TransferTime(engine.Transfer) engine.Duration { return 7 }

func (g *groupPricer) GroupTransferTime(t engine.Transfer, groupBlocks []int) engine.Duration {
	g.got[t.ID] = append([]int(nil), groupBlocks...)
	return 9
}

// A pricer that prices by KV cache group is handed each offload job's GPU blocks by group
// (GPULoadStoreSpec.group_sizes, vllm/v1/kv_offload/base.py:438-469@5df63a72), which sum to the
// transfer's blocks: vLLM's worker copies each group's blocks over that group's layers only
// (gpu_worker.py:395-416), so a hybrid model's job is priced by them rather than by its blocks.
func TestAnOffloadTransferIsPricedByItsGroups(t *testing.T) {
	g := &groupPricer{FixedPricer: 100, got: map[engine.TransferID][]int{}}
	s := spec(t, cpuTier+"num_gpu_blocks: 24\nmax_model_len: 160\nkv_layers:\n"+
		"- {count: 1, spec: full, num_kv_heads: 1, head_size: 64}\n"+
		"- {count: 1, spec: sliding_window, sliding_window: 32, num_kv_heads: 1, head_size: 64}\n")
	eng, err := Factory{Pricer: func(engine.Spec) (Pricer, error) { return g, nil }}.New(s)
	if err != nil {
		t.Fatal(err)
	}
	e := eng.(*Engine)
	if err := e.Add(req("a", 96, 2), 0); err != nil {
		t.Fatal(err)
	}
	now := engine.Time(0)
	priced := 0
	for steps := 0; steps < 1000 && e.s.HasUnfinished(); steps++ {
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		for _, tr := range b.TransfersStarted {
			d, ok := e.TransferTime(tr)
			sizes := g.got[tr.ID]
			sum := 0
			for _, n := range sizes {
				sum += n
			}
			if !ok || d != 9 || len(sizes) != 2 || sum != tr.Blocks {
				t.Fatalf("transfer %+v priced %d (%v) by groups %v, want the job's two groups' blocks", tr, d, ok, sizes)
			}
			priced++
		}
		if b.Empty() {
			now += 1000
			continue
		}
		now += e.StepTime(b)
		if _, err := e.Complete(b.ID, engine.Outcome{}, now); err != nil {
			t.Fatal(err)
		}
		for _, tr := range b.TransfersStarted {
			if err := e.TransferDone(tr.ID, true, now); err != nil {
				t.Fatal(err)
			}
		}
	}
	if priced == 0 {
		t.Fatal("no offload transfer started")
	}
}
