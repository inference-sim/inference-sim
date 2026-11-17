// Original to blis-vllm.

package tiering_test

import (
	"encoding/json"
	"sort"
	"testing"

	"github.com/inference-sim/blis-vllm/core/connector/offloading"
	"github.com/inference-sim/blis-vllm/core/connector/tiering"
	"github.com/inference-sim/blis-vllm/core/kv"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// The fs tier reached through the OffloadingConnector: the connector's scheduler side calls the
// tiering manager (tiering.Connector) as its OffloadingManager, over the connector's own CPU tier.

const block = 16

type rig struct {
	s     *sched.Scheduler
	conn  *offloading.Scheduler
	fs    *tiering.FSTier
	files tiering.MemFiles
}

func newRig(t *testing.T, files tiering.MemFiles) rig {
	t.Helper()
	return newRigWith(t, files, func(fs *tiering.FSTier) tiering.Tier { return fs })
}

// newRigWith is newRig with the tier the manager sees wrapped by wrap.
func newRigWith(t *testing.T, files tiering.MemFiles, wrap func(*tiering.FSTier) tiering.Tier) rig {
	t.Helper()
	s, err := sched.New(sched.Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 512, BlockSize: block, NumBlocks: 64,
		MaxModelLen: 1024, EnableChunkedPrefill: true, EnablePrefixCaching: true, ReserveFullSequence: true,
		KVGroups: kv.ResolveGroups([]kv.Group{{Kind: kv.FullAttention}}, false), MaxConcurrentBatches: 1})
	if err != nil {
		t.Fatal(err)
	}
	fs := tiering.NewFSTier(tiering.FSConfig{RootDir: "/kv", BlockBytes: 4096}, files)
	conn, err := offloading.New(offloading.Config{Groups: []offloading.Group{{BlockSize: block}}, TokensPerHash: s.KV.HashBlockSize(),
		NumChunks: 32, OffloadPromptOnly: true, PrefixCaching: true, Workers: 1,
		Manager: func(cpu *offloading.CPUPrimary) (offloading.Manager, error) {
			return tiering.Connector{M: tiering.New(tiering.CPU(cpu), []tiering.Tier{wrap(fs)}, nil)}, nil
		}})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetConnector(conn, sched.ConnectorConfig{Consumer: true, Producer: true}); err != nil {
		t.Fatal(err)
	}
	return rig{s: s, conn: conn, fs: fs, files: files}
}

// step schedules one step, completes every GPU job it lists within the step, and then lets the
// fs tier's threads finish: its lookup batch resolves and its queued jobs run, before the next step.
func (r rig) step(t *testing.T) offloading.Meta {
	t.Helper()
	out := r.s.Schedule()
	m := out.KVConnectorMetadata.(offloading.Meta)
	var o sched.TransferOutcome
	var done []offloading.Completion
	for _, j := range append(append([]offloading.Job(nil), m.Loads...), m.Stores...) {
		done = append(done, offloading.Completion{Job: j.ID, Workers: 1})
		if !j.Store {
			o.FinishedRecving = append(o.FinishedRecving, j.Req)
		}
	}
	sort.Slice(done, func(i, k int) bool { return done[i].Job < done[k].Job })
	if len(done) > 0 {
		o.WorkerMeta = done
	}
	sampled := map[string][]int64{}
	for _, id := range out.Order {
		if !r.s.Request(id).IsPrefillChunk {
			sampled[id] = []int64{7}
		}
	}
	r.s.UpdateWith(out, sampled, o)
	r.fs.ResolveLookups()
	r.fs.RunAll()
	return m
}

func promptOf(n int) []int64 {
	p := make([]int64, n)
	for i := range p {
		p[i] = int64(100 + i)
	}
	return p
}

// A block file another engine wrote is found by the fs tier's asynchronous lookup, promoted into
// the CPU tier and loaded into the request's blocks; its tokens come from DISK.
func TestABlockFileOnSharedStorageIsLoadedThroughTheConnector(t *testing.T) {
	// The writer stores the prompt's three full chunks to the fs tier.
	files := tiering.MemFiles{}
	w := newRig(t, files)
	if err := w.s.Add(&sched.Request{ID: "w", Prompt: promptOf(3*block + 5), MaxTokens: 1, EOS: -1}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20 && (w.s.HasRequests() || w.conn.HasPendingPushWork()); i++ {
		w.step(t)
	}
	if len(files) != 3 {
		t.Fatalf("writer left %d block files, want its 3 chunks cascaded to the fs tier", len(files))
	}

	// The reader shares the directory and has an empty CPU tier.
	r := newRig(t, files)
	if err := r.s.Add(&sched.Request{ID: "r", Prompt: promptOf(3*block + 5), MaxTokens: 1, EOS: -1}); err != nil {
		t.Fatal(err)
	}
	req := r.s.Request("r")
	loaded, deferredSteps := 0, 0
	for i := 0; i < 20 && r.s.HasRequests(); i++ {
		m := r.step(t)
		if len(m.Loads) == 0 && loaded == 0 {
			deferredSteps++
		}
		for _, j := range m.Loads {
			loaded += len(j.GPU)
			if got := r.conn.ExternalHitSources(req, len(j.GPU)*block); got["DISK"] != 3*block || len(got) != 1 {
				t.Errorf("hit sources %v, want all %d tokens from DISK", got, 3*block)
			}
		}
	}
	if loaded != 3 {
		t.Fatalf("loaded %d blocks into the reader, want the 3 chunks on shared storage", loaded)
	}
	// The request waits while the lookup resolves (RETRY) and the promotion runs (HIT_PENDING).
	if deferredSteps == 0 {
		t.Error("the request was loaded at its first step, want it deferred while the fs lookup is in flight")
	}
}

// writerFiles are the block files of a prompt of three chunks that another engine cascaded to the
// shared storage, its request stating params.
func writerFiles(t *testing.T, params map[string]any) tiering.MemFiles {
	t.Helper()
	files := tiering.MemFiles{}
	w := newRig(t, files)
	w.conn.SetRequestParams("w", params)
	if err := w.s.Add(&sched.Request{ID: "w", Prompt: promptOf(3*block + 5), MaxTokens: 1, EOS: -1}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20 && (w.s.HasRequests() || w.conn.HasPendingPushWork()); i++ {
		w.step(t)
	}
	return files
}

// A request's kv_load_tiers decide whether it reads the shared storage (_create_req_context): with
// none, or one that allows the storage medium, the reader promotes the 3 chunks from the fs tier and
// loads them; one that allows only the CPU tier, or an empty list, computes its whole prompt.
func TestARequestsLoadTiersDecideWhetherItReadsTheSharedStorage(t *testing.T) {
	for _, c := range []struct {
		name   string
		params map[string]any
		want   int
	}{
		{"no params", nil, 3},
		{"params without a filter", map[string]any{"other": 1}, 3},
		{"storage allowed", map[string]any{"kv_load_tiers": []any{map[string]any{"medium": "storage"}}}, 3},
		{"cpu only", map[string]any{"kv_load_tiers": []any{map[string]any{"medium": "cpu"}}}, 0},
		{"none", map[string]any{"kv_load_tiers": []any{}}, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			r := newRig(t, writerFiles(t, nil))
			r.conn.SetRequestParams("r", c.params)
			if err := r.s.Add(&sched.Request{ID: "r", Prompt: promptOf(3*block + 5), MaxTokens: 1, EOS: -1}); err != nil {
				t.Fatal(err)
			}
			loaded := 0
			for i := 0; i < 20 && r.s.HasRequests(); i++ {
				m := r.step(t)
				for _, j := range m.Loads {
					loaded += len(j.GPU)
				}
			}
			if loaded != c.want {
				t.Errorf("loaded %d blocks from the shared storage, want %d", loaded, c.want)
			}
		})
	}
}

// paramsTier is an fs tier that records the kv_transfer_params of the context it is told a request
// starts with.
type paramsTier struct {
	*tiering.FSTier
	seen map[string]map[string]any
}

func (p paramsTier) OnNewRequest(c *tiering.ReqContext) tiering.Policy {
	p.seen[c.ReqID] = c.Params
	return p.FSTier.OnNewRequest(c)
}

// A tier is given the request's kv_transfer_params in its context, as vLLM's connector gives its
// tiers (ReqContext.kv_transfer_params): the keys the client stated, unread.
func TestATierSeesTheRequestsTransferParams(t *testing.T) {
	seen := map[string]map[string]any{}
	r := newRigWith(t, tiering.MemFiles{}, func(fs *tiering.FSTier) tiering.Tier { return paramsTier{fs, seen} })
	params := map[string]any{"remote_kv_source": map[string]any{"host": "h"}, "kv_load_tiers": []any{}}
	for id, p := range map[string]map[string]any{"a": params, "b": nil} {
		r.conn.SetRequestParams(id, p)
		if err := r.s.Add(&sched.Request{ID: id, Prompt: promptOf(block + 5), MaxTokens: 1, EOS: -1}); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 10 && r.s.HasRequests(); i++ {
		r.step(t)
	}
	if got := seen["a"]; got["remote_kv_source"] == nil || len(got) != 2 {
		t.Errorf("tier saw %v for a, want its two stated keys", got)
	}
	if got, ok := seen["b"]; !ok || got != nil {
		t.Errorf("tier saw %v (started: %v) for b, want a request with no params", got, ok)
	}
}

// A request's max_load_tokens and max_offload_tokens bound what the connector loads and what it
// offloads (RequestOffloadState, offloading/scheduler.py:383-410, _calc_num_offloadable_tokens :766
// and _lookup_complete_chunks :866-920@4ba787ec): a load is whole chunks within the limit, an offload
// stops at it, and a limit that is not a non-negative int is ignored.
func TestARequestsTokenLimitsBoundWhatItLoadsAndOffloads(t *testing.T) {
	for _, c := range []struct {
		name   string
		params map[string]any
		chunks int
	}{
		{"unstated", nil, 3},
		{"within the first two chunks", map[string]any{"max_load_tokens": 2 * block}, 2},
		{"past two chunks, short of a third", map[string]any{"max_load_tokens": 3*block - 1}, 2},
		{"less than a chunk", map[string]any{"max_load_tokens": block - 1}, 0},
		{"zero", map[string]any{"max_load_tokens": 0}, 0},
		{"more than the prompt", map[string]any{"max_load_tokens": 1000}, 3},
		{"negative is ignored", map[string]any{"max_load_tokens": -1}, 3},
		{"a float is ignored", map[string]any{"max_load_tokens": 16.0}, 3},
		{"a string is ignored", map[string]any{"max_load_tokens": "16"}, 3},
		{"an exact JSON int", map[string]any{"max_load_tokens": json.Number("32")}, 2},
	} {
		t.Run("load, "+c.name, func(t *testing.T) {
			r := newRig(t, writerFiles(t, nil))
			r.conn.SetRequestParams("r", c.params)
			if err := r.s.Add(&sched.Request{ID: "r", Prompt: promptOf(3*block + 5), MaxTokens: 1, EOS: -1}); err != nil {
				t.Fatal(err)
			}
			loaded := 0
			for i := 0; i < 20 && r.s.HasRequests(); i++ {
				for _, j := range r.step(t).Loads {
					loaded += len(j.GPU)
				}
			}
			if loaded != c.chunks {
				t.Errorf("loaded %d chunks, want %d", loaded, c.chunks)
			}
		})
	}
	for _, c := range []struct {
		name   string
		params map[string]any
		files  int
	}{
		{"offload unstated", nil, 3},
		{"offload two chunks", map[string]any{"max_offload_tokens": 2 * block}, 2},
		{"offload less than a chunk", map[string]any{"max_offload_tokens": block - 1}, 0},
		{"offload zero", map[string]any{"max_offload_tokens": 0}, 0},
		{"offload negative is ignored", map[string]any{"max_offload_tokens": -3}, 3},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := len(writerFiles(t, c.params)); got != c.files {
				t.Errorf("%d block files on the shared storage, want %d", got, c.files)
			}
		})
	}
}
