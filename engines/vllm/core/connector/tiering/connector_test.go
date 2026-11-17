// Original to blis-vllm.

package tiering_test

import (
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
			return tiering.Connector{M: tiering.New(tiering.CPU(cpu), []tiering.Tier{fs}, nil)}, nil
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
