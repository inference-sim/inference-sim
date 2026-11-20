package vllm

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

const asyncTight = "async_scheduling: true\nuse_v2_model_runner: true\nblock_size: 4\nmax_num_seqs: 3\n" +
	"max_num_batched_tokens: 16\nnum_gpu_blocks: 18\nmax_model_len: 64\n"

// both is a kv_both pool beside concurrent batches, built through the production factory.
const both = "kv_transfer: {connector: NixlConnector, kv_role: kv_both}\n"

// asyncMix is decode legs that pull their prompts beside local requests, the shape of the tier-4
// scenario nixl-both-async-defer-free.
func asyncMix() []engine.Request {
	var rs []engine.Request
	for i := 0; i < 14; i++ {
		r := req(fmt.Sprintf("r%02d", i), 9+(i*7)%23, 2+(i*5)%9)
		if r.RemotePrefill = i%2 == 0; r.RemotePrefill {
			r.Peer = peer
		}
		rs = append(rs, r)
	}
	return rs
}

func runAsyncMix(t *testing.T, extra string) []int {
	t.Helper()
	e := mustEngine(t, asyncTight+extra)
	for _, r := range asyncMix() {
		if err := e.Add(r, 0); err != nil {
			t.Fatal(err)
		}
	}
	// Pulls land 250 time units after they start. Up to two batches are in flight, as the
	// asynchronous scheduler allows: a second is scheduled before the first completes, and the
	// oldest completes when the queue is full or nothing more schedules. Each batch's shape is kept.
	type due struct {
		at engine.Time
		id engine.TransferID
	}
	var shape []int
	var transfers []due
	var inFlight []engine.Batch
	now := engine.Time(0)
	for steps := 0; steps < 10000; steps++ {
		for len(transfers) > 0 && transfers[0].at <= now {
			if err := e.TransferDone(transfers[0].id, true, now); err != nil {
				t.Fatal(err)
			}
			transfers = transfers[1:]
		}
		scheduled := false
		if len(inFlight) < 2 {
			b, err := e.Schedule(now)
			if err != nil && err != engine.ErrNoCapacity {
				t.Fatal(err)
			}
			for _, tr := range b.TransfersStarted {
				transfers = append(transfers, due{now + 250, tr.ID})
			}
			if err == nil && !b.Empty() {
				inFlight = append(inFlight, b)
				scheduled = true
				for _, sc := range b.Reqs {
					shape = append(shape, int(sc.ID[1]-'0')*100+int(sc.ID[2]-'0')*10000+sc.Tokens)
				}
				shape = append(shape, -1)
			}
		}
		if len(inFlight) == 2 || (!scheduled && len(inFlight) > 0) {
			now += e.StepTime(inFlight[0])
			if _, err := e.Complete(inFlight[0].ID, engine.Outcome{}, now); err != nil {
				t.Fatal(err)
			}
			inFlight = inFlight[1:]
			continue
		}
		if !scheduled {
			if !e.s.HasUnfinished() && len(transfers) == 0 {
				return shape
			}
			next := now + 1000
			if len(transfers) > 0 {
				next = min(next, transfers[0].at)
			}
			if w, ok := e.NextWake(now); ok {
				next = min(next, w)
			}
			now = max(next, now+1)
		}
	}
	t.Fatal("no progress")
	return nil
}

// A kv_both engine with more than one batch in flight defers a finished request's block free as
// a consumer does (is_kv_consumer, vllm/config/kv_transfer.py:129-135@5df63a72; defer_block_free,
// vllm/v1/core/sched/scheduler.py:184-194@5df63a72). This goes through core/vllm's
// buildConnector, so a role mapping that gave kv_both neither flag fails here, which the tier-4
// diff test's own mapping cannot show. The pool is tight, so when a block is released changes the
// schedule; a kv_both engine must schedule exactly as a kv_consumer does.
func TestKVBothDefersBlockFreeBesideConcurrentBatches(t *testing.T) {
	consumerShape := runAsyncMix(t, consumer)
	if got := runAsyncMix(t, both); !reflect.DeepEqual(got, consumerShape) {
		t.Errorf("kv_both under async scheduling schedules\n %v\nwant a consumer's\n %v", got, consumerShape)
	}
}
