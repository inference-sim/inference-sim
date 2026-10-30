package vllm

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
)

// A schedule that runs no tokens can still change the prefix cache: a decoder admitting a remote
// prefill allocates the pull's blocks, evicting cached ones (BlockRemoved). vLLM publishes the
// events every update_from_output takes (scheduler.py:2370-2385@5df63a72), an empty step's too,
// so the engine reports them with the next applied result rather than dropping or reordering
// them: every block it stored is removed in the stream before it is stored again.
func TestKVEventsOfAnEmptyScheduleAreReported(t *testing.T) {
	for _, mode := range []string{"", "async_scheduling: true\n"} {
		e := mustEngine(t, consumer+"num_gpu_blocks: 12\nmax_model_len: 160\n"+mode)
		e.EnableKVEvents()
		var events []engine.KVEvent
		now := engine.Time(0)
		pending := map[engine.TransferID]bool{}
		for _, id := range []string{"a", "bb", "ccc"} {
			r := req(id, 80, 2)
			r.RemotePrefill = true
			if err := e.Add(r, now); err != nil {
				t.Fatal(err)
			}
			for steps := 0; steps < 1000 && (e.s.HasRequests() || len(e.pending) > 0); steps++ {
				for tid := range pending {
					if err := e.TransferDone(tid, true, now); err != nil {
						t.Fatal(err)
					}
					delete(pending, tid)
				}
				b, err := e.Schedule(now)
				if err != nil && err != engine.ErrNoCapacity {
					t.Fatal(err)
				}
				for _, tr := range b.TransfersStarted {
					pending[tr.ID] = true
				}
				if b.Empty() {
					if len(e.pending) > 0 {
						now += 100
						res, err := e.Complete(e.pending[0].id, engine.Outcome{}, now)
						if err != nil {
							t.Fatal(err)
						}
						events = append(events, res.KVEvents...)
					} else {
						now += 1000
					}
					continue
				}
				now += e.StepTime(b)
				res, err := e.Complete(b.ID, engine.Outcome{}, now)
				if err != nil {
					t.Fatal(err)
				}
				events = append(events, res.KVEvents...)
			}
		}
		got := len(e.s.TakeKVEvents())
		if got != 0 {
			t.Errorf("%q: %d KV events left queued after the last result", mode, got)
		}
		stored := map[string]int{}
		removals := 0
		for _, ev := range events {
			if ev.Removed {
				removals++
				if stored[ev.Hash] == 0 {
					t.Errorf("%q: removal of %s, which the stream never stored", mode, ev.Hash)
				}
				stored[ev.Hash]--
				continue
			}
			stored[ev.Hash]++
		}
		if removals == 0 {
			t.Errorf("%q: no removal reported; the check would be vacuous", mode)
		}
		held := 0
		for _, n := range stored {
			held += n
		}
		if cached := e.s.KV.Pool.NumCached(); held != cached {
			t.Errorf("%q: the published stream holds %d blocks, the prefix cache %d", mode, held, cached)
		}
	}
}
