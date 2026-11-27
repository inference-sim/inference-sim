// Original to blis-sglang.

package sched

import (
	"reflect"
	"testing"
)

// What the overlap loop changes is when a result reaches the scheduler: one iteration late. These
// tests drive one request through both loops and read what the scheduler does, without a golden.

func runOne(t *testing.T, overlap bool, sampled int64, ignoreEOS bool, maxNew int, steps int) (forwards [][]string, finishedAt int, s *Scheduler, before int) {
	t.Helper()
	cfg := baseConfig(t)
	cfg.EnableOverlap = overlap
	s, err := build(cfg, sampled)
	if err != nil {
		t.Fatal(err)
	}
	before = s.alloc.AvailableSize()
	finishedAt = -1
	for i := 0; i < steps; i++ {
		var arr []Arrival
		if i == 0 {
			arr = []Arrival{{RID: "x", Prompt: []int64{11, 12, 13, 14, 15}, MaxNewTokens: maxNew, IgnoreEOS: ignoreEOS}}
		}
		rec, err := s.Step(arr)
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		var rids []string
		for _, f := range rec.Forwards {
			rids = append(rids, string(f.Mode))
		}
		forwards = append(forwards, rids)
		if len(rec.Finished) > 0 && finishedAt < 0 {
			finishedAt = i
		}
	}
	return forwards, finishedAt, s, before
}

// A request that ends at its first token (the model's end-of-sequence token) is known to have
// ended only when the overlap loop processes the prefill, one iteration after it launched. By then
// the loop has formed and launched a decode batch that holds the request: the extra step whose
// token is discarded. The normal loop launches no such step.
func TestOverlapLaunchesOneDiscardedStepAfterAnEarlyStop(t *testing.T) {
	cfg := baseConfig(t)
	eos := cfg.EOSTokenIDs[0]

	normal, normalAt, _, _ := runOne(t, false, eos, false, 8, 4)
	if want := [][]string{{"EXTEND"}, nil, nil, nil}; !reflect.DeepEqual(normal, want) || normalAt != 0 {
		t.Fatalf("normal loop: forwards %v finished at %d, want %v at 0", normal, normalAt, want)
	}

	over, overAt, s, before := runOne(t, true, eos, false, 8, 5)
	if want := [][]string{{"EXTEND"}, {"DECODE"}, nil, nil, nil}; !reflect.DeepEqual(over, want) {
		t.Fatalf("overlap loop: forwards %v, want %v", over, want)
	}
	if overAt != 1 {
		t.Fatalf("overlap loop: the request finished at step %d, want 1 (one iteration after its prefill)", overAt)
	}
	// The discarded step wrote a token the request never owned; it is freed with the request.
	if got := s.alloc.AvailableSize() + s.tree.EvictableSize(); got != before {
		t.Fatalf("pool holds %d tokens after the stop, had %d", got, before)
	}
}

// A request that ends by length is known to end before its last token is decoded: the overlap loop
// leaves it out of the decode batch that its queued result completes, so no step is wasted.
func TestOverlapWastesNoStepOnALengthStop(t *testing.T) {
	const maxNew = 4
	normal, _, _, _ := runOne(t, false, 13, true, maxNew, 8)
	over, overAt, _, _ := runOne(t, true, 13, true, maxNew, 8)
	count := func(fs [][]string) (n int) {
		for _, f := range fs {
			n += len(f)
		}
		return
	}
	if count(normal) != maxNew || count(over) != maxNew {
		t.Fatalf("forward passes: normal %d, overlap %d, want %d each (one prefill, %d decodes)", count(normal), count(over), maxNew, maxNew-1)
	}
	// Normal: prefill at 0, decodes at 1..3, finished at 3. Overlap: the same passes, finished at 4.
	if overAt != maxNew {
		t.Fatalf("overlap loop: finished at step %d, want %d (the last result is processed one iteration late)", overAt, maxNew)
	}
}

// With SGLANG_DISABLE_OVERLAP_AFTER_PREFILL, the result of a prefill is processed before the next
// batch is formed, so the next batch sees what that result freed. The radix cache is off, so a
// finished request's KV is freed at once. Request a ends with its first token (one new token), so
// its three prompt tokens come back when its prefill result is processed; b, which arrives a step
// later, is launched with them free only when that result went first.
func TestOverlapAfterPrefillProcessesTheResultBeforeScheduling(t *testing.T) {
	cfg := baseConfig(t)
	eos := cfg.EOSTokenIDs[0]
	freeAtSecondLaunch := func(disable bool) int {
		cfg := cfg
		cfg.EnableOverlap = true
		cfg.DisableOverlapAfterPrefill = disable
		cfg.IsMixedChunk = false
		cfg.Disable = true
		s, err := build(cfg, eos)
		if err != nil {
			t.Fatal(err)
		}
		free := -1
		for i := 0; i < 3; i++ {
			var arr []Arrival
			switch i {
			case 0:
				arr = []Arrival{{RID: "a", Prompt: []int64{11, 12, 13}, MaxNewTokens: 8}}
			case 1:
				arr = []Arrival{{RID: "b", Prompt: []int64{21, 22, 23}, MaxNewTokens: 8}}
			}
			rec, err := s.Step(arr)
			if err != nil {
				t.Fatal(err)
			}
			if i == 1 {
				if len(rec.Forwards) != 1 || rec.Forwards[0].Mode != Extend {
					t.Fatalf("step 1 launched %+v, want b's prefill", rec.Forwards)
				}
				free = rec.Forwards[0].AvailableTokens
			}
		}
		return free
	}
	overlapped, serialized := freeAtSecondLaunch(false), freeAtSecondLaunch(true)
	if serialized != overlapped+3 {
		t.Fatalf("free tokens at b's launch: %d overlapped, %d serialized; want a's 3 prompt tokens freed first when serialized", overlapped, serialized)
	}
}
