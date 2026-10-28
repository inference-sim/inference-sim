package sim

import (
	"sort"
	"testing"

	"pgregory.net/rapid"
)

// renderJob is one request offered to a Renderer.
type renderJob struct{ arrival, service int64 }

func drawRenderJobs(rt *rapid.T) []renderJob {
	n := rapid.IntRange(1, 40).Draw(rt, "requests")
	jobs := make([]renderJob, n)
	for i := range jobs {
		jobs[i] = renderJob{
			arrival: rapid.Int64Range(0, 5_000).Draw(rt, "arrival"),
			service: rapid.Int64Range(0, 2_000).Draw(rt, "service"),
		}
	}
	sort.SliceStable(jobs, func(a, b int) bool { return jobs[a].arrival < jobs[b].arrival })
	return jobs
}

// TestRenderer_EachServerIsAFIFOQueueOfItsThreads is INV-R1's queue law. For every admission
// sequence in arrival order: a render starts no earlier than its arrival and lasts exactly its
// service; a server never runs more renders at once than it has threads; a request waits only
// while every thread of its server is busy (work-conserving); renders on a server start in
// arrival order (FIFO); and requests go to the servers round-robin.
func TestRenderer_EachServerIsAFIFOQueueOfItsThreads(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		servers := rapid.IntRange(1, 3).Draw(rt, "servers")
		workers := rapid.IntRange(1, 3).Draw(rt, "workers")
		jobs := drawRenderJobs(rt)
		r, err := NewRenderer(servers, workers)
		if err != nil {
			rt.Fatal(err)
		}
		type span struct{ start, end int64 }
		byServer := make([][]span, servers)
		for i, j := range jobs {
			done := r.Admit(j.arrival, j.service)
			start := done - j.service
			if start < j.arrival {
				rt.Fatalf("request %d starts at %d before it arrives at %d", i, start, j.arrival)
			}
			s := i % servers
			if start > j.arrival {
				busy := 0
				for _, p := range byServer[s] {
					if p.start <= j.arrival && j.arrival < p.end {
						busy++
					}
				}
				if busy < workers {
					rt.Fatalf("request %d waits from %d to %d with only %d of server %d's %d threads busy",
						i, j.arrival, start, busy, s, workers)
				}
			}
			if prev := byServer[s]; len(prev) > 0 && start < prev[len(prev)-1].start {
				rt.Fatalf("request %d starts at %d, before the earlier arrival's start %d on server %d",
					i, start, prev[len(prev)-1].start, s)
			}
			byServer[s] = append(byServer[s], span{start, done})
		}
		for s, spans := range byServer {
			for _, p := range spans {
				if p.start == p.end {
					continue
				}
				running := 0
				for _, q := range spans {
					if q.start <= p.start && p.start < q.end {
						running++
					}
				}
				if running > workers {
					rt.Fatalf("server %d runs %d renders at %d with %d threads", s, running, p.start, workers)
				}
			}
		}
		if r.Rendered() != int64(len(jobs)) {
			rt.Fatalf("Rendered = %d, want every admission (%d)", r.Rendered(), len(jobs))
		}
	})
}

// TestRenderer_FreeOrUncontendedRenderIsTheBareOverhead is the metamorphic edge of INV-R1: a
// render that costs nothing, or a stage with a thread for every request, finishes each request
// at arrival + service -- exactly the uncontended per-request delay of an inert front end.
func TestRenderer_FreeOrUncontendedRenderIsTheBareOverhead(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		jobs := drawRenderJobs(rt)
		free, _ := NewRenderer(rapid.IntRange(1, 3).Draw(rt, "servers"), rapid.IntRange(1, 3).Draw(rt, "workers"))
		wide, _ := NewRenderer(1, len(jobs))
		for i, j := range jobs {
			if got := free.Admit(j.arrival, 0); got != j.arrival {
				rt.Fatalf("free render of request %d finishes at %d, want its arrival %d", i, got, j.arrival)
			}
			if got := wide.Admit(j.arrival, j.service); got != j.arrival+j.service {
				rt.Fatalf("uncontended render of request %d finishes at %d, want %d", i, got, j.arrival+j.service)
			}
		}
		if free.WaitTicks() != 0 || wide.WaitTicks() != 0 {
			rt.Fatalf("waits %d and %d, want none", free.WaitTicks(), wide.WaitTicks())
		}
	})
}

func TestNewRenderer_RefusesAnEmptyStage(t *testing.T) {
	for _, c := range [][2]int{{0, 1}, {1, 0}, {-1, 1}} {
		if _, err := NewRenderer(c[0], c[1]); err == nil {
			t.Errorf("NewRenderer(%d, %d) accepted", c[0], c[1])
		}
	}
	if err := (FrontendConfig{RendererNumWorkers: -1}).Validate(); err == nil {
		t.Error("a negative renderer_num_workers was accepted")
	}
	if err := (FrontendConfig{APIServerCount: -1}).Validate(); err == nil {
		t.Error("a negative api_server_count was accepted")
	}
}
