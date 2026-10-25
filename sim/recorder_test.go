package sim

import (
	"reflect"
	"testing"
)

// countingRecorder wraps the Metrics recorder and counts each call, so a test can tell the
// engine wrote the report through the recorder rather than around it.
type countingRecorder struct {
	*Metrics
	calls map[string]int
}

func (c *countingRecorder) count(name string) { c.calls[name]++ }

func (c *countingRecorder) requestDropped(req *Request, overflow bool) {
	c.count("requestDropped")
	c.Metrics.requestDropped(req, overflow)
}
func (c *countingRecorder) inputReceived(n int64) {
	c.count("inputReceived")
	c.Metrics.inputReceived(n)
}
func (c *countingRecorder) timedOut(req *Request)   { c.count("timedOut"); c.Metrics.timedOut(req) }
func (c *countingRecorder) preempted()              { c.count("preempted"); c.Metrics.preempted() }
func (c *countingRecorder) firstToken(req *Request) { c.count("firstToken"); c.Metrics.firstToken(req) }
func (c *countingRecorder) completed(cp completion) { c.count("completed"); c.Metrics.completed(cp) }
func (c *countingRecorder) lengthCapped(req *Request) {
	c.count("lengthCapped")
	c.Metrics.lengthCapped(req)
}
func (c *countingRecorder) queueSnapshot(w, r int) {
	c.count("queueSnapshot")
	c.Metrics.queueSnapshot(w, r)
}
func (c *countingRecorder) scheduled(req *Request, now int64, l PrefixLookup) {
	c.count("scheduled")
	c.Metrics.scheduled(req, now, l)
}
func (c *countingRecorder) kvUsage(used, d int64, w bool) {
	c.count("kvUsage")
	c.Metrics.kvUsage(used, d, w)
}
func (c *countingRecorder) stillHeld(w, r int) { c.count("stillHeld"); c.Metrics.stillHeld(w, r) }

// The engine writes the host's report record only through its recorder (#2021 B2, E2.3b): a
// run whose recorder counts the calls ends with the same report as one writing straight into
// Metrics, and every report family a run produces came through the recorder.
func TestEngineWritesTheReportOnlyThroughItsRecorder(t *testing.T) {
	run := func(wrap bool) (*Simulator, *countingRecorder) {
		cfg := newTestSimConfig()
		cfg.TotalKVBlocks = 40 // tight enough to preempt
		cfg.MaxModelLen = 400
		cfg.LatencyModel = newFakeLatency()
		s, err := NewEngine("rec", cfg)
		if err != nil {
			t.Fatal(err)
		}
		var c *countingRecorder
		if wrap {
			c = &countingRecorder{Metrics: s.Metrics, calls: map[string]int{}}
			s.rec = c
		}
		for i := 0; i < 12; i++ {
			s.InjectArrival(&Request{ID: string(rune('a' + i)), ArrivalTime: int64(i * 50),
				InputTokens: make([]TokenID, 150), OutputTokens: make([]TokenID, 120),
				MaxOutputLen: 120, State: StateQueued, Deadline: int64(i*50) + 400_000})
		}
		// One request that cannot fit the context window: dropped at enqueue.
		s.InjectArrival(&Request{ID: "big", ArrivalTime: 1, InputTokens: make([]TokenID, 500),
			OutputTokens: make([]TokenID, 1), State: StateQueued})
		s.Run()
		return s, c
	}
	plain, _ := run(false)
	wrapped, c := run(true)
	if !reflect.DeepEqual(plain.Metrics, wrapped.Metrics) {
		t.Fatalf("the report differs when the recorder is wrapped:\nplain   %+v\nwrapped %+v", *plain.Metrics, *wrapped.Metrics)
	}
	m := wrapped.Metrics
	for name, produced := range map[string]bool{
		"completed":      m.CompletedRequests > 0,
		"requestDropped": m.DroppedUnservable > 0,
		"inputReceived":  m.TotalInputTokens > 0,
		"preempted":      m.PreemptionCount > 0,
		"firstToken":     len(m.RequestTTFTs) > 0,
		"scheduled":      len(m.RequestSchedulingDelays) > 0,
		"queueSnapshot":  len(m.NumWaitQRequests) > 0,
		"kvUsage":        m.PeakKVBlocksUsed > 0,
		"stillHeld":      true, // Finalize always records it
	} {
		if !produced {
			t.Errorf("the fixture produced no %s values; it must exercise every family it checks", name)
			continue
		}
		if c.calls[name] == 0 {
			t.Errorf("the report has %s values but the engine never called recorder.%s: it wrote around the seam", name, name)
		}
	}
}
