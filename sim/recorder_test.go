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

func (c *countingRecorder) RecordDrop(req *Request, overflow bool) {
	c.count("requestDropped")
	c.Metrics.RecordDrop(req, overflow)
}
func (c *countingRecorder) RecordInputTokens(n int64) {
	c.count("inputReceived")
	c.Metrics.RecordInputTokens(n)
}
func (c *countingRecorder) RecordTimedOut(req *Request) {
	c.count("timedOut")
	c.Metrics.RecordTimedOut(req)
}
func (c *countingRecorder) RecordPreemption() { c.count("preempted"); c.Metrics.RecordPreemption() }
func (c *countingRecorder) RecordFirstToken(req *Request) {
	c.count("firstToken")
	c.Metrics.RecordFirstToken(req)
}
func (c *countingRecorder) RecordCompletion(cp Completion) {
	c.count("completed")
	c.Metrics.RecordCompletion(cp)
}
func (c *countingRecorder) RecordLengthCapped(req *Request) {
	c.count("lengthCapped")
	c.Metrics.RecordLengthCapped(req)
}
func (c *countingRecorder) RecordQueueSnapshot(w, r int) {
	c.count("queueSnapshot")
	c.Metrics.RecordQueueSnapshot(w, r)
}
func (c *countingRecorder) RecordScheduled(req *Request, now int64, l PrefixLookup) {
	c.count("scheduled")
	c.Metrics.RecordScheduled(req, now, l)
}
func (c *countingRecorder) RecordKVUsage(used, d int64, w bool) {
	c.count("kvUsage")
	c.Metrics.RecordKVUsage(used, d, w)
}
func (c *countingRecorder) RecordStillHeldAtEnd(w, r int) {
	c.count("stillHeld")
	c.Metrics.RecordStillHeldAtEnd(w, r)
}

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
			s.Rec = c
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
