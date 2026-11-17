package cluster

import "github.com/inference-sim/inference-sim/sim"

// The windowed collector (inference-sim#1141, design DEV-4): a Collector that keeps the last N
// ticks of each model's replica readings, so an analyzer can see how a signal moved and not only
// where it stands, and that fills the latency fields the point-in-time snapshot leaves zero from
// the change in each instance's completed-request totals since the previous tick. Each tick's
// reading carries the time it was taken and a sequence number, the keys of a published value
// (DEV-4): a cache of a window is keyed on both.

// LatencyTotals are one instance's cumulative completed-request totals, the values the windowed
// collector differences between ticks.
type LatencyTotals struct {
	Completed int
	TTFTSumUs float64 // sum over completed requests of time to first token, microseconds
	ITLSumUs  float64 // sum over completed requests with an ITL sample of mean inter-token latency, microseconds
	// ITLCount is how many completed requests have an ITL sample: one with fewer than two output
	// tokens has a TTFT and no inter-token gap (#1894), so ITLCount can be below Completed.
	ITLCount  int
	InTokens  int64
	OutTokens int64
}

// WindowSample is one tick's readings for one model, with the time and sequence it was published at.
type WindowSample struct {
	TimeUs   int64
	Seq      int64
	Replicas []ReplicaMetrics
}

// WindowedCollector decorates a Collector. With size 0 the cluster uses the inner collector
// unchanged, so a run that asks for no window is byte-identical.
type WindowedCollector struct {
	inner  Collector
	size   int
	totals func(instanceID string) (LatencyTotals, bool)
	prev   map[string]LatencyTotals
	prevAt int64
	seq    int64
	hist   map[string][]WindowSample
}

// NewWindowedCollector wraps inner, keeping the last size ticks (size >= 1); totals reads an
// instance's cumulative totals (false for an instance it does not know).
func NewWindowedCollector(inner Collector, size int, totals func(string) (LatencyTotals, bool)) *WindowedCollector {
	if size < 1 {
		panic("NewWindowedCollector: size must be >= 1")
	}
	return &WindowedCollector{inner: inner, size: size, totals: totals, prev: map[string]LatencyTotals{},
		hist: map[string][]WindowSample{}}
}

// Collect implements Collector.
func (w *WindowedCollector) Collect(state *sim.ClusterState) []ModelSignals {
	signals := w.inner.Collect(state)
	if state == nil {
		return signals
	}
	w.seq++
	elapsedS := float64(state.Clock-w.prevAt) / 1e6
	next := map[string]LatencyTotals{}
	for i := range signals {
		for j := range signals[i].Replicas {
			r := &signals[i].Replicas[j]
			cur, ok := w.totals(r.InstanceID)
			if !ok {
				continue
			}
			next[r.InstanceID] = cur
			d := LatencyTotals{Completed: cur.Completed - w.prev[r.InstanceID].Completed,
				TTFTSumUs: cur.TTFTSumUs - w.prev[r.InstanceID].TTFTSumUs, ITLSumUs: cur.ITLSumUs - w.prev[r.InstanceID].ITLSumUs,
				ITLCount: cur.ITLCount - w.prev[r.InstanceID].ITLCount,
				InTokens: cur.InTokens - w.prev[r.InstanceID].InTokens, OutTokens: cur.OutTokens - w.prev[r.InstanceID].OutTokens}
			if d.Completed > 0 {
				n := float64(d.Completed)
				// TTFT is the mean over every completion of the tick, ITL over those that have an
				// inter-token gap (the whole-run LatencyStats' populations).
				r.TTFT = d.TTFTSumUs / n
				if d.ITLCount > 0 {
					r.ITL = d.ITLSumUs / float64(d.ITLCount)
				}
				r.AvgInTokens, r.AvgOutTokens = float64(d.InTokens)/n, float64(d.OutTokens)/n
				if elapsedS > 0 {
					r.DispatchRate = n / elapsedS
				}
			}
		}
		id := signals[i].ModelID
		sample := WindowSample{TimeUs: state.Clock, Seq: w.seq, Replicas: append([]ReplicaMetrics(nil), signals[i].Replicas...)}
		h := append(w.hist[id], sample)
		if len(h) > w.size {
			h = h[len(h)-w.size:]
		}
		w.hist[id] = h
		signals[i].Window = append([]WindowSample(nil), h...) // oldest first
	}
	w.prev, w.prevAt = next, state.Clock
	return signals
}

// instanceTotals reads an instance's cumulative totals from its metrics, by instance ID.
func (cs *ClusterSimulator) instanceTotals(id string) (LatencyTotals, bool) {
	inst := cs.instanceByID(InstanceID(id))
	if inst == nil {
		return LatencyTotals{}, false
	}
	return inst.LatencyTotals(), true
}
