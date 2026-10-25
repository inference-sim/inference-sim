package sim

import (
	"fmt"
	"sort"
)

// EngineMetrics is the state a model server publishes on its Prometheus endpoint, as vLLM's
// PrometheusStatLogger records it from the SchedulerStats of one engine step
// (vllm/v1/core/sched/scheduler.py make_stats, vllm/v1/metrics/loggers.py record).
//
// The values describe the engine at the end of the step, and they become visible to a scraper
// only when that step ends: a scrape that lands while a step runs sees the previous step's
// values, exactly as a vLLM scrape does.
type EngineMetrics struct {
	// PublishedAt is the simulated time (µs) at which these values were recorded.
	PublishedAt int64
	// Waiting is len(scheduler.waiting), including preempted requests (vllm:num_requests_waiting).
	Waiting int
	// Running is len(scheduler.running) (vllm:num_requests_running).
	Running int
	// KVUsage is the fraction of KV blocks in use (vllm:kv_cache_usage_perc). The block pool
	// holds only usable blocks, so used/total is vLLM's BlockPool.get_usage().
	KVUsage float64
	// Preemptions is the cumulative preemption count (vllm:num_preemptions_total).
	Preemptions int64
	// RunningAdapters and WaitingAdapters are the LoRA adapters with at least one running or
	// waiting request (vllm:lora_requests_info), sorted.
	RunningAdapters []string
	WaitingAdapters []string
}

// engineMetricsLog holds the latest published EngineMetrics and the one a running step will
// publish when it ends.
type engineMetricsLog struct {
	visible EngineMetrics
	pending EngineMetrics
	hasNext bool
}

// EngineMetricsAt returns the metrics a scrape at simulated time t observes.
func (sim *engineCore) EngineMetricsAt(t int64) EngineMetrics {
	log := &sim.engineMetrics
	if log.hasNext && log.pending.PublishedAt <= t {
		log.visible = log.pending
		log.hasNext = false
	}
	return log.visible
}

// publishEngineMetrics records the engine state at the end of a step that ends at time at.
// It only reads simulation state.
func (sim *engineCore) publishEngineMetrics(at int64) {
	m := EngineMetrics{
		PublishedAt: at,
		Waiting:     sim.WaitQ.Len() + sim.remoteKVWaiting,
		Preemptions: sim.Metrics.PreemptionCount,
	}
	if total := sim.KVCache.TotalCapacity(); total > 0 {
		m.KVUsage = float64(sim.KVCache.UsedBlocks()) / float64(total)
	}
	if sim.RunningBatch != nil {
		m.Running = len(sim.RunningBatch.Requests)
	}
	if sim.residentAdapters != nil {
		m.RunningAdapters = adapterSet(sim.RunningBatch)
		m.WaitingAdapters = adapterSet(&Batch{Requests: sim.WaitQ.Items()})
	}
	log := &sim.engineMetrics
	if log.hasNext && log.pending.PublishedAt <= at {
		log.visible = log.pending
	}
	log.pending = m
	log.hasNext = true
}

func adapterSet(b *Batch) []string {
	if b == nil {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, r := range b.Requests {
		if r.Adapter != "" && !seen[r.Adapter] {
			seen[r.Adapter] = true
			out = append(out, r.Adapter)
		}
	}
	sort.Strings(out)
	return out
}

// AdapterCapacity returns the per-instance resident LoRA adapter slots (vLLM max_loras), or 0
// when the LoRA subsystem is inert.
func (sim *engineCore) AdapterCapacity() int {
	if sim.residentAdapters == nil || sim.adapterCapacity <= 0 {
		return 0
	}
	return sim.adapterCapacity
}

// SetKVEventSink makes the engine publish its prefix cache's KV events, as vLLM does: at the end
// of every step that changed the set of cached block hashes, sink receives that step's events, in
// order, with the time the step ends. With prefix caching disabled the engine caches no block for
// reuse and, as vLLM, publishes nothing.
func (sim *engineCore) SetKVEventSink(sink func(at int64, events []KVEvent)) {
	src, ok := sim.KVCache.(KVEventSource)
	if !ok {
		panic(fmt.Sprintf("SetKVEventSink: KV cache %T cannot publish KV events", sim.KVCache))
	}
	if sim.prefixCachingDisabled {
		return
	}
	src.EnableKVEvents()
	sim.kvEventSink = sink
}

// publishKVEvents hands the events of the step that ends at time at to the sink.
func (sim *engineCore) publishKVEvents(at int64) {
	if sim.kvEventSink == nil {
		return
	}
	if events := sim.KVCache.(KVEventSource).TakeKVEvents(); len(events) > 0 {
		sim.kvEventSink(at, events)
	}
}
