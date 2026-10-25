package engine

// Per-engine server time series (--server-series-interval-s): the vLLM v0.31.0 Prometheus
// series a benchmark client scrapes from /metrics, binned on the simulation clock so a
// simulated run can be compared with a measured one bin for bin.
//
// One recorder per engine (Simulator): vLLM labels every scheduler metric with
// engine=<index> (vllm/v1/metrics/loggers.py:479-485 at v0.31.0), one per EngineCore, so each
// data-parallel rank is its own engine.
//
// Gauges -- kv_cache_usage_perc, num_requests_running, num_requests_waiting -- are set once
// per scheduler step from SchedulerStats (loggers.py:1034-1048), which make_stats builds after
// the step's outputs are processed (core/sched/scheduler.py:2859-2900), so a gauge holds its
// value from one step's end to the next. Each bin carries the time-weighted mean of that step
// function (what averaging scrapes over a slice estimates) and, as <name>_last, the value set
// by the last step that ended in the bin or before it (the point sample vLLM's periodic log
// line reports).
//
// Counters -- prefix_cache_queries/hits, prompt_tokens, prompt_tokens_by_source,
// generation_tokens -- are per-bin increments at the simulated instant vLLM increments them.

// ServerSeriesOutput is the file-only server_series block.
type ServerSeriesOutput struct {
	IntervalS float64              `json:"interval_s"`
	Origin    string               `json:"origin"`
	Engines   []ServerSeriesEngine `json:"engines"`
}

// ServerSeriesEngine is one engine's binned series. Bin k covers [k, k+1) x interval of
// simulated time from time 0.
type ServerSeriesEngine struct {
	ID   string `json:"id"`
	Bins int    `json:"bins"`

	KVCacheUsage           []float64 `json:"kv_cache_usage"`
	KVCacheUsageLast       []float64 `json:"kv_cache_usage_last"`
	NumRequestsRunning     []float64 `json:"num_requests_running"`
	NumRequestsRunningLast []float64 `json:"num_requests_running_last"`
	NumRequestsWaiting     []float64 `json:"num_requests_waiting"`
	NumRequestsWaitingLast []float64 `json:"num_requests_waiting_last"`

	PrefixCacheQueries             []int64 `json:"prefix_cache_queries"`
	PrefixCacheHits                []int64 `json:"prefix_cache_hits"`
	PromptTokens                   []int64 `json:"prompt_tokens"`
	PromptTokensLocalCompute       []int64 `json:"prompt_tokens_local_compute"`
	PromptTokensLocalCacheHit      []int64 `json:"prompt_tokens_local_cache_hit"`
	PromptTokensExternalKVTransfer []int64 `json:"prompt_tokens_external_kv_transfer"`
	GenerationTokens               []int64 `json:"generation_tokens"`
}

// stepGauge bins a step function: value v holds from time at until the next set.
type stepGauge struct {
	v    float64
	at   int64
	sums []float64 // integral of v over each bin, in value x ticks
	last []float64 // value in effect at the end of each closed bin
}

func (g *stepGauge) advanceTo(t, interval int64) {
	for g.at < t {
		k := g.at / interval
		end := min((k+1)*interval, t)
		for int64(len(g.sums)) <= k {
			g.sums = append(g.sums, 0)
			g.last = append(g.last, g.v)
		}
		g.sums[k] += g.v * float64(end-g.at)
		g.last[k] = g.v
		g.at = end
	}
}

func (g *stepGauge) set(t, interval int64, v float64) {
	g.advanceTo(t, interval)
	g.v = v
	// A step ending exactly at a bin's start belongs to that bin's "last" sample.
	k := t / interval
	for int64(len(g.sums)) <= k {
		g.sums = append(g.sums, 0)
		g.last = append(g.last, g.v)
	}
	g.last[k] = v
}

// serverSeries is one engine's recorder. Nil on a Simulator means the feature is off.
type serverSeries struct {
	interval                int64
	end                     int64
	usage, running, waiting stepGauge

	queries, hits, prompt, local, compute, external, generation []int64

	// firstSched is each request's prompt split from its first scheduling, emitted with its
	// first token (PrefillStats.set only when num_preemptions <= 0, scheduler.py:1034-1040).
	firstSched map[string]PrefixLookup
}

func newServerSeries(intervalUs int64) *serverSeries {
	return &serverSeries{interval: intervalUs, firstSched: make(map[string]PrefixLookup)}
}

func (s *serverSeries) add(c *[]int64, t, n int64) {
	if n == 0 {
		return
	}
	k := t / s.interval
	for int64(len(*c)) <= k {
		*c = append(*c, 0)
	}
	(*c)[k] += n
}

// kvCacheUsage is vLLM's BlockPool.get_usage: 1 - free / (num_gpu_blocks - 1)
// (core/block_pool.py:880-891 at v0.31.0). The null block (block 0, popped from the free
// queue at start-up, block_pool.py:185-188) is excluded from the denominator and is never
// free; cached blocks no request references are on the free queue and so count as free.
//
// BLIS's TotalCapacity is vLLM's num_gpu_blocks (the kernel's KV budget, from which the
// start-up check already subtracts the null block, kernelmodel/settings.go checkBlocks) and
// its free list likewise holds cached-unreferenced blocks, but BLIS reserves no null block:
// all TotalCapacity blocks are allocatable. vLLM's free count is therefore BLIS's minus one,
// so usage = UsedBlocks / (TotalCapacity - 1), clamped to 1 for the one extra block BLIS can
// hand out.
func kvCacheUsage(kv KVStore) float64 {
	total := kv.TotalCapacity() - 1
	if total <= 0 {
		return 0
	}
	return min(1, float64(kv.UsedBlocks())/float64(total))
}

// recordGauges sets the three gauges at a step's end (make_stats after update_from_output):
// running = len(scheduler.running) and waiting = len(waiting) + len(kv_holding_waiting),
// deferred included (scheduler.py:2549-2551, 2887-2892; loggers.py:1039-1043).
func (s *serverSeries) recordGauges(t int64, running, waiting int, usage float64) {
	s.usage.set(t, s.interval, usage)
	s.running.set(t, s.interval, float64(running))
	s.waiting.set(t, s.interval, float64(waiting))
	s.end = max(s.end, t)
}

// recordAdmission counts one new admission's prefix lookup at the instant it is scheduled:
// queries are request.num_tokens and hits the local hit tokens, recorded only for a first
// admission (kv_cache_manager.py:253-262, metrics/stats.py:132-144, scheduler.py:1252-1256).
// promptLen is kept for the prompt-by-source counters, which vLLM emits with the first token.
func (s *serverSeries) recordAdmission(t int64, req *Request, l PrefixLookup) {
	s.add(&s.queries, t, l.QueryTokens)
	s.add(&s.hits, t, l.HitTokens)
	if req.preemptionCount() == 0 && !req.IsDecodeSubRequest {
		if _, seen := s.firstSched[req.ID]; !seen {
			s.firstSched[req.ID] = PrefixLookup{QueryTokens: req.InputLen(), HitTokens: l.HitTokens,
				CPUHitTokens: l.CPUHitTokens}
		}
	}
	s.end = max(s.end, t)
}

// recordFirstToken emits a request's prompt tokens when its first token is produced: vLLM
// adds prefill_stats to IterationStats in the iteration whose output first carries a token
// (output_processor.py:697-707, metrics/stats.py:466-471, loggers.py:1121-1133), split as
// local_compute = prompt - local hit - external, local_cache_hit, external_kv_transfer
// (metrics/stats.py:283-295, 381-430). The external share is the offload tier's (BLIS
// PrefixLookup.CPUHitTokens; the OffloadingConnector reports it as external tokens).
func (s *serverSeries) recordFirstToken(t int64, req *Request) {
	l, ok := s.firstSched[req.ID]
	if !ok {
		return
	}
	delete(s.firstSched, req.ID)
	s.add(&s.prompt, t, l.QueryTokens)
	s.add(&s.local, t, l.HitTokens)
	s.add(&s.external, t, l.CPUHitTokens)
	s.add(&s.compute, t, l.QueryTokens-l.HitTokens-l.CPUHitTokens)
	s.end = max(s.end, t)
}

// recordGeneration counts tokens emitted in a step at its end (loggers.py:1134-1136).
func (s *serverSeries) recordGeneration(t, n int64) {
	s.add(&s.generation, t, n)
	s.end = max(s.end, t)
}

// output closes the series at end (ticks) and pads every array to bins.
func (s *serverSeries) output(id string, end int64, bins int) ServerSeriesEngine {
	for _, g := range []*stepGauge{&s.usage, &s.running, &s.waiting} {
		g.advanceTo(end, s.interval)
	}
	mean := func(g *stepGauge) ([]float64, []float64) {
		m := make([]float64, bins)
		l := make([]float64, bins)
		for k := range bins {
			lo := int64(k) * s.interval
			hi := min(lo+s.interval, end)
			switch {
			case k < len(g.sums) && hi > lo:
				m[k] = g.sums[k] / float64(hi-lo)
				l[k] = g.last[k]
			default: // past this engine's end: the gauge keeps its last value
				m[k], l[k] = g.v, g.v
			}
		}
		return m, l
	}
	pad := func(c []int64) []int64 {
		out := make([]int64, bins)
		copy(out, c)
		return out
	}
	e := ServerSeriesEngine{ID: id, Bins: bins}
	e.KVCacheUsage, e.KVCacheUsageLast = mean(&s.usage)
	e.NumRequestsRunning, e.NumRequestsRunningLast = mean(&s.running)
	e.NumRequestsWaiting, e.NumRequestsWaitingLast = mean(&s.waiting)
	e.PrefixCacheQueries, e.PrefixCacheHits = pad(s.queries), pad(s.hits)
	e.PromptTokens, e.PromptTokensLocalCompute = pad(s.prompt), pad(s.compute)
	e.PromptTokensLocalCacheHit, e.PromptTokensExternalKVTransfer = pad(s.local), pad(s.external)
	e.GenerationTokens = pad(s.generation)
	return e
}

// ServerSeriesEnd is the last instant this engine recorded anything, in ticks (0 when off).
func (sim *EngineCore) ServerSeriesEnd() int64 {
	if sim.serverSeries == nil {
		return 0
	}
	return sim.serverSeries.end
}

// ServerSeries returns this engine's binned series closed at end, padded to bins bins; ok is
// false when the feature is off.
func (sim *EngineCore) ServerSeries(id string, end int64, bins int) (ServerSeriesEngine, bool) {
	if sim.serverSeries == nil {
		return ServerSeriesEngine{}, false
	}
	return sim.serverSeries.output(id, end, bins), true
}

// ServerSeriesIntervalUs is the bin width in ticks, 0 when the feature is off.
func (sim *EngineCore) ServerSeriesIntervalUs() int64 {
	if sim.serverSeries == nil {
		return 0
	}
	return sim.serverSeries.interval
}

// BuildServerSeries assembles the block from engines, binning every engine to the latest
// engine's end so the arrays line up. Nil when no engine records a series.
func BuildServerSeries(ids []string, sims []*EngineCore, clocks []int64) *ServerSeriesOutput {
	var interval, end int64
	for i, s := range sims {
		if s.serverSeries != nil {
			interval = s.serverSeries.interval
			end = max(end, s.serverSeries.end, clocks[i])
		}
	}
	if interval <= 0 {
		return nil
	}
	// Bin k covers [k, k+1) x interval, so the last recorded instant end falls in bin
	// end/interval.
	bins := int(end/interval) + 1
	out := &ServerSeriesOutput{IntervalS: float64(interval) / 1e6, Origin: "simulation time 0 (us)"}
	for i, s := range sims {
		if e, ok := s.ServerSeries(ids[i], end, bins); ok {
			out.Engines = append(out.Engines, e)
		}
	}
	return out
}

// recordServerStepEnd sets the gauges at a step's end t, after its finished requests have
// left the batch and freed their blocks: make_stats runs after update_from_output
// (core/sched/scheduler.py:2340-2350, 2859-2900), so the running count excludes the requests
// the step finished. running is the requests still in the batch.
func (sim *EngineCore) recordServerStepEnd(t int64, running int) {
	if sim.serverSeries == nil {
		return
	}
	sim.serverSeries.recordGauges(t, running, sim.WaitQ.Len(), kvCacheUsage(sim.KVCache))
}
