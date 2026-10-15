package sim

import (
	"math/rand"
	"sort"
)

// This file exposes BLIS's llm-d plugin ports over plain endpoint states for the
// differential conformance harness (test/llmd-conformance), which drives the same
// states through llm-d-router's own plugin code and compares the outputs. It is
// not used by the simulator itself.

// ConformanceEndpoint is one endpoint's routing state as the harness sets it on
// both implementations.
type ConformanceEndpoint struct {
	ID string
	// Prefix match attribute (PrefixCacheMatchInfo); absent unless HasPrefixInfo.
	HasPrefixInfo                       bool
	MatchBlocks, TotalBlocks, BlockSize int
	CachedBlocks                        int
	InFlightRequests                    int
	InFlightTokens                      int64
	WaitingQueue, RunningRequests       int
	KVUtilization                       float64
}

// ConformanceHarness holds BLIS's data layer for one request over a candidate set.
type ConformanceHarness struct {
	p     *RoutingPipeline
	req   *Request
	snaps []RoutingSnapshot
}

// fixedPrefixProducer reports preset match info (the harness's prefix attribute).
type fixedPrefixProducer struct{ info map[string]prefixMatchInfo }

func (f *fixedPrefixProducer) produce(_ *Request, snapshots []RoutingSnapshot, _ int64) map[string]prefixMatchInfo {
	out := make(map[string]prefixMatchInfo, len(snapshots))
	for _, s := range snapshots {
		if info, ok := f.info[s.ID]; ok {
			out[s.ID] = info
		}
	}
	return out
}

func (f *fixedPrefixProducer) record(*Request, string, int64) {}

// NewConformanceHarness builds the harness for a request of inputTokens prompt
// tokens. The prefix attribute is published under the default approx producer
// name; the inflight producer is llm-d's default for the version (inflight
// tokens taken from the endpoints).
func NewConformanceHarness(v EPPVersion, inputTokens int, eps []ConformanceEndpoint, inflight InFlightLoadConfig) *ConformanceHarness {
	p := NewRoutingPipeline(&RoutingPipelineConfig{Version: v, InFlightLoads: []InFlightLoadConfig{inflight}},
		RoutingEnv{FilterRNG: rand.New(rand.NewSource(1))}, nil)
	fixed := &fixedPrefixProducer{info: map[string]prefixMatchInfo{}}
	p.producers[LLMDApproxPrefixProducer] = fixed
	p.producerNames = append(p.producerNames, LLMDApproxPrefixProducer)
	t := p.inflight[0]
	snaps := make([]RoutingSnapshot, len(eps))
	for i, e := range eps {
		if e.HasPrefixInfo {
			fixed.info[e.ID] = prefixMatchInfo{match: e.MatchBlocks, total: e.TotalBlocks, blockSize: e.BlockSize, cached: e.CachedBlocks}
		}
		t.tokens[e.ID] = e.InFlightTokens
		s := NewRoutingSnapshot(e.ID)
		s.InFlightRequests, s.InFlightTokens = e.InFlightRequests, e.InFlightTokens
		s.QueueDepth, s.BatchSize, s.KVUtilization = e.WaitingQueue, e.RunningRequests, e.KVUtilization
		snaps[i] = s
	}
	req := &Request{ID: "conformance", InputTokens: make([]TokenID, inputTokens)}
	p.begin(req, snaps, 0)
	return &ConformanceHarness{p: p, req: req, snaps: snaps}
}

func ids(s []RoutingSnapshot) []string {
	out := make([]string, len(s))
	for i, x := range s {
		out[i] = x.ID
	}
	return out
}

// AffinityFilter runs prefix-cache-affinity-filter (no exploration draw when
// ExplorationProbability is 0 or 1).
func (h *ConformanceHarness) AffinityFilter(cfg PrefixAffinityFilterConfig) []string {
	return ids(newPrefixAffinityFilter(cfg, h.p, nil)(h.req, h.snaps))
}

// ConcurrencyFilter runs the concurrency-detector filter.
func (h *ConformanceHarness) ConcurrencyFilter(cfg ConcurrencyDetectorConfig) []string {
	return ids(newConcurrencyFilter(cfg, h.p)(h.req, h.snaps))
}

// ConcurrencySaturation returns the detector's pool saturation.
func (h *ConformanceHarness) ConcurrencySaturation(cfg ConcurrencyDetectorConfig, reservations int) float64 {
	d := NewConcurrencyDetectorFromConfig(cfg)
	for i := 0; i < reservations; i++ {
		d.ReserveDispatch(string(rune('a' + i)))
	}
	return d.Saturation(&RouterState{Snapshots: h.snaps})
}

// UncachedTokens returns the inflight producer's per-endpoint UncachedRequestTokens.
func (h *ConformanceHarness) UncachedTokens() map[string]int64 {
	out := map[string]int64{}
	t := h.p.inflight[0]
	for _, s := range h.snaps {
		out[s.ID] = h.p.uncachedTokens(t, h.req, s.ID)
	}
	return out
}

// Score runs one BLIS scorer by its BLIS name (token-load, prefix-cache,
// queue-depth, kv-utilization, active-requests).
func (h *ConformanceHarness) Score(name string, params ScorerParams) map[string]float64 {
	switch name {
	case "token-load":
		return newTokenLoadScorer(params, h.p)(h.req, h.snaps)
	case "prefix-cache":
		return newPrefixCacheScorer(params, h.p)(h.req, h.snaps)
	}
	s, _ := newScorerWithObserver(name, 16, nil)
	return s(h.req, h.snaps)
}

// PDDecide runs prefix-based-pd-decider with the endpoint selected for decode.
func (h *ConformanceHarness) PDDecide(cfg LLMDPrefixPDDeciderConfig, selected string) bool {
	return NewLLMDPrefixPDDecider(cfg, h.p).Decide(h.req, &RouterState{Snapshots: h.snaps, SelectedInstance: selected}).Disaggregate
}

// MaxScoreCandidates returns the endpoints a max-score pick may choose for the
// weighted, clamped scores: the tied maxima, sorted. BLIS picks one uniformly with
// its seeded router RNG (llm-d: a shuffle, then rotation by a request counter).
func (h *ConformanceHarness) MaxScoreCandidates(scorers []ScorerConfig) []string {
	ws := newWeightedScoring(scorers, nil, 16, nil, nil, h.p)
	d := ws.Route(h.req, &RouterState{Snapshots: h.snaps})
	best := -1.0
	for _, s := range d.Scores {
		if s > best {
			best = s
		}
	}
	var out []string
	for id, s := range d.Scores {
		if s == best {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}

// ConformanceNoHitLRU drives the producer-bound no-hit-lru scorer over a request
// sequence (it keeps an endpoint LRU across requests).
type ConformanceNoHitLRU struct {
	scorer scorerFunc
	rec    func(*Request, string, int64)
	fixed  *fixedPrefixProducer
	p      *RoutingPipeline
}

// NewConformanceNoHitLRU builds the scorer with the given LRU size (0 = default).
func NewConformanceNoHitLRU(lruSize int) *ConformanceNoHitLRU {
	p := NewRoutingPipeline(&RoutingPipelineConfig{Version: EPPVersionMain}, RoutingEnv{}, nil)
	fixed := &fixedPrefixProducer{info: map[string]prefixMatchInfo{}}
	p.producers[LLMDApproxPrefixProducer] = fixed
	p.producerNames = append(p.producerNames, LLMDApproxPrefixProducer)
	s, rec := newProducerNoHitLRUScorer(ScorerParams{ProducerBound: true, LRUSize: lruSize}, p)
	return &ConformanceNoHitLRU{scorer: s, rec: rec, fixed: fixed, p: p}
}

// Score scores one request over the endpoints (their prefix attributes decide
// cold vs warm).
func (c *ConformanceNoHitLRU) Score(reqID string, eps []ConformanceEndpoint) map[string]float64 {
	c.fixed.info = map[string]prefixMatchInfo{}
	snaps := make([]RoutingSnapshot, len(eps))
	for i, e := range eps {
		if e.HasPrefixInfo {
			c.fixed.info[e.ID] = prefixMatchInfo{match: e.MatchBlocks, total: e.TotalBlocks, blockSize: e.BlockSize, cached: e.CachedBlocks}
		}
		snaps[i] = NewRoutingSnapshot(e.ID)
	}
	req := &Request{ID: reqID}
	c.p.begin(req, snaps, 0)
	return c.scorer(req, snaps)
}

// Record reports the routed target of the request just scored (PreRequest).
func (c *ConformanceNoHitLRU) Record(reqID, target string) { c.rec(&Request{ID: reqID}, target, 0) }
