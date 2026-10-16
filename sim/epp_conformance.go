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
	// Role is the llm-d.ai/role label ("prefill", "decode", or "" for none).
	Role string
}

// ConformanceRequest describes the request and data-layer shape of a case.
type ConformanceRequest struct {
	InputTokens int
	// MaxOutputLen is the client output cap (vLLM max_tokens); 0 = none.
	MaxOutputLen int
	// NoInFlight runs without an inflight-load-producer (the InFlightLoad and
	// UncachedRequestTokens attributes are absent on every endpoint).
	NoInFlight bool
	InFlight   InFlightLoadConfig
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
	return NewConformanceHarnessFor(v, ConformanceRequest{InputTokens: inputTokens, InFlight: inflight}, eps)
}

// NewConformanceHarnessFor builds the harness for a request shape.
func NewConformanceHarnessFor(v EPPVersion, r ConformanceRequest, eps []ConformanceEndpoint) *ConformanceHarness {
	cfg := &RoutingPipelineConfig{Version: v}
	if !r.NoInFlight {
		cfg.InFlightLoads = []InFlightLoadConfig{r.InFlight}
	}
	roles := map[string]string{}
	for _, e := range eps {
		roles[e.ID] = e.Role
	}
	p := NewRoutingPipeline(cfg, RoutingEnv{FilterRNG: rand.New(rand.NewSource(1)), Role: func(id string) string { return roles[id] }}, nil)
	fixed := &fixedPrefixProducer{info: map[string]prefixMatchInfo{}}
	p.producers[LLMDApproxPrefixProducer] = fixed
	p.producerNames = append(p.producerNames, LLMDApproxPrefixProducer)
	snaps := make([]RoutingSnapshot, len(eps))
	for i, e := range eps {
		if e.HasPrefixInfo {
			fixed.info[e.ID] = prefixMatchInfo{match: e.MatchBlocks, total: e.TotalBlocks, blockSize: e.BlockSize, cached: e.CachedBlocks}
		}
		s := NewRoutingSnapshot(e.ID)
		s.InFlightRequests = e.InFlightRequests
		if !r.NoInFlight {
			p.inflight[0].tokens[e.ID] = e.InFlightTokens
			s.InFlightTokens = e.InFlightTokens
		}
		s.QueueDepth, s.BatchSize, s.KVUtilization = e.WaitingQueue, e.RunningRequests, e.KVUtilization
		snaps[i] = s
	}
	req := &Request{ID: "conformance", InputTokens: make([]TokenID, r.InputTokens), MaxOutputLen: r.MaxOutputLen}
	p.begin(req, snaps, 0)
	return &ConformanceHarness{p: p, req: req, snaps: snaps}
}

// Filter runs one profile filter.
func (h *ConformanceHarness) Filter(f FilterConfig) []string {
	return ids(h.p.buildFilter(f)(h.req, h.snaps))
}

// Route runs a whole profile (filters, weighted scorers, max-score pick) and
// returns the tied maxima ("" set = the filters emptied the candidates).
func (h *ConformanceHarness) Route(filters []FilterConfig, scorers []ScorerConfig) (maxima []string, rejected bool) {
	ws := newWeightedScoring(scorers, filters, 16, rand.New(rand.NewSource(1)), nil, h.p)
	h.p.SetDeferred(true) // a conformance route must not record
	d := ws.Route(h.req, &RouterState{Snapshots: h.snaps})
	if d.TargetInstance == "" {
		return nil, true
	}
	best := -1.0
	for _, sc := range d.Scores {
		if sc > best {
			best = sc
		}
	}
	for id, sc := range d.Scores {
		if sc == best {
			maxima = append(maxima, id)
		}
	}
	sort.Strings(maxima)
	return maxima, false
}

// ConformanceLedger replays the in-flight load lifecycle (routing, first token,
// end of stream) on BLIS's tracker.
type ConformanceLedger struct {
	t       *InFlightLoadTracker
	streams map[string]*Request // request ID → client stream request
	inst    map[string]string   // request ID → stream (decode/primary) endpoint
}

// NewConformanceLedger starts an empty ledger.
func NewConformanceLedger(cfg InFlightLoadConfig) *ConformanceLedger {
	return &ConformanceLedger{t: newInFlightLoadTracker(cfg), streams: map[string]*Request{}, inst: map[string]string{}}
}

// Route charges a request routed to primary (and, for P/D, prefill), with the
// uncached-token estimates of harness h (the request's endpoint state).
func (l *ConformanceLedger) Route(reqID string, h *ConformanceHarness, primary, prefill string) {
	est := newInFlightLoadTracker(l.t.cfg)
	tokens := func(ep string) int64 {
		if info, ok := h.p.matchInfo("", ep); ok {
			return est.estimate(h.req, &info, h.p.role(ep))
		}
		return est.estimate(h.req, nil, h.p.role(ep))
	}
	parent := &Request{ID: reqID, InputTokens: h.req.InputTokens, MaxOutputLen: h.req.MaxOutputLen}
	l.t.charge(parent, primary, tokens(primary))
	l.streams[reqID], l.inst[reqID] = parent, primary
	if prefill != "" {
		l.t.AliasPrefill(reqID+"_prefill", reqID)
		l.t.charge(&Request{ID: reqID + "_prefill"}, prefill, tokens(prefill))
		decode := &Request{ID: reqID + "_decode"}
		l.t.SetStream(reqID, decode)
		l.streams[reqID] = decode
	}
}

// FirstToken delivers the request's first output token (StartOfStream).
func (l *ConformanceLedger) FirstToken(reqID string) {
	if s, ok := l.streams[reqID]; ok {
		s.TTFTSet = true
		l.t.ObserveInstance(l.inst[reqID])
	}
}

// End completes the request (EndOfStream).
func (l *ConformanceLedger) End(reqID string) {
	if s, ok := l.streams[reqID]; ok {
		s.State = StateCompleted
		l.t.OnRequestDone(s)
	}
}

// Tokens returns an endpoint's in-flight tokens.
func (l *ConformanceLedger) Tokens(ep string) int64 { return l.t.Tokens(ep) }

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
	if len(h.p.inflight) == 0 {
		return out
	}
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

// MaxScoreCandidates (single scorer list, no filters) returns the endpoints a max-score pick may choose for the
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
	decide func(*Request, string) func()
	fixed  *fixedPrefixProducer
	p      *RoutingPipeline
}

// NewConformanceNoHitLRU builds the scorer with the given LRU size (0 = default).
func NewConformanceNoHitLRU(lruSize int) *ConformanceNoHitLRU {
	p := NewRoutingPipeline(&RoutingPipelineConfig{Version: EPPVersionMain}, RoutingEnv{}, nil)
	fixed := &fixedPrefixProducer{info: map[string]prefixMatchInfo{}}
	p.producers[LLMDApproxPrefixProducer] = fixed
	p.producerNames = append(p.producerNames, LLMDApproxPrefixProducer)
	s, decide := newProducerNoHitLRUScorer(ScorerParams{ProducerBound: true, LRUSize: lruSize}, p)
	return &ConformanceNoHitLRU{scorer: s, decide: decide, fixed: fixed, p: p}
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
func (c *ConformanceNoHitLRU) Record(reqID, target string) {
	if a := c.decide(&Request{ID: reqID}, target); a != nil {
		a()
	}
}
