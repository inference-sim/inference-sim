package sim

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"

	"pgregory.net/rapid"
)

// Parity tests for the llm-d EPP data layer. Cases cite the llm-d-router unit test
// they are lifted from ("main:" = main@981acb8a, paths under
// pkg/epp/framework/plugins/). Endpoint states are built with a precise producer of
// block size 1 over a 100-token request, so an instance's match is exactly the
// count its cache function returns (llm-d tests use 100 total blocks the same way).

func eppTokens(n int) []TokenID {
	out := make([]TokenID, n)
	for i := range out {
		out[i] = TokenID(i + 1)
	}
	return out
}

func eppReq(id string, n int) *Request {
	return &Request{ID: id, InputTokens: eppTokens(n)}
}

func eppSnaps(ids ...string) []RoutingSnapshot {
	out := make([]RoutingSnapshot, len(ids))
	for i, id := range ids {
		out[i] = NewRoutingSnapshot(id)
	}
	return out
}

func snapIDs(s []RoutingSnapshot) []string {
	out := make([]string, len(s))
	for i, x := range s {
		out[i] = x.ID
	}
	return out
}

// matchPipeline builds a pipeline whose producer "m" reports match[id] blocks of
// a 100-block request, plus an inflight tracker "inflight-load-producer" bound to it.
func matchPipeline(v EPPVersion, match map[string]int) *RoutingPipeline {
	cacheFn := map[string]func([]TokenID) int{}
	for id, m := range match {
		m := m
		cacheFn[id] = func([]TokenID) int { return m }
	}
	cfg := &RoutingPipelineConfig{
		Version:   v,
		Producers: []PrefixProducerConfig{{Name: "m", Type: LLMDPrecisePrefixProducer, BlockSizeTokens: 1, TierWeights: map[string]float64{"gpu": 1}}},
		InFlightLoads: []InFlightLoadConfig{func() InFlightLoadConfig {
			c := DefaultInFlightLoadConfig(LLMDInFlightLoadProducer, v)
			c.PrefixMatchInfoProducerName = "m"
			return c
		}()},
	}
	return NewRoutingPipeline(cfg, RoutingEnv{EngineBlockSize: 1, FilterRNG: rand.New(rand.NewSource(1))}, cacheFn)
}

func setInFlightTokens(p *RoutingPipeline, tokens map[string]int64) {
	t := p.tracker("")
	for id, n := range tokens {
		t.tokens[id] = n
	}
}

// --- prefix-cache-affinity-filter (main:scheduling/filter/prefixcacheaffinity/plugin_test.go) ---

func TestPrefixAffinityFilter_LLMDParity(t *testing.T) {
	cases := []struct {
		name   string // llm-d test it is lifted from
		cfg    PrefixAffinityFilterConfig
		match  map[string]int
		tokens map[string]int64
		order  []string
		want   []string
	}{
		{"TestFilter_AffinityThresholdDisabled (:64)", PrefixAffinityFilterConfig{AffinityThreshold: 0},
			map[string]int{"a": 90, "b": 10}, nil, []string{"a", "b"}, []string{"a", "b"}},
		{"TestFilter_SingleEndpoint (:74)", PrefixAffinityFilterConfig{AffinityThreshold: 0.8},
			map[string]int{"a": 10}, nil, []string{"a"}, []string{"a"}},
		{"TestFilter_NoStickyEndpoints (:81)", PrefixAffinityFilterConfig{AffinityThreshold: 0.8},
			map[string]int{"a": 10, "b": 20, "c": 50}, nil, []string{"a", "b", "c"}, []string{"a", "b", "c"}},
		{"TestFilter_NarrowToSticky (:92, throughput source, idle)", PrefixAffinityFilterConfig{AffinityThreshold: 0.8, MaxTTFTPenaltyMs: 5000, PeakPrefillThroughput: 1000},
			map[string]int{"a": 90, "b": 85, "c": 10}, nil, []string{"a", "b", "c"}, []string{"a", "b"}},
		{"TestFilter_ThroughputTTFTBreaksStickiness (:115)", PrefixAffinityFilterConfig{AffinityThreshold: 0.8, MaxTTFTPenaltyMs: 100, PeakPrefillThroughput: 1000},
			map[string]int{"a": 90, "b": 10}, map[string]int64{"a": 500, "b": 50}, []string{"a", "b"}, []string{"a", "b"}},
		{"TestFilter_ThroughputTTFTWithinThreshold (:125)", PrefixAffinityFilterConfig{AffinityThreshold: 0.8, MaxTTFTPenaltyMs: 1000, PeakPrefillThroughput: 1000},
			map[string]int{"a": 90, "b": 10}, map[string]int64{"a": 500, "b": 50}, []string{"a", "b"}, []string{"a"}},
		{"TestFilter_TTFTPenaltyDisabled (:136)", PrefixAffinityFilterConfig{AffinityThreshold: 0.8, MaxTTFTPenaltyMs: 0, PeakPrefillThroughput: 1000},
			map[string]int{"a": 90, "b": 10}, map[string]int64{"a": 5000, "b": 50}, []string{"a", "b"}, []string{"a"}},
		{"TestFilter_ExplorationProbability (:147)", PrefixAffinityFilterConfig{AffinityThreshold: 0.8, ExplorationProbability: 1.0},
			map[string]int{"a": 90, "b": 10}, nil, []string{"a", "b"}, []string{"a", "b"}},
		{"sticky at exactly the threshold (>=, plugin.go:220)", PrefixAffinityFilterConfig{AffinityThreshold: 0.5},
			map[string]int{"a": 50, "b": 49}, nil, []string{"a", "b"}, []string{"a"}},
		{"penalty exactly at the limit keeps stickiness (> not >=)", PrefixAffinityFilterConfig{AffinityThreshold: 0.8, MaxTTFTPenaltyMs: 450, PeakPrefillThroughput: 1000},
			map[string]int{"a": 90, "b": 10}, map[string]int64{"a": 500, "b": 50}, []string{"a", "b"}, []string{"a"}},
		{"best sticky vs best non-sticky (minima, plugin.go:290-305)", PrefixAffinityFilterConfig{AffinityThreshold: 0.8, MaxTTFTPenaltyMs: 100, PeakPrefillThroughput: 1000},
			map[string]int{"a": 90, "b": 95, "c": 10, "d": 5}, map[string]int64{"a": 900, "b": 120, "c": 30, "d": 5000}, []string{"a", "b", "c", "d"}, []string{"a", "b"}},
	}
	for _, v := range []EPPVersion{EPPVersionMain, EPPVersionV0_10_0} {
		for _, tc := range cases {
			t.Run(string(v)+"/"+tc.name, func(t *testing.T) {
				p := matchPipeline(v, tc.match)
				p.begin(eppReq("r", 100), eppSnaps(tc.order...), 0)
				setInFlightTokens(p, tc.tokens)
				cfg := tc.cfg
				cfg.PrefixMatchInfoProducerName = "m"
				f := newPrefixAffinityFilter(cfg, p, rand.New(rand.NewSource(1)))
				got := snapIDs(f(eppReq("r", 100), eppSnaps(tc.order...)))
				if fmt.Sprint(got) != fmt.Sprint(tc.want) {
					t.Errorf("got %v, want %v", got, tc.want)
				}
			})
		}
	}
}

// A missing in-flight signal (no inflight producer) skips the TTFT gate on main
// (main:…/prefixcacheaffinity/plugin.go:240-250, TestFilter_*MissingSignalThroughput
// :177/:187) and reads as zero load on v0.10.0 (v0.10.0:…/plugin.go:264-276).
func TestPrefixAffinityFilter_MissingSignal_PerVersion(t *testing.T) {
	for _, tc := range []struct {
		v    EPPVersion
		want []string
	}{{EPPVersionMain, []string{"a", "b"}}, {EPPVersionV0_10_0, []string{"a"}}} {
		p := matchPipeline(tc.v, map[string]int{"a": 90, "b": 10})
		p.inflightByNm = map[string]*InFlightLoadTracker{}
		p.begin(eppReq("r", 100), eppSnaps("a", "b"), 0)
		cfg := DefaultPrefixAffinityFilterConfig(tc.v)
		cfg.PrefixMatchInfoProducerName = "m"
		got := snapIDs(newPrefixAffinityFilter(cfg, p, nil)(eppReq("r", 100), eppSnaps("a", "b")))
		if fmt.Sprint(got) != fmt.Sprint(tc.want) {
			t.Errorf("%s: got %v, want %v", tc.v, got, tc.want)
		}
	}
}

// Without a seeded RNG the filter never takes the exploration branch, so an
// unseeded caller stays deterministic.
func TestPrefixAffinityFilter_NoRNGNeverExplores(t *testing.T) {
	p := matchPipeline(EPPVersionMain, map[string]int{"a": 90, "b": 10})
	p.begin(eppReq("r", 100), eppSnaps("a", "b"), 0)
	cfg := PrefixAffinityFilterConfig{AffinityThreshold: 0.8, ExplorationProbability: 0.999, PrefixMatchInfoProducerName: "m"}
	if got := snapIDs(newPrefixAffinityFilter(cfg, p, nil)(nil, eppSnaps("a", "b"))); fmt.Sprint(got) != "[a]" {
		t.Errorf("got %v, want [a]", got)
	}
}

// Validation (main:…/prefixcacheaffinity/plugin_test.go:268-345): main rejects a
// negative threshold, v0.10.0 accepts it; both reject > 1 and a zero
// peakPrefillThroughput with the gate on.
func TestPrefixAffinityFilterConfig_Validate(t *testing.T) {
	neg := DefaultPrefixAffinityFilterConfig(EPPVersionMain)
	neg.AffinityThreshold = -0.1
	if neg.Validate(EPPVersionMain) == nil || neg.Validate(EPPVersionV0_10_0) != nil {
		t.Error("affinityThreshold -0.1: main must reject, v0.10.0 must accept")
	}
	hi := DefaultPrefixAffinityFilterConfig(EPPVersionMain)
	hi.AffinityThreshold = 1.5
	zero := DefaultPrefixAffinityFilterConfig(EPPVersionMain)
	zero.PeakPrefillThroughput = 0
	unused := zero
	unused.MaxTTFTPenaltyMs = 0
	for _, v := range []EPPVersion{EPPVersionMain, EPPVersionV0_10_0} {
		if hi.Validate(v) == nil || zero.Validate(v) == nil || unused.Validate(v) != nil {
			t.Errorf("%s: validation mismatch", v)
		}
	}
}

// --- token-load-scorer (main:scheduling/scorer/tokenload/token_load_test.go:34-105) ---

func TestTokenLoadScorer_LLMDParity(t *testing.T) {
	t.Run("in-flight load only (:44)", func(t *testing.T) {
		// Uncached incoming tokens are 0: every block cached.
		p := matchPipeline(EPPVersionMain, map[string]int{"pod1": 100, "pod2": 100, "pod3": 100})
		p.begin(eppReq("r", 100), eppSnaps("pod1", "pod2", "pod3"), 0)
		setInFlightTokens(p, map[string]int64{"pod2": 500, "pod3": 1000})
		s := newTokenLoadScorer(ScorerParams{QueueThresholdTokens: 1000}, p)(eppReq("r", 100), eppSnaps("pod1", "pod2", "pod3"))
		assertScores(t, s, map[string]float64{"pod1": 1.0, "pod2": 0.5, "pod3": 0.0})
	})
	t.Run("in-flight load plus current request impact (:66)", func(t *testing.T) {
		// 250 uncached tokens on every endpoint: a 1000-token prompt, 750 blocks cached.
		p := matchPipeline(EPPVersionMain, map[string]int{"pod1": 750, "pod2": 750, "pod3": 750})
		p.begin(eppReq("r", 1000), eppSnaps("pod1", "pod2", "pod3"), 0)
		setInFlightTokens(p, map[string]int64{"pod2": 250, "pod3": 750})
		s := newTokenLoadScorer(ScorerParams{QueueThresholdTokens: 1000}, p)(eppReq("r", 1000), eppSnaps("pod1", "pod2", "pod3"))
		assertScores(t, s, map[string]float64{"pod1": 0.75, "pod2": 0.5, "pod3": 0.0})
	})
	t.Run("missing data degrades to an idle score (:91)", func(t *testing.T) {
		p := matchPipeline(EPPVersionMain, nil)
		p.inflightByNm = map[string]*InFlightLoadTracker{}
		s := newTokenLoadScorer(ScorerParams{}, p)(eppReq("r", 100), eppSnaps("x"))
		assertScores(t, s, map[string]float64{"x": 1.0})
	})
	t.Run("default threshold 4194304 (token_load.go:35, :65)", func(t *testing.T) {
		p := matchPipeline(EPPVersionMain, map[string]int{"x": 100})
		p.begin(eppReq("r", 100), eppSnaps("x"), 0)
		setInFlightTokens(p, map[string]int64{"x": 4194304 / 4})
		s := newTokenLoadScorer(ScorerParams{QueueThresholdTokens: -5}, p)(eppReq("r", 100), eppSnaps("x"))
		assertScores(t, s, map[string]float64{"x": 0.75})
	})
}

func assertScores(t *testing.T, got, want map[string]float64) {
	t.Helper()
	for id, w := range want {
		if math.Abs(got[id]-w) > 1e-9 {
			t.Errorf("%s: score %v, want %v", id, got[id], w)
		}
	}
}

// --- prefix-cache-scorer (main:scheduling/scorer/prefix/plugin.go:135-185) ---

func TestPrefixCacheScorer_Formula(t *testing.T) {
	p := matchPipeline(EPPVersionMain, map[string]int{"a": 50, "b": 100, "c": 0})
	p.begin(eppReq("r", 100), eppSnaps("a", "b", "c"), 0)
	ratio := newPrefixCacheScorer(ScorerParams{PrefixMatchInfoProducerName: "m"}, p)(nil, eppSnaps("a", "b", "c"))
	assertScores(t, ratio, map[string]float64{"a": 0.5, "b": 1, "c": 0})
	// matchLengthWeight 0.5, scale 200 tokens: a = 0.5·(50/200)² + 0.5·0.5.
	mixed := newPrefixCacheScorer(ScorerParams{PrefixMatchInfoProducerName: "m", MatchLengthWeight: 0.5, MatchLengthScaleTokens: 200}, p)(nil, eppSnaps("a", "b"))
	assertScores(t, mixed, map[string]float64{"a": 0.5*0.0625 + 0.25, "b": 0.5*0.25 + 0.5})
	// A producer that is absent scores 0 (missing attribute).
	missing := newPrefixCacheScorer(ScorerParams{PrefixMatchInfoProducerName: "absent"}, p)(nil, eppSnaps("a"))
	assertScores(t, missing, map[string]float64{"a": 0})
}

// --- concurrency-detector (main:flowcontrol/saturationdetector/concurrency/detector_test.go) ---

func TestConcurrencyDetector_Saturation_LLMDParity(t *testing.T) {
	// TestDetector_Saturation (:228-310), maxConcurrency 10, requests mode.
	d := NewConcurrencyDetector(10)
	cases := []struct {
		name string
		load []int
		want float64
	}{
		{"empty_candidate_list_fail_closed", nil, 1.0},
		{"single_endpoint_empty", []int{0}, 0},
		{"single_endpoint_half_full", []int{5}, 0.5},
		{"single_endpoint_full", []int{10}, 1.0},
		{"multi_endpoint_mixed_load", []int{10, 0}, 0.5},
		{"multi_endpoint_overloaded", []int{15, 5}, 1.0},
		{"multi_endpoint_very_overloaded", []int{15, 15}, 1.5},
	}
	for _, tc := range cases {
		var snaps []RoutingSnapshot
		for i, l := range tc.load {
			s := NewRoutingSnapshot(fmt.Sprint(i))
			s.InFlightRequests = l
			snaps = append(snaps, s)
		}
		if got := d.Saturation(&RouterState{Snapshots: snaps}); math.Abs(got-tc.want) > 1e-9 {
			t.Errorf("%s: saturation %v, want %v", tc.name, got, tc.want)
		}
	}
	// TestDetector_HybridSaturation (:683-745): per-endpoint max(req, tok), averaged.
	h := NewConcurrencyDetectorFromConfig(ConcurrencyDetectorConfig{Mode: ConcurrencyModeHybrid, MaxConcurrency: 10, MaxTokenConcurrency: 100})
	for _, tc := range []struct {
		req  int
		tok  int64
		want float64
	}{{0, 0, 0}, {2, 80, 0.8}, {9, 10, 0.9}, {10, 5, 1.0}, {1, 100, 1.0}} {
		s := NewRoutingSnapshot("a")
		s.InFlightRequests, s.InFlightTokens = tc.req, tc.tok
		if got := h.Saturation(&RouterState{Snapshots: []RoutingSnapshot{s}}); math.Abs(got-tc.want) > 1e-9 {
			t.Errorf("hybrid {%d,%d}: %v, want %v", tc.req, tc.tok, got, tc.want)
		}
	}
}

// Dispatch reservations count toward request saturation on main
// (TestDetector_DispatchReservations, detector_test.go:312-360); v0.10.0 has none.
func TestConcurrencyDetector_DispatchReservations(t *testing.T) {
	cfg := DefaultConcurrencyDetectorConfig()
	cfg.MaxConcurrency = 2
	on := cfg
	on.DispatchReservations = true
	for _, tc := range []struct {
		cfg  ConcurrencyDetectorConfig
		want float64
	}{{on, 1.0}, {cfg, 0.0}} {
		d := NewConcurrencyDetectorFromConfig(tc.cfg)
		d.ReserveDispatch("r1")
		d.ReserveDispatch("r2")
		d.ReserveDispatch("r2") // duplicate: no-op
		st := &RouterState{Snapshots: eppSnaps("a")}
		if got := d.Saturation(st); got != tc.want {
			t.Errorf("reservations=%v: saturation %v, want %v", tc.cfg.DispatchReservations, got, tc.want)
		}
		d.ReleaseDispatch("r1")
		d.ReleaseDispatch("r1")
		d.ReleaseDispatch("unknown")
		if got := d.Saturation(st); tc.cfg.DispatchReservations && got != 0.5 {
			t.Errorf("after release: %v, want 0.5", got)
		}
	}
}

func loadSnap(id string, req int, tok int64) RoutingSnapshot {
	s := NewRoutingSnapshot(id)
	s.InFlightRequests, s.InFlightTokens = req, tok
	return s
}

func TestConcurrencyDetector_Filter_LLMDParity(t *testing.T) {
	main := func(c ConcurrencyDetectorConfig) ConcurrencyDetectorConfig {
		c.FailOpen, c.CountIncomingTokens = true, true
		return c
	}
	tok := ConcurrencyDetectorConfig{Mode: ConcurrencyModeTokens, MaxConcurrency: 100, MaxTokenConcurrency: 100, Headroom: 0.2}
	// TestDetector_TokenFilter (:483-525): burst limit 100·1.2 = 120.
	d := NewConcurrencyDetectorFromConfig(main(tok))
	if got := snapIDs(d.Filter([]RoutingSnapshot{loadSnap("e", 0, 44), loadSnap("clean", 0, 0)})); len(got) != 2 {
		t.Errorf("below burst: kept %v", got)
	}
	if got := snapIDs(d.Filter([]RoutingSnapshot{loadSnap("e", 0, 120), loadSnap("clean", 0, 0)})); fmt.Sprint(got) != "[clean]" {
		t.Errorf("at burst limit: kept %v, want [clean]", got)
	}
	if got := d.Filter([]RoutingSnapshot{loadSnap("e", 0, 120)}); len(got) != 1 {
		t.Errorf("main fails open when all endpoints exceed the limit: kept %d", len(got))
	}
	// v0.10.0 has no fail-open: the all-dropped result is empty.
	if got := NewConcurrencyDetectorFromConfig(tok).Filter([]RoutingSnapshot{loadSnap("e", 0, 120)}); len(got) != 0 {
		t.Errorf("v0.10.0 does not fail open: kept %d", len(got))
	}
	// TestDetector_TokenFilterIncomingTokens (:527-555): load 110, limit 120.
	for _, tc := range []struct {
		in   int64
		kept int
	}{{9, 2}, {10, 2}, {11, 1}} {
		in := tc.in
		got := d.filterWithIncoming([]RoutingSnapshot{loadSnap("e", 0, 110), loadSnap("clean", 0, 0)}, func(id string) int64 {
			if id == "e" {
				return in
			}
			return 0
		})
		if len(got) != tc.kept {
			t.Errorf("incoming %d: kept %d, want %d", tc.in, len(got), tc.kept)
		}
	}
	// TestDetector_TokenFilterOversizedRequest (:557-584): an idle endpoint takes a
	// request above the limit; a loaded one does not (fail closed).
	for _, mode := range []ConcurrencyMode{ConcurrencyModeTokens, ConcurrencyModeHybrid} {
		c := main(ConcurrencyDetectorConfig{Mode: mode, MaxConcurrency: 10, MaxTokenConcurrency: 100})
		c.FailOpen = false
		dd := NewConcurrencyDetectorFromConfig(c)
		all120 := func(string) int64 { return 120 }
		if got := snapIDs(dd.filterWithIncoming([]RoutingSnapshot{loadSnap("loaded", 1, 1), loadSnap("idle", 0, 0)}, all120)); fmt.Sprint(got) != "[idle]" {
			t.Errorf("%s oversized: kept %v, want [idle]", mode, got)
		}
		if got := dd.filterWithIncoming([]RoutingSnapshot{loadSnap("loaded", 1, 1)}, all120); len(got) != 0 {
			t.Errorf("%s oversized loaded-only: kept %d, want 0", mode, len(got))
		}
	}
	// TestDetector_HybridFilter (:747-811): request limit 10, token limit 100.
	hy := NewConcurrencyDetectorFromConfig(ConcurrencyDetectorConfig{Mode: ConcurrencyModeHybrid, MaxConcurrency: 10, MaxTokenConcurrency: 100})
	for _, tc := range []struct {
		req  int
		tok  int64
		kept int
	}{{5, 50, 2}, {10, 50, 1}, {5, 100, 1}, {20, 200, 1}} {
		if got := hy.Filter([]RoutingSnapshot{loadSnap("e", tc.req, tc.tok), loadSnap("clean", 0, 0)}); len(got) != tc.kept {
			t.Errorf("hybrid {%d,%d}: kept %d, want %d", tc.req, tc.tok, len(got), tc.kept)
		}
	}
	// Headroom truncation: int64(256·1.0) = 256; 255 passes, 256 is dropped.
	r := NewConcurrencyDetectorFromConfig(ConcurrencyDetectorConfig{Mode: ConcurrencyModeRequests, MaxConcurrency: 256, MaxTokenConcurrency: 1})
	if got := snapIDs(r.Filter([]RoutingSnapshot{loadSnap("a", 255, 0), loadSnap("b", 256, 0)})); fmt.Sprint(got) != "[a]" {
		t.Errorf("requests limit 256: kept %v", got)
	}
}

// --- inflight-load-producer (main:requestcontrol/dataproducer/inflightload) ---

func TestInFlightLoad_EstimatePerVersion(t *testing.T) {
	info := &prefixMatchInfo{match: 2, total: 4, blockSize: 64} // 256-token index, 128 cached
	req := &Request{ID: "r", InputTokens: eppTokens(300), MaxOutputLen: 0}
	v010 := newInFlightLoadTracker(DefaultInFlightLoadConfig("x", EPPVersionV0_10_0))
	if got := v010.estimate(req, info, ""); got != 128+44 {
		t.Errorf("uncached = (4-2)·64 + tail 44: got %d", got)
	}
	if got := v010.estimate(req, nil, ""); got != 300 {
		t.Errorf("no match info ⇒ full input: got %d", got)
	}
	v010.cfg.AddEstimatedOutput = true
	if got := v010.estimate(req, info, ""); got != 172+450 {
		t.Errorf("v0.10.0 output = round(300·1.5): got %d", got)
	}
	req.MaxOutputLen = 100
	if got := v010.estimate(req, info, ""); got != 172+100 {
		t.Errorf("client cap clamps the output estimate: got %d", got)
	}
	mainT := newInFlightLoadTracker(DefaultInFlightLoadConfig("x", EPPVersionMain))
	mainT.cfg.AddEstimatedOutput = true
	req.MaxOutputLen = 0
	if got := mainT.estimate(req, info, ""); got != 172+1000 {
		t.Errorf("main unknown-bucket output 1000: got %d", got)
	}
	if got := mainT.estimate(req, info, "prefill"); got != 172 {
		t.Errorf("main prefill role charges prompt only: got %d", got)
	}
	if got := mainT.estimate(req, info, "decode"); got != 1000 {
		t.Errorf("main decode role with output estimation charges output only: got %d", got)
	}
	mainT.cfg.MaxEstimatedOutput = 10
	if got := mainT.estimate(req, info, ""); got != 182 {
		t.Errorf("operator cap: got %d", got)
	}
}

// Without output estimation a request's token charge is released at its first
// token; with it, prefill charges at the first token and the rest at completion
// (main:…/inflightload/producer.go:569-640). The counter floors at zero.
func TestInFlightLoad_ReleaseLifecycle(t *testing.T) {
	for _, add := range []bool{false, true} {
		cfg := DefaultInFlightLoadConfig("x", EPPVersionMain)
		cfg.AddEstimatedOutput = add
		tr := newInFlightLoadTracker(cfg)
		parent := &Request{ID: "p"}
		tr.charge(parent, "d", 100)
		tr.AliasPrefill("p_prefill", "p")
		tr.charge(&Request{ID: "p_prefill"}, "pf", 40)
		decode := &Request{ID: "p_decode"}
		tr.SetStream("p", decode)
		tr.OnRequestDone(&Request{ID: "p_prefill", State: StateCompleted}) // prefill completion releases nothing
		if tr.Tokens("pf") != 40 || tr.Tokens("d") != 100 {
			t.Fatalf("add=%v: prefill completion must not release: pf=%d d=%d", add, tr.Tokens("pf"), tr.Tokens("d"))
		}
		decode.TTFTSet = true
		tr.ObserveInstance("d")
		wantD := int64(0)
		if add {
			wantD = 100
		}
		if tr.Tokens("pf") != 0 || tr.Tokens("d") != wantD {
			t.Errorf("add=%v after first token: pf=%d d=%d, want 0/%d", add, tr.Tokens("pf"), tr.Tokens("d"), wantD)
		}
		tr.OnRequestDone(decode)
		if tr.Tokens("d") != 0 || len(tr.groups) != 0 {
			t.Errorf("add=%v after completion: d=%d groups=%d", add, tr.Tokens("d"), len(tr.groups))
		}
		tr.tokens["d"] = 5
		tr.charge(&Request{ID: "q"}, "d", 10)
		tr.tokens["d"] = 3
		tr.ReleaseGroup("q")
		if tr.Tokens("d") != 0 {
			t.Errorf("decrement floors at zero: %d", tr.Tokens("d"))
		}
	}
}

// A prefill sub-request that ends without completing (timeout, drop) ends the
// request: the whole group is released (main:…/inflightload/producer.go:621-633).
func TestInFlightLoad_PrefillTimeoutReleasesGroup(t *testing.T) {
	for _, state := range []RequestState{StateTimedOut, StateQueued} {
		tr := newInFlightLoadTracker(DefaultInFlightLoadConfig("x", EPPVersionMain))
		tr.charge(&Request{ID: "p"}, "d", 100)
		tr.AliasPrefill("p_prefill", "p")
		tr.charge(&Request{ID: "p_prefill"}, "pf", 40)
		tr.OnRequestDone(&Request{ID: "p_prefill", State: state})
		if tr.Tokens("d") != 0 || tr.Tokens("pf") != 0 || len(tr.groups) != 0 || len(tr.alias) != 0 {
			t.Errorf("state %s: d=%d pf=%d groups=%d alias=%d", state, tr.Tokens("d"), tr.Tokens("pf"), len(tr.groups), len(tr.alias))
		}
	}
}

// --- approx-prefix-cache-producer (main:requestcontrol/dataproducer/approximateprefix) ---

func TestApproxProducer_BlockSizeCapacityAndCap(t *testing.T) {
	// autoTune: engine block 16 floors to 64 (plugin.go:384-400); capacity = KV blocks.
	cfg := DefaultApproxPrefixProducerConfig("a", EPPVersionMain)
	env := RoutingEnv{EngineBlockSize: 16, KVBlocks: func(id string) int64 { return 3 }}
	p := newApproxPrefixProducer(cfg, env)
	if p.idx.BlockSize() != 64 {
		t.Errorf("block size %d, want floor 64", p.idx.BlockSize())
	}
	// 200 tokens ⇒ 4 blocks (3 full + 1 partial, hashing.go:107-130).
	r := eppReq("r", 200)
	if got := p.produce(r, eppSnaps("x"), 0)["x"].total; got != 4 {
		t.Errorf("total blocks %d, want 4 (partial block counted)", got)
	}
	p.record(r, "x", 0)
	if n := p.idx.InstanceBlockCount("x"); n != 3 {
		t.Errorf("LRU sized to the instance's 3 KV blocks holds %d", n)
	}
	// maxPrefixTokensToMatch 128 at block 64 ⇒ 2 blocks (plugin.go:410-419).
	cfg.MaxPrefixTokensToMatch = 128
	if got := newApproxPrefixProducer(cfg, env).produce(r, eppSnaps("x"), 0)["x"].total; got != 2 {
		t.Errorf("capped total %d, want 2", got)
	}
	// autoTune off without a configured capacity uses lruCapacityPerServer.
	cfg = DefaultApproxPrefixProducerConfig("a", EPPVersionMain)
	cfg.AutoTune, cfg.LRUCapacityPerServer = false, 2
	q := newApproxPrefixProducer(cfg, env)
	q.record(r, "x", 0)
	if n := q.idx.InstanceBlockCount("x"); n != 2 {
		t.Errorf("lruCapacityPerServer 2 holds %d", n)
	}
}

// main inserts a prompt's blocks tail first, so an over-capacity LRU keeps the
// prompt's head and the next request still matches it (indexer.go:94-103);
// v0.10.0 inserts head first and keeps the tail, losing the match.
func TestApproxProducer_InsertionOrderPerVersion(t *testing.T) {
	env := RoutingEnv{EngineBlockSize: 64, KVBlocks: func(string) int64 { return 2 }}
	r := eppReq("r", 64*4)
	for _, tc := range []struct {
		v    EPPVersion
		want int
	}{{EPPVersionMain, 2}, {EPPVersionV0_10_0, 0}} {
		p := newApproxPrefixProducer(DefaultApproxPrefixProducerConfig("a", tc.v), env)
		p.record(r, "x", 0)
		if got := p.produce(eppReq("r2", 64*4), eppSnaps("x"), 0)["x"].match; got != tc.want {
			t.Errorf("%s: match %d, want %d", tc.v, got, tc.want)
		}
	}
}

// The galloping match equals the linear scan when held blocks form a prefix.
func TestPrefixCacheIndex_GallopingEqualsLinearOnPrefixes(t *testing.T) {
	idx := NewPrefixCacheIndex(4, 100)
	hashes := idx.ComputeBlockHashes(eppTokens(4 * 37))
	for k := 0; k <= len(hashes); k++ {
		id := fmt.Sprint(k)
		idx.RecordBlocks(hashes[:k], id)
		if g, l := idx.MatchLengthGalloping(hashes, id), idx.MatchLength(hashes, id); g != l || g != k {
			t.Errorf("held %d: galloping %d, linear %d", k, g, l)
		}
	}
}

// --- precise-prefix-cache-producer ---

func TestPreciseProducer_TierWeightsAndSpeculative(t *testing.T) {
	gpu := map[string]func([]TokenID) int{"x": func([]TokenID) int { return 2 }}
	cpu := map[string]func([]TokenID) int{"x": func([]TokenID) int { return 5 }}
	cfg := PrefixProducerConfig{Name: "p", Type: LLMDPrecisePrefixProducer, BlockSizeTokens: 1, TierWeights: map[string]float64{"gpu": 1, "cpu": 0.5}}
	p := NewRoutingPipeline(&RoutingPipelineConfig{Producers: []PrefixProducerConfig{cfg}}, RoutingEnv{CPUTierCacheFn: cpu}, gpu)
	p.begin(eppReq("r", 10), eppSnaps("x"), 0)
	info, _ := p.matchInfo("p", "x")
	// 2 GPU blocks (1.0) + 3 more CPU-only blocks (0.5) = 3.5, truncated to 3
	// (producer.go:441 int(WeightedScore)); cached counts all 5.
	if info.match != 3 || info.cached != 5 || info.total != 10 {
		t.Errorf("tier-weighted match %+v, want match 3 cached 5 total 10", info)
	}
	// Speculative entries extend the prefix until the TTL passes.
	cfg.Speculative, cfg.SpeculativeTTLUs = true, 100
	p = NewRoutingPipeline(&RoutingPipelineConfig{Producers: []PrefixProducerConfig{cfg}}, RoutingEnv{}, gpu)
	p.recordRouted(eppReq("r", 10), "x", 1000)
	p.begin(eppReq("r", 10), eppSnaps("x"), 1050)
	if info, _ := p.matchInfo("p", "x"); info.cached != 10 || info.match != 10 {
		t.Errorf("speculative within TTL: %+v", info)
	}
	p.begin(eppReq("r", 10), eppSnaps("x"), 1100)
	if info, _ := p.matchInfo("p", "x"); info.cached != 2 {
		t.Errorf("speculative after TTL: %+v", info)
	}
}

// --- no-hit-lru-scorer (main:scheduling/scorer/nohitlru/no_hit_lru.go:184-340) ---

func TestProducerNoHitLRU_ColdRequestsSpread(t *testing.T) {
	p := matchPipeline(EPPVersionMain, map[string]int{"a": 0, "b": 0, "c": 0})
	scorer, decide := newProducerNoHitLRUScorer(ScorerParams{PrefixMatchInfoProducerName: "m", ProducerBound: true}, p)
	rec := func(r *Request, target string, _ int64) {
		if a := decide(r, target); a != nil {
			a()
		}
	}
	snaps := eppSnaps("a", "b", "c")
	pick := func(id string) string {
		r := eppReq(id, 100)
		p.begin(r, snaps, 0)
		s := scorer(r, snaps)
		best := ""
		for _, sn := range snaps {
			if best == "" || s[sn.ID] > s[best] {
				best = sn.ID
			}
		}
		rec(r, best, 0)
		return best
	}
	// Never-used endpoints first in candidate order, then least recently used.
	got := []string{pick("1"), pick("2"), pick("3"), pick("4")}
	if fmt.Sprint(got) != "[a b c a]" {
		t.Errorf("cold picks %v, want [a b c a]", got)
	}
	// A warm request scores 0.5 everywhere and does not touch the LRU.
	warm := matchPipeline(EPPVersionMain, map[string]int{"a": 1})
	ws, _ := newProducerNoHitLRUScorer(ScorerParams{PrefixMatchInfoProducerName: "m", ProducerBound: true}, warm)
	warm.begin(eppReq("w", 100), eppSnaps("a", "b"), 0)
	assertScores(t, ws(eppReq("w", 100), eppSnaps("a", "b")), map[string]float64{"a": 0.5, "b": 0.5})
}

// --- prefix-based-pd-decider (main:…/disagg/prefix_based_pd_decider_test.go:354-470) ---

func TestLLMDPrefixPDDecider_LLMDParity(t *testing.T) {
	cases := []struct {
		name               string
		nonCached, prompt  int
		cached             int
		input              int
		selected           string
		expectDisaggregate bool
	}{
		{"threshold zero disables disaggregation", 0, 0, 5, 10, "x", false},
		{"no selected endpoint returns false", 5, 0, 0, 10, "", false},
		{"input shorter than promptTokens threshold", 5, 20, 0, 10, "x", false},
		{"input shorter than threshold", 20, 0, 0, 10, "x", false},
		{"input equals promptTokens threshold", 5, 10, 5, 10, "x", true},
		{"non-cached suffix below threshold", 5, 0, 8, 10, "x", false},
		{"non-cached suffix equals threshold", 5, 0, 5, 10, "x", true},
		{"non-cached suffix above threshold", 3, 0, 2, 10, "x", true},
		{"fully cached prompt", 1, 0, 10, 10, "x", false},
		{"no cache hit at all", 5, 0, 0, 10, "x", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := matchPipeline(EPPVersionMain, map[string]int{"x": tc.cached})
			r := eppReq("r", tc.input)
			p.begin(r, eppSnaps("x"), 0)
			d := NewLLMDPrefixPDDecider(LLMDPrefixPDDeciderConfig{NonCachedTokens: tc.nonCached, PromptTokens: tc.prompt, PrefixMatchInfoProducerName: "m"}, p)
			got := d.Decide(r, &RouterState{Snapshots: eppSnaps("x"), SelectedInstance: tc.selected}).Disaggregate
			if got != tc.expectDisaggregate {
				t.Errorf("disaggregate %v, want %v", got, tc.expectDisaggregate)
			}
		})
	}
}

// --- invariants, properties, metamorphic relations ---

// eppPolicy builds a weighted policy with the curvebender filter chain and scorers
// over a match pipeline.
func eppPolicy(v EPPVersion, match map[string]int, weights [2]float64, guard ConcurrencyDetectorConfig, threshold float64) *WeightedScoring {
	p := matchPipeline(v, match)
	aff := DefaultPrefixAffinityFilterConfig(v)
	aff.AffinityThreshold, aff.MaxTTFTPenaltyMs, aff.PeakPrefillThroughput, aff.PrefixMatchInfoProducerName = threshold, 4500, 15000, "m"
	filters := []FilterConfig{{Type: LLMDConcurrencyDetector, Name: "g", Concurrency: guard}, {Type: LLMDPrefixAffinityFilter, Name: "f", Affinity: aff}}
	scorers := []ScorerConfig{
		{Name: "token-load", Weight: weights[0]},
		{Name: "prefix-cache", Weight: weights[1], Params: ScorerParams{PrefixMatchInfoProducerName: "m"}},
	}
	return newWeightedScoring(scorers, filters, 1, rand.New(rand.NewSource(7)), nil, p)
}

func guardCfg(v EPPVersion, max int64) ConcurrencyDetectorConfig {
	c := DefaultConcurrencyDetectorConfig()
	c.MaxConcurrency = max
	if v == EPPVersionMain {
		c.FailOpen, c.CountIncomingTokens = true, true
	}
	return c
}

type eppState struct {
	ids    []string
	match  map[string]int
	reqs   map[string]int
	tokens map[string]int64
}

func drawEPPState(rt *rapid.T) eppState {
	n := rapid.IntRange(1, 6).Draw(rt, "endpoints")
	s := eppState{match: map[string]int{}, reqs: map[string]int{}, tokens: map[string]int64{}}
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("i%d", i)
		s.ids = append(s.ids, id)
		s.match[id] = rapid.IntRange(0, 100).Draw(rt, "match_"+id)
		s.reqs[id] = rapid.IntRange(0, 11).Draw(rt, "requests_"+id)
		s.tokens[id] = rapid.Int64Range(0, 200000).Draw(rt, "tokens_"+id)
	}
	return s
}

func (s eppState) snaps() []RoutingSnapshot {
	out := eppSnaps(s.ids...)
	for i := range out {
		out[i].InFlightRequests = s.reqs[out[i].ID]
	}
	return out
}

func routeOnce(ws *WeightedScoring, s eppState, snaps []RoutingSnapshot) RoutingDecision {
	setInFlightTokens(ws.pipeline, s.tokens)
	return ws.Route(eppReq("r", 100), &RouterState{Snapshots: snaps})
}

// Invariants over random endpoint states, both versions: every composite score is
// in [0,1] (clamp + normalized weights); the decision is a candidate the guard
// admits, unless the guard failed open (main) — and the guard empties the set
// only where llm-d does (v0.10.0, all endpoints at the limit); the same seed
// gives the same decision.
func TestEPPPipeline_Invariants(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		s := drawEPPState(rt)
		max := rapid.Int64Range(1, 10).Draw(rt, "maxConcurrency")
		for _, v := range []EPPVersion{EPPVersionMain, EPPVersionV0_10_0} {
			d := routeOnce(eppPolicy(v, s.match, [2]float64{1, 1}, guardCfg(v, max), 0.5), s, s.snaps())
			again := routeOnce(eppPolicy(v, s.match, [2]float64{1, 1}, guardCfg(v, max), 0.5), s, s.snaps())
			if d.TargetInstance != again.TargetInstance {
				rt.Fatalf("INV-6: same seed, different targets %q vs %q", d.TargetInstance, again.TargetInstance)
			}
			admitted := 0
			for _, id := range s.ids {
				if int64(s.reqs[id]) < max {
					admitted++
				}
			}
			if d.TargetInstance == "" {
				if v == EPPVersionMain || admitted > 0 {
					rt.Fatalf("%s: empty decision with %d admitted endpoints", v, admitted)
				}
				continue
			}
			if admitted > 0 && int64(s.reqs[d.TargetInstance]) >= max {
				rt.Fatalf("%s: guard admitted %s at %d >= limit %d while %d endpoints were below it", v, d.TargetInstance, s.reqs[d.TargetInstance], max, admitted)
			}
			for id, sc := range d.Scores {
				if sc < 0 || sc > 1+1e-12 || math.IsNaN(sc) {
					rt.Fatalf("score %s = %v outside [0,1]", id, sc)
				}
			}
		}
	})
}

// Metamorphic relations:
//   - scaling every scorer weight by c > 0 leaves the decision unchanged;
//   - affinityThreshold 0 routes like a profile without the affinity filter;
//   - maxConcurrency above every endpoint's load routes like no guard;
//   - permuting endpoint order changes nothing when the maximum is unique.
func TestEPPPipeline_Metamorphic(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		s := drawEPPState(rt)
		for _, v := range []EPPVersion{EPPVersionMain, EPPVersionV0_10_0} {
			base := routeOnce(eppPolicy(v, s.match, [2]float64{1, 2}, guardCfg(v, 4), 0.5), s, s.snaps())
			scaled := routeOnce(eppPolicy(v, s.match, [2]float64{3.5, 7}, guardCfg(v, 4), 0.5), s, s.snaps())
			if base.TargetInstance != scaled.TargetInstance {
				rt.Fatalf("weight scaling changed the decision: %q vs %q", base.TargetInstance, scaled.TargetInstance)
			}

			open := guardCfg(v, 1<<40)
			thr0 := eppPolicy(v, s.match, [2]float64{1, 2}, open, 0)
			noFilters := eppPolicy(v, s.match, [2]float64{1, 2}, open, 0)
			noFilters.filters = nil
			a, b := routeOnce(thr0, s, s.snaps()), routeOnce(noFilters, s, s.snaps())
			if a.TargetInstance != b.TargetInstance || fmt.Sprint(a.Scores) != fmt.Sprint(b.Scores) {
				rt.Fatalf("threshold 0 + open guard differs from no filters: %v vs %v", a, b)
			}

			perm := s.snaps()
			sort.Slice(perm, func(i, j int) bool { return perm[i].ID > perm[j].ID })
			x := routeOnce(eppPolicy(v, s.match, [2]float64{1, 2}, guardCfg(v, 4), 0.5), s, s.snaps())
			y := routeOnce(eppPolicy(v, s.match, [2]float64{1, 2}, guardCfg(v, 4), 0.5), s, perm)
			if unique(x.Scores) && x.TargetInstance != y.TargetInstance {
				rt.Fatalf("permutation changed a unique-max decision: %q vs %q", x.TargetInstance, y.TargetInstance)
			}
		}
	})
}

func unique(scores map[string]float64) bool {
	best, n := -1.0, 0
	for _, s := range scores {
		switch {
		case s > best:
			best, n = s, 1
		case s == best:
			n++
		}
	}
	return n == 1
}

// A pipeline-free weighted policy (the --routing-scorers path) is built exactly as
// before: the EPP constructor with a nil pipeline makes the same decisions as
// NewRoutingPolicyWithCache over a request stream (INV-6 byte-identity).
func TestNewRoutingPolicyWithPipeline_NilPipelineMatchesLegacy(t *testing.T) {
	scorers := []ScorerConfig{{Name: "prefix-affinity", Weight: 2}, {Name: "queue-depth", Weight: 1}, {Name: "kv-utilization", Weight: 1}}
	legacy := NewRoutingPolicyWithCache("weighted", scorers, 16, rand.New(rand.NewSource(3)), nil)
	epp := NewRoutingPolicyWithPipeline("weighted", scorers, 16, rand.New(rand.NewSource(3)), nil, nil, RoutingEnv{})
	r := rand.New(rand.NewSource(9))
	for i := 0; i < 500; i++ {
		snaps := eppSnaps("a", "b", "c")
		for j := range snaps {
			snaps[j].QueueDepth = r.Intn(4)
			snaps[j].KVUtilization = float64(r.Intn(4)) / 4
		}
		req := &Request{ID: fmt.Sprint(i), InputTokens: eppTokens(16 * (1 + r.Intn(4)))}
		a, b := legacy.Route(req, &RouterState{Snapshots: snaps}), epp.Route(req, &RouterState{Snapshots: snaps})
		if a.TargetInstance != b.TargetInstance {
			t.Fatalf("request %d: legacy %q vs nil-pipeline %q", i, a.TargetInstance, b.TargetInstance)
		}
	}
}

type countingProducer struct {
	prefixProducer
	produced int
}

func (c *countingProducer) produce(r *Request, s []RoutingSnapshot, clock int64) map[string]prefixMatchInfo {
	c.produced++
	return c.prefixProducer.produce(r, s, clock)
}

// P/D timing (llm-d runs producers once per request, before any profile, and
// PreRequest once after scheduling — never on a scheduling error): with a
// deferred pipeline the decode and prefill picks produce once, record nothing
// until Commit, and nothing at all after Discard.
func TestRoutingPipeline_DeferredPreRequestAndSingleProduce(t *testing.T) {
	build := func() (*RoutingPipeline, *countingProducer, RoutingPolicy, RoutingPolicy) {
		cfg, _ := ResolveRoutingPipeline([]ScorerConfig{{Name: "token-load", Weight: 1}, {Name: "prefix-cache", Weight: 1}}, &RoutingPipelineConfig{Version: EPPVersionMain})
		p := NewRoutingPipeline(cfg, RoutingEnv{EngineBlockSize: 64}, nil)
		cp := &countingProducer{prefixProducer: p.producers[LLMDApproxPrefixProducer]}
		p.producers[LLMDApproxPrefixProducer] = cp
		p.SetDeferred(true)
		sc := []ScorerConfig{{Name: "token-load", Weight: 1}, {Name: "prefix-cache", Weight: 1}}
		return p, cp, NewWeightedRoutingOnPipeline(sc, nil, 64, rand.New(rand.NewSource(1)), nil, p),
			NewWeightedRoutingOnPipeline(sc, nil, 64, rand.New(rand.NewSource(2)), nil, p)
	}
	approx := func(p *RoutingPipeline) *approxPrefixProducer {
		return p.producers[LLMDApproxPrefixProducer].(*countingProducer).prefixProducer.(*approxPrefixProducer)
	}
	for _, commit := range []bool{true, false} {
		p, cp, decode, prefill := build()
		parent := eppReq("p", 256)
		p.BeginRequest(parent, eppSnaps("d0", "d1", "p0", "p1"), 0)
		d := decode.Route(parent, &RouterState{Snapshots: eppSnaps("d0", "d1")})
		p.AliasPrefill("p_prefill", "p")
		sub := &Request{ID: "p_prefill", InputTokens: parent.InputTokens}
		pf := prefill.Route(sub, &RouterState{Snapshots: eppSnaps("p0", "p1")})
		if cp.produced != 1 {
			t.Errorf("producers ran %d times for one request, want 1", cp.produced)
		}
		tr := p.InFlightLoadTracker("")
		if tr.Tokens(d.TargetInstance) != 0 || approx(p).idx.InstanceBlockCount(d.TargetInstance) != 0 {
			t.Fatal("PreRequest ran before the request was fully scheduled")
		}
		if commit {
			p.Commit("p")
			if tr.Tokens(d.TargetInstance) != 256 || tr.Tokens(pf.TargetInstance) != 256 ||
				approx(p).idx.InstanceBlockCount(d.TargetInstance) == 0 || approx(p).idx.InstanceBlockCount(pf.TargetInstance) == 0 {
				t.Errorf("Commit: decode %d prefill %d tokens", tr.Tokens(d.TargetInstance), tr.Tokens(pf.TargetInstance))
			}
		} else {
			p.Discard("p")
			p.Commit("p")
			if tr.Tokens(d.TargetInstance) != 0 || approx(p).idx.InstanceBlockCount(d.TargetInstance) != 0 {
				t.Error("Discard: the decode pick of a failed request must not be recorded")
			}
		}
	}
}

// A CPU-tier approx producer is llm-d's router-side LRU in both curvebender forms —
// autoTune (GPU-sized: CacheNumBlocks) and lruCapacityPerServer 30659 — credited at
// PreRequest; it ignores the CPU tier unless the BLIS opt-in marks it CPUTier.
func TestApproxProducer_CPUTierFormsAreLRUUnlessOptedIn(t *testing.T) {
	cpu := map[string]func([]TokenID) int{"x": func([]TokenID) int { return 3 }}
	env := RoutingEnv{EngineBlockSize: 64, KVBlocks: func(string) int64 { return 777 }, CPUTierCacheFn: cpu}
	auto := DefaultApproxPrefixProducerConfig("cpu-prefix-cache-producer", EPPVersionMain)
	fixed := auto
	fixed.AutoTune, fixed.LRUCapacityPerServer = false, 30659
	for _, tc := range []struct {
		cfg  PrefixProducerConfig
		want int
	}{{auto, 777}, {fixed, 30659}} {
		p := NewRoutingPipeline(&RoutingPipelineConfig{Producers: []PrefixProducerConfig{tc.cfg}}, env, nil)
		ap, ok := p.producers[tc.cfg.Name].(*approxPrefixProducer)
		if !ok {
			t.Fatalf("%+v: not an approx LRU producer", tc.cfg)
		}
		if got := ap.idx.capacityFor("x"); got != tc.want {
			t.Errorf("capacity %d, want %d", got, tc.want)
		}
		r := eppReq("r", 64*4)
		p.begin(r, eppSnaps("x"), 0)
		if info, _ := p.matchInfo(tc.cfg.Name, "x"); info.match != 0 {
			t.Errorf("before routing the LRU holds nothing, got match %d", info.match)
		}
		p.recordRouted(r, "x", 0)
		p.begin(eppReq("r2", 64*4), eppSnaps("x"), 0)
		if info, _ := p.matchInfo(tc.cfg.Name, "x"); info.match != 4 {
			t.Errorf("after routing: match %d, want 4 (credited at PreRequest)", info.match)
		}
	}
	opt := auto
	opt.CPUTier = true
	p := NewRoutingPipeline(&RoutingPipelineConfig{Producers: []PrefixProducerConfig{opt}}, env, nil)
	p.begin(eppReq("r", 64*4), eppSnaps("x"), 0)
	if info, _ := p.matchInfo(opt.Name, "x"); info.match != 3 {
		t.Errorf("opt-in CPU-tier view: match %d, want the tier's 3 blocks", info.match)
	}
}
