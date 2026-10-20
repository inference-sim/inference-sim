// Package conformance runs llm-d-router (main@981acb8a) plugin code and BLIS's ports
// on the same randomized endpoint states and requires identical outputs.
package conformance

import (
	"context"
	"fmt"
	"math"
	"sort"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"pgregory.net/rapid"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/prefixcacheaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/activerequest"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/kvcacheutilization"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/nohitlru"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/tokenload"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"

	"github.com/inference-sim/inference-sim/sim"
)

const version = sim.EPPVersionMain

var (
	prefixKey   = attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("")
	inflightKey = attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName("")
)

// state is one randomized routing situation.
type state struct {
	input int
	eps   []sim.ConformanceEndpoint
}

func drawState(t *rapid.T) state {
	n := rapid.IntRange(1, 6).Draw(t, "endpoints")
	bs := rapid.SampledFrom([]int{1, 16, 64}).Draw(t, "blockSize")
	total := rapid.IntRange(0, 40).Draw(t, "totalBlocks")
	input := total*bs + rapid.IntRange(0, 3*bs).Draw(t, "tail")
	s := state{input: input}
	for i := 0; i < n; i++ {
		m := rapid.IntRange(0, total).Draw(t, fmt.Sprintf("match%d", i))
		s.eps = append(s.eps, sim.ConformanceEndpoint{
			ID:            fmt.Sprintf("default/pod%d", i),
			HasPrefixInfo: true,
			MatchBlocks:   m, TotalBlocks: total, BlockSize: bs, CachedBlocks: m,
			InFlightRequests: rapid.IntRange(0, 12).Draw(t, fmt.Sprintf("req%d", i)),
			InFlightTokens:   rapid.Int64Range(0, 150000).Draw(t, fmt.Sprintf("tok%d", i)),
			WaitingQueue:     rapid.IntRange(0, 20).Draw(t, fmt.Sprintf("wait%d", i)),
			RunningRequests:  rapid.IntRange(0, 20).Draw(t, fmt.Sprintf("run%d", i)),
			KVUtilization:    float64(rapid.IntRange(0, 100).Draw(t, fmt.Sprintf("kv%d", i))) / 100,
		})
	}
	return s
}

func handle() fwkplugin.Handle {
	return fwkplugin.NewEppHandle(context.Background(), func() []k8stypes.NamespacedName { return nil },
		fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))
}

// llmdEndpoints builds llm-d scheduling endpoints carrying the state's attributes.
func llmdEndpoints(s state) []fwksched.Endpoint {
	out := make([]fwksched.Endpoint, len(s.eps))
	for i, e := range s.eps {
		ep := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "default", Name: e.ID[len("default/"):]}},
			&fwkdl.Metrics{WaitingQueueSize: e.WaitingQueue, RunningRequestsSize: e.RunningRequests, KVCacheUsagePercent: e.KVUtilization, UpdateTime: time.Now()}, nil)
		if e.HasPrefixInfo {
			ep.Put(prefixKey, attrprefix.NewPrefixCacheMatchInfo(e.MatchBlocks, e.TotalBlocks, e.BlockSize).WithCachedBlockCount(e.CachedBlocks))
		}
		ep.Put(inflightKey, &attrconcurrency.InFlightLoad{Requests: int64(e.InFlightRequests), Tokens: e.InFlightTokens})
		out[i] = ep
	}
	return out
}

func llmdRequest(s state) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{RequestID: "conformance", Body: &fwkrh.InferenceRequestBody{
		TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: make([]uint32, s.input)}}}}}
}

// produceUncached runs llm-d's inflight-load-producer Produce so endpoints carry
// UncachedRequestTokens exactly as in the EPP.
func produceUncached(t *testing.T, s state, eps []fwksched.Endpoint, req *fwksched.InferenceRequest) {
	p, err := inflightload.InFlightLoadProducerFactory("inflight-load-producer", nil, handle())
	if err != nil {
		t.Fatal(err)
	}
	if err := p.(*inflightload.InFlightLoadProducer).Produce(context.Background(), req, eps); err != nil {
		t.Fatal(err)
	}
}

func names(eps []fwksched.Endpoint) []string {
	out := make([]string, len(eps))
	for i, e := range eps {
		out[i] = e.GetMetadata().ID.String()
	}
	return out
}

func harness(s state) *sim.ConformanceHarness {
	return sim.NewConformanceHarness(version, s.input, s.eps, sim.DefaultInFlightLoadConfig(sim.LLMDInFlightLoadProducer, version))
}

func eqStrings(a, b []string) bool { return fmt.Sprint(a) == fmt.Sprint(b) }

// --- per-plugin differential properties ---

func TestConformance_UncachedRequestTokens(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		s := drawState(rt)
		eps, req := llmdEndpoints(s), llmdRequest(s)
		produceUncached(t, s, eps, req)
		key := attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName("inflight-load-producer")
		blis := harness(s).UncachedTokens()
		for _, ep := range eps {
			v, _ := ep.Get(key)
			got := v.(*attrconcurrency.UncachedRequestTokens).Tokens
			if want := blis[ep.GetMetadata().ID.String()]; got != want {
				rt.Fatalf("%s: llm-d %d, BLIS %d", ep.GetMetadata().ID, got, want)
			}
		}
	})
}

func TestConformance_PrefixCacheAffinityFilter(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		s := drawState(rt)
		thr := float64(rapid.IntRange(0, 100).Draw(rt, "thr")) / 100
		if total := s.eps[0].TotalBlocks; total > 0 && rapid.Bool().Draw(rt, "thrAtARatio") {
			thr = float64(rapid.IntRange(0, total).Draw(rt, "thrBlocks")) / float64(total) // hit ratio == threshold
		}
		pen := float64(rapid.IntRange(0, 3).Draw(rt, "pen") * 1500)
		peak := float64(rapid.IntRange(1, 40).Draw(rt, "peak") * 1000)
		if rapid.Bool().Draw(rt, "penAtAGap") {
			// peak 1000 tok/s makes TTFT_ms == in-flight tokens, so a gap between two
			// endpoints' loads hits penalty == maxTTFTPenaltyMs exactly.
			i := rapid.IntRange(0, len(s.eps)-1).Draw(rt, "gapI")
			j := rapid.IntRange(0, len(s.eps)-1).Draw(rt, "gapJ")
			peak, pen = 1000, math.Abs(float64(s.eps[i].InFlightTokens-s.eps[j].InFlightTokens))
		}
		raw := fmt.Sprintf(`{"affinityThreshold": %v, "maxTTFTPenaltyMs": %v, "peakPrefillThroughput": %v}`, thr, pen, peak)
		pl, err := prefixcacheaffinity.Factory("f", fwkplugin.StrictDecoder([]byte(raw)), nil)
		if err != nil {
			rt.Fatal(err)
		}
		eps := llmdEndpoints(s)
		got := names(pl.(fwksched.Filter).Filter(context.Background(), llmdRequest(s), eps))
		cfg := sim.DefaultPrefixAffinityFilterConfig(version)
		cfg.AffinityThreshold, cfg.MaxTTFTPenaltyMs, cfg.PeakPrefillThroughput = thr, pen, peak
		want := harness(s).AffinityFilter(cfg)
		if !eqStrings(got, want) {
			rt.Fatalf("filter: llm-d %v, BLIS %v (%s)", got, want, raw)
		}
	})
}

func TestConformance_ConcurrencyDetector(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		s := drawState(rt)
		mode := rapid.SampledFrom([]string{"requests", "tokens", "hybrid"}).Draw(rt, "mode")
		maxC := rapid.Int64Range(1, 12).Draw(rt, "maxConcurrency")
		maxT := rapid.Int64Range(1, 160000).Draw(rt, "maxTokenConcurrency")
		head := float64(rapid.IntRange(0, 4).Draw(rt, "headroom")) / 4
		failOpen := rapid.Bool().Draw(rt, "failOpen")
		raw := fmt.Sprintf(`{"concurrencyMode": %q, "maxConcurrency": %d, "maxTokenConcurrency": %d, "headroom": %v, "failOpen": %v}`, mode, maxC, maxT, head, failOpen)
		pl, err := concurrency.ConcurrencyDetectorFactory("g", fwkplugin.StrictDecoder([]byte(raw)), handle())
		if err != nil {
			rt.Fatal(err)
		}
		eps, req := llmdEndpoints(s), llmdRequest(s)
		produceUncached(t, s, eps, req)
		got := names(pl.(fwksched.Filter).Filter(context.Background(), req, eps))
		cfg := sim.ConcurrencyDetectorConfig{Mode: sim.ConcurrencyMode(mode), MaxConcurrency: maxC, MaxTokenConcurrency: maxT, Headroom: head,
			FailOpen: failOpen, CountIncomingTokens: true, DispatchReservations: true}
		h := harness(s)
		if want := h.ConcurrencyFilter(cfg); !eqStrings(got, want) {
			rt.Fatalf("filter: llm-d %v, BLIS %v (%s)", got, want, raw)
		}
		dlEps := make([]fwkdl.Endpoint, len(s.eps))
		for i, e := range s.eps {
			ms := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "default", Name: e.ID[len("default/"):]}}, &fwkdl.Metrics{})
			ms.GetAttributes().Put(inflightKey, &attrconcurrency.InFlightLoad{Requests: int64(e.InFlightRequests), Tokens: e.InFlightTokens})
			dlEps[i] = ms
		}
		res := rapid.IntRange(0, 3).Draw(rt, "reservations")
		sd := pl.(interface {
			Saturation(context.Context, []fwkdl.Endpoint) float64
			ReserveDispatch(string) bool
		})
		for i := 0; i < res; i++ {
			sd.ReserveDispatch(fmt.Sprint("r", i))
		}
		gotSat := sd.Saturation(context.Background(), dlEps)
		if wantSat := h.ConcurrencySaturation(cfg, res); math.Abs(gotSat-wantSat) > 1e-12 {
			rt.Fatalf("saturation: llm-d %v, BLIS %v (%s, reservations %d)", gotSat, wantSat, raw, res)
		}
	})
}

type scorerCase struct {
	blis   string
	params sim.ScorerParams
	llmd   func() (fwksched.Scorer, error)
}

func scorerCases() []scorerCase {
	return []scorerCase{
		{"token-load", sim.ScorerParams{}, func() (fwksched.Scorer, error) {
			p, err := tokenload.TokenLoadScorerFactory("t", nil, nil)
			return asScorer(p, err)
		}},
		{"token-load", sim.ScorerParams{QueueThresholdTokens: 50000}, func() (fwksched.Scorer, error) {
			p, err := tokenload.TokenLoadScorerFactory("t", fwkplugin.StrictDecoder([]byte(`{"queueThresholdTokens": 50000}`)), nil)
			return asScorer(p, err)
		}},
		{"prefix-cache", sim.ScorerParams{}, func() (fwksched.Scorer, error) {
			p, err := prefix.PrefixCachePluginFactory("p", nil, handle())
			return asScorer(p, err)
		}},
		{"prefix-cache", sim.ScorerParams{MatchLengthWeight: 0.5, MatchLengthScaleTokens: 512}, func() (fwksched.Scorer, error) {
			p, err := prefix.PrefixCachePluginFactory("p", fwkplugin.StrictDecoder([]byte(`{"matchLengthWeight": 0.5, "matchLengthScaleTokens": 512}`)), handle())
			return asScorer(p, err)
		}},
		{"queue-depth", sim.ScorerParams{}, func() (fwksched.Scorer, error) {
			p, err := queuedepth.QueueScorerFactory("q", nil, handle())
			return asScorer(p, err)
		}},
		{"kv-utilization", sim.ScorerParams{}, func() (fwksched.Scorer, error) {
			p, err := kvcacheutilization.KvCacheUtilizationScorerFactory("k", nil, handle())
			return asScorer(p, err)
		}},
		{"active-requests", sim.ScorerParams{}, func() (fwksched.Scorer, error) {
			p, err := activerequest.Factory("a", nil, handle())
			return asScorer(p, err)
		}},
	}
}

func asScorer(p fwkplugin.Plugin, err error) (fwksched.Scorer, error) {
	if err != nil {
		return nil, err
	}
	return p.(fwksched.Scorer), nil
}

func TestConformance_Scorers(t *testing.T) {
	for _, sc := range scorerCases() {
		t.Run(sc.blis, func(t *testing.T) {
			rapid.Check(t, func(rt *rapid.T) {
				s := drawState(rt)
				scorer, err := sc.llmd()
				if err != nil {
					rt.Fatal(err)
				}
				eps, req := llmdEndpoints(s), llmdRequest(s)
				produceUncached(t, s, eps, req)
				got := scorer.Score(context.Background(), req, eps)
				want := harness(s).Score(sc.blis, sc.params)
				for _, ep := range eps {
					id := ep.GetMetadata().ID.String()
					if math.Abs(got[ep]-want[id]) > 1e-9 {
						rt.Fatalf("%s %s: llm-d %v, BLIS %v", sc.blis, id, got[ep], want[id])
					}
				}
			})
		})
	}
}

// The full llm-d scheduling profile (filters, weighted scorers, clamp, max-score
// picker) chooses an endpoint among BLIS's tied maxima for the same weights.
// Tie-break: llm-d shuffles a map-ordered candidate list and rotates ties by a
// request counter; BLIS draws a seeded uniform pick — compared as sets.
func TestConformance_ProfilePick(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		s := drawState(rt)
		w1 := float64(rapid.IntRange(1, 5).Draw(rt, "w1"))
		w2 := float64(rapid.IntRange(1, 5).Draw(rt, "w2"))
		tl, _ := tokenload.TokenLoadScorerFactory("t", nil, nil)
		pc, _ := prefix.PrefixCachePluginFactory("p", nil, handle())
		prof := scheduling.NewSchedulerProfile().
			WithScorers(scheduling.NewWeightedScorer(tl.(fwksched.Scorer), w1), scheduling.NewWeightedScorer(pc.(fwksched.Scorer), w2)).
			WithPicker(maxscore.NewMaxScorePicker(1))
		eps, req := llmdEndpoints(s), llmdRequest(s)
		produceUncached(t, s, eps, req)
		res, err := prof.Run(context.Background(), req, eps)
		if err != nil {
			rt.Fatal(err)
		}
		picked := res.TargetEndpoints[0].GetMetadata().ID.String()
		blis := harness(s).MaxScoreCandidates([]sim.ScorerConfig{{Name: "token-load", Weight: w1}, {Name: "prefix-cache", Weight: w2}})
		i := sort.SearchStrings(blis, picked)
		if i == len(blis) || blis[i] != picked {
			rt.Fatalf("llm-d picked %s, BLIS maxima %v", picked, blis)
		}
	})
}

type nopProfile struct{}

func (nopProfile) Run(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
	return nil, nil
}

// prefix-based-pd-decider through llm-d's disagg-profile-handler (decode-first):
// after the decode profile picks an endpoint, the handler asks for the prefill
// profile iff the decider disaggregates.
func TestConformance_PrefixBasedPDDecider(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		s := drawState(rt)
		sel := rapid.IntRange(0, len(s.eps)-1).Draw(rt, "selected")
		e := s.eps[sel]
		nonCached := s.input - e.CachedBlocks*e.BlockSize
		nc := rapid.SampledFrom([]int{0, nonCached, nonCached + 1, rapid.IntRange(0, 200).Draw(rt, "ncRandom")}).Draw(rt, "nonCachedTokens")
		if nc < 0 {
			nc = 0
		}
		pt := rapid.SampledFrom([]int{0, s.input, s.input + 1, rapid.IntRange(0, 200).Draw(rt, "ptRandom")}).Draw(rt, "promptTokens")
		d, err := disagg.NewPrefixBasedPDDecider(disagg.PrefixBasedPDDeciderConfig{NonCachedTokens: nc, PromptTokens: pt})
		if err != nil {
			rt.Fatal(err)
		}
		h := disagg.NewDisaggProfileHandler("decode", "prefill", "encode", d, nil)
		eps, req := llmdEndpoints(s), llmdRequest(s)
		profiles := map[string]fwksched.SchedulerProfile{"decode": nopProfile{}, "prefill": nopProfile{}}
		results := map[string]*fwksched.ProfileRunResult{"decode": {TargetEndpoints: []fwksched.Endpoint{eps[sel]}}}
		next := h.Pick(context.Background(), req, profiles, results)
		_, got := next["prefill"]
		want := harness(s).PDDecide(sim.LLMDPrefixPDDeciderConfig{NonCachedTokens: nc, PromptTokens: pt}, s.eps[sel].ID)
		if got != want {
			rt.Fatalf("disaggregate: llm-d %v, BLIS %v (nc=%d pt=%d input=%d cached=%d bs=%d)", got, want, nc, pt, s.input, s.eps[sel].CachedBlocks, s.eps[sel].BlockSize)
		}
	})
}

// no-hit-lru-scorer over a request sequence: per step, identical scores; both
// sides then record the same target (llm-d's first maximum in candidate order).
func TestConformance_NoHitLRUSequence(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(1, 5).Draw(rt, "endpoints")
		size := rapid.IntRange(1, 4).Draw(rt, "lruSize")
		llmd := nohitlru.NewNoHitLRU(context.Background(), "n", &nohitlru.Parameters{LRUSize: size})
		blis := sim.NewConformanceNoHitLRU(size)
		steps := rapid.IntRange(1, 25).Draw(rt, "steps")
		for step := 0; step < steps; step++ {
			s := state{input: 64}
			for i := 0; i < n; i++ {
				m := 0
				if rapid.IntRange(0, 4).Draw(rt, fmt.Sprintf("warm%d_%d", step, i)) == 0 {
					m = 1
				}
				s.eps = append(s.eps, sim.ConformanceEndpoint{ID: fmt.Sprintf("default/pod%d", i), HasPrefixInfo: true, MatchBlocks: m, TotalBlocks: 1, BlockSize: 64, CachedBlocks: m})
			}
			eps := llmdEndpoints(s)
			req := llmdRequest(s)
			req.RequestID = fmt.Sprint("req", step)
			got := llmd.Score(context.Background(), req, eps)
			want := blis.Score(req.RequestID, s.eps)
			target := eps[0]
			for _, ep := range eps {
				id := ep.GetMetadata().ID.String()
				if math.Abs(got[ep]-want[id]) > 1e-12 {
					rt.Fatalf("step %d %s: llm-d %v, BLIS %v", step, id, got[ep], want[id])
				}
				if got[ep] > got[target] {
					target = ep
				}
			}
			res := &fwksched.SchedulingResult{PrimaryProfileName: "default", ProfileResults: map[string]*fwksched.ProfileRunResult{"default": {TargetEndpoints: []fwksched.Endpoint{target}}}}
			if err := llmd.PreRequest(context.Background(), req, res); err != nil {
				rt.Fatal(err)
			}
			blis.Record(req.RequestID, target.GetMetadata().ID.String())
		}
	})
}
