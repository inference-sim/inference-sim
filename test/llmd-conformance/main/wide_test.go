package conformance

import (
	"context"
	"fmt"
	"math"
	"sort"
	"testing"
	"time"

	k8stypes "k8s.io/apimachinery/pkg/types"
	"pgregory.net/rapid"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/prefixcacheaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/tokenload"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"

	"github.com/inference-sim/inference-sim/sim"
)

// The wide generators: per-endpoint block size and total, absent prefix
// attributes, tier-weighted match below the cached count, P/D role labels,
// output estimation on/off with client caps, absent inflight producer, and zero
// endpoints.

const roleLabel = "llm-d.ai/role"

type wideCase struct {
	req sim.ConformanceRequest
	eps []sim.ConformanceEndpoint
}

func drawWide(t *rapid.T, minEndpoints int, allowNoInFlight bool) wideCase {
	n := rapid.IntRange(minEndpoints, 6).Draw(t, "endpoints")
	c := wideCase{req: sim.ConformanceRequest{InputTokens: rapid.IntRange(0, 3000).Draw(t, "input")}}
	if rapid.Bool().Draw(t, "hasCap") {
		c.req.MaxOutputLen = rapid.IntRange(1, 3000).Draw(t, "maxOutput")
	}
	c.req.InFlight = sim.DefaultInFlightLoadConfig(sim.LLMDInFlightLoadProducer, version)
	c.req.InFlight.AddEstimatedOutput = rapid.Bool().Draw(t, "addEstimatedOutput")
	c.req.NoInFlight = allowNoInFlight && rapid.IntRange(0, 4).Draw(t, "noInFlight") == 0
	for i := 0; i < n; i++ {
		bs := rapid.SampledFrom([]int{1, 16, 64}).Draw(t, fmt.Sprint("bs", i))
		total := rapid.IntRange(0, 50).Draw(t, fmt.Sprint("total", i))
		cached := rapid.IntRange(0, total).Draw(t, fmt.Sprint("cached", i))
		c.eps = append(c.eps, sim.ConformanceEndpoint{
			ID:               fmt.Sprintf("default/pod%d", i),
			HasPrefixInfo:    rapid.IntRange(0, 5).Draw(t, fmt.Sprint("noPrefix", i)) != 0,
			MatchBlocks:      rapid.IntRange(0, cached).Draw(t, fmt.Sprint("match", i)), // tier-weighted ≤ cached
			TotalBlocks:      total,
			BlockSize:        bs,
			CachedBlocks:     cached,
			InFlightRequests: rapid.IntRange(0, 12).Draw(t, fmt.Sprint("req", i)),
			InFlightTokens:   rapid.Int64Range(0, 150000).Draw(t, fmt.Sprint("tok", i)),
			WaitingQueue:     rapid.IntRange(0, 20).Draw(t, fmt.Sprint("wait", i)),
			KVUtilization:    float64(rapid.IntRange(0, 100).Draw(t, fmt.Sprint("kv", i))) / 100,
			Role:             rapid.SampledFrom([]string{"", "prefill", "decode"}).Draw(t, fmt.Sprint("role", i)),
		})
	}
	return c
}

func wideEndpoints(c wideCase) []fwksched.Endpoint {
	out := make([]fwksched.Endpoint, len(c.eps))
	for i, e := range c.eps {
		meta := &fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "default", Name: e.ID[len("default/"):]}, Labels: map[string]string{}}
		if e.Role != "" {
			meta.Labels[roleLabel] = e.Role
		}
		ep := fwksched.NewEndpoint(meta, &fwkdl.Metrics{WaitingQueueSize: e.WaitingQueue, KVCacheUsagePercent: e.KVUtilization, UpdateTime: time.Now()}, nil)
		if e.HasPrefixInfo {
			ep.Put(prefixKey, attrprefix.NewPrefixCacheMatchInfo(e.MatchBlocks, e.TotalBlocks, e.BlockSize).WithCachedBlockCount(e.CachedBlocks))
		}
		if !c.req.NoInFlight {
			ep.Put(inflightKey, &attrconcurrency.InFlightLoad{Requests: int64(e.InFlightRequests), Tokens: e.InFlightTokens})
		}
		out[i] = ep
	}
	return out
}

func wideRequest(c wideCase, id string) *fwksched.InferenceRequest {
	body := &fwkrh.InferenceRequestBody{TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: make([]uint32, c.req.InputTokens)}}}}
	if c.req.MaxOutputLen > 0 {
		v := int64(c.req.MaxOutputLen)
		body.MaxOutputTokens = &v
	}
	return &fwksched.InferenceRequest{RequestID: id, Body: body}
}

func inflightProducer(t *rapid.T, add bool) *inflightload.InFlightLoadProducer {
	p, err := inflightload.InFlightLoadProducerFactory("inflight-load-producer",
		fwkplugin.StrictDecoder([]byte(fmt.Sprintf(`{"addEstimatedOutputTokens": %v}`, add))), handle())
	if err != nil {
		t.Fatal(err)
	}
	return p.(*inflightload.InFlightLoadProducer)
}

func wideProduce(t *rapid.T, c wideCase, eps []fwksched.Endpoint, req *fwksched.InferenceRequest) {
	if c.req.NoInFlight {
		return
	}
	if err := inflightProducer(t, c.req.InFlight.AddEstimatedOutput).Produce(context.Background(), req, eps); err != nil {
		t.Fatal(err)
	}
}

func wideHarness(c wideCase) *sim.ConformanceHarness {
	return sim.NewConformanceHarnessFor(version, c.req, c.eps)
}

// UncachedRequestTokens under roles, output estimation and client caps.
func TestWide_UncachedRequestTokens(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		c := drawWide(rt, 0, false)
		eps, req := wideEndpoints(c), wideRequest(c, "r")
		wideProduce(rt, c, eps, req)
		key := attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName("inflight-load-producer")
		want := wideHarness(c).UncachedTokens()
		for _, ep := range eps {
			v, _ := ep.Get(key)
			if got := v.(*attrconcurrency.UncachedRequestTokens).Tokens; got != want[ep.GetMetadata().ID.String()] {
				rt.Fatalf("%s (role %q, add %v, cap %d): llm-d %d, BLIS %d", ep.GetMetadata().ID, ep.GetMetadata().Labels[roleLabel],
					c.req.InFlight.AddEstimatedOutput, c.req.MaxOutputLen, got, want[ep.GetMetadata().ID.String()])
			}
		}
	})
}

// prefix-cache-affinity-filter with exploration: llm-d draws from the global
// math/rand source, so its result is either every endpoint (explored) or the
// deterministic gate result; BLIS's seeded draw yields one of the same two sets.
func TestWide_AffinityFilterWithExploration(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		c := drawWide(rt, 0, true)
		thr := float64(rapid.IntRange(0, 100).Draw(rt, "thr")) / 100
		expl := rapid.SampledFrom([]float64{0, 0.3, 1}).Draw(rt, "exploration")
		pen := float64(rapid.IntRange(0, 3).Draw(rt, "pen") * 1500)
		raw := fmt.Sprintf(`{"affinityThreshold": %v, "explorationProbability": %v, "maxTTFTPenaltyMs": %v, "peakPrefillThroughput": 1000}`, thr, expl, pen)
		pl, err := prefixcacheaffinity.Factory("f", fwkplugin.StrictDecoder([]byte(raw)), nil)
		if err != nil {
			rt.Fatal(err)
		}
		eps := wideEndpoints(c)
		got := names(pl.(fwksched.Filter).Filter(context.Background(), wideRequest(c, "r"), eps))
		cfg := sim.DefaultPrefixAffinityFilterConfig(version)
		cfg.AffinityThreshold, cfg.ExplorationProbability, cfg.MaxTTFTPenaltyMs, cfg.PeakPrefillThroughput = thr, 0, pen, 1000
		gated := wideHarness(c).AffinityFilter(cfg)
		all := names(eps)
		switch {
		case expl == 0 && !eqStrings(got, gated):
			rt.Fatalf("no exploration: llm-d %v, BLIS %v (%s)", got, gated, raw)
		case expl == 1 && !eqStrings(got, all):
			rt.Fatalf("always exploring: llm-d %v, want all", got)
		case !eqStrings(got, gated) && !eqStrings(got, all):
			rt.Fatalf("llm-d %v is neither the gate result %v nor all %v", got, gated, all)
		}
	})
}

// Scorers with absent attributes (no prefix info, no inflight producer).
func TestWide_ScorersWithMissingAttributes(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		c := drawWide(rt, 1, true)
		eps, req := wideEndpoints(c), wideRequest(c, "r")
		wideProduce(rt, c, eps, req)
		tl, _ := tokenload.TokenLoadScorerFactory("t", nil, nil)
		pc, _ := prefix.PrefixCachePluginFactory("p", nil, handle())
		h := wideHarness(c)
		for name, sc := range map[string]fwksched.Scorer{"token-load": tl.(fwksched.Scorer), "prefix-cache": pc.(fwksched.Scorer)} {
			got := sc.Score(context.Background(), req, eps)
			want := h.Score(name, sim.ScorerParams{})
			for _, ep := range eps {
				id := ep.GetMetadata().ID.String()
				if math.Abs(got[ep]-want[id]) > 1e-9 {
					rt.Fatalf("%s %s (noInFlight %v): llm-d %v, BLIS %v", name, id, c.req.NoInFlight, got[ep], want[id])
				}
			}
		}
	})
}

// The full profile WITH filters: concurrency guard (failOpen random) then the
// affinity filter, then token-load + prefix-cache + queue scorers and the
// max-score picker. llm-d fails the request ("no endpoints available") exactly
// when BLIS rejects it; otherwise llm-d's pick is among BLIS's tied maxima.
func TestWide_FullProfileWithFilters(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		c := drawWide(rt, 1, false)
		maxC := rapid.Int64Range(1, 12).Draw(rt, "maxConcurrency")
		failOpen := rapid.Bool().Draw(rt, "failOpen")
		thr := float64(rapid.IntRange(0, 100).Draw(rt, "thr")) / 100
		w := [3]float64{float64(rapid.IntRange(1, 4).Draw(rt, "w0")), float64(rapid.IntRange(1, 4).Draw(rt, "w1")), float64(rapid.IntRange(1, 4).Draw(rt, "w2"))}
		guard, err := concurrency.ConcurrencyDetectorFactory("g", fwkplugin.StrictDecoder([]byte(fmt.Sprintf(`{"maxConcurrency": %d, "failOpen": %v}`, maxC, failOpen))), handle())
		if err != nil {
			rt.Fatal(err)
		}
		aff, _ := prefixcacheaffinity.Factory("f", fwkplugin.StrictDecoder([]byte(fmt.Sprintf(`{"affinityThreshold": %v, "maxTTFTPenaltyMs": 4500, "peakPrefillThroughput": 15000}`, thr))), nil)
		tl, _ := tokenload.TokenLoadScorerFactory("t", nil, nil)
		pc, _ := prefix.PrefixCachePluginFactory("p", nil, handle())
		qs, _ := queuedepth.QueueScorerFactory("q", nil, handle())
		prof := scheduling.NewSchedulerProfile().
			WithFilters(guard.(fwksched.Filter), aff.(fwksched.Filter)).
			WithScorers(scheduling.NewWeightedScorer(tl.(fwksched.Scorer), w[0]), scheduling.NewWeightedScorer(pc.(fwksched.Scorer), w[1]), scheduling.NewWeightedScorer(qs.(fwksched.Scorer), w[2])).
			WithPicker(maxscore.NewMaxScorePicker(1))
		eps, req := wideEndpoints(c), wideRequest(c, "r")
		wideProduce(rt, c, eps, req)
		res, runErr := prof.Run(context.Background(), req, eps)
		gcfg := sim.ConcurrencyDetectorConfig{Mode: sim.ConcurrencyModeRequests, MaxConcurrency: maxC, MaxTokenConcurrency: 1000000,
			FailOpen: failOpen, CountIncomingTokens: true, DispatchReservations: true}
		acfg := sim.DefaultPrefixAffinityFilterConfig(version)
		acfg.AffinityThreshold, acfg.MaxTTFTPenaltyMs, acfg.PeakPrefillThroughput = thr, 4500, 15000
		maxima, rejected := wideHarness(c).Route(
			[]sim.FilterConfig{{Type: sim.LLMDConcurrencyDetector, Concurrency: gcfg}, {Type: sim.LLMDPrefixAffinityFilter, Affinity: acfg}},
			[]sim.ScorerConfig{{Name: "token-load", Weight: w[0]}, {Name: "prefix-cache", Weight: w[1]}, {Name: "queue-depth", Weight: w[2]}})
		if (runErr != nil) != rejected {
			rt.Fatalf("empty-result path: llm-d err=%v, BLIS rejected=%v", runErr, rejected)
		}
		if rejected {
			return
		}
		picked := res.TargetEndpoints[0].GetMetadata().ID.String()
		if i := sort.SearchStrings(maxima, picked); i == len(maxima) || maxima[i] != picked {
			rt.Fatalf("llm-d picked %s, BLIS maxima %v", picked, maxima)
		}
	})
}

// Concurrency detector over zero endpoints too.
func TestWide_ConcurrencyZeroEndpoints(t *testing.T) {
	pl, err := concurrency.ConcurrencyDetectorFactory("g", nil, handle())
	if err != nil {
		t.Fatal(err)
	}
	h := sim.NewConformanceHarnessFor(version, sim.ConformanceRequest{InFlight: sim.DefaultInFlightLoadConfig(sim.LLMDInFlightLoadProducer, version)}, nil)
	cfg := sim.DefaultConcurrencyDetectorConfig()
	cfg.FailOpen, cfg.CountIncomingTokens, cfg.DispatchReservations = true, true, true
	got := pl.(interface {
		Saturation(context.Context, []fwkdl.Endpoint) float64
	}).Saturation(context.Background(), nil)
	if want := h.ConcurrencySaturation(cfg, 0); got != want {
		t.Errorf("zero endpoints: llm-d %v, BLIS %v", got, want)
	}
	if f := pl.(fwksched.Filter).Filter(context.Background(), nil, nil); len(f) != 0 || len(h.ConcurrencyFilter(cfg)) != 0 {
		t.Error("zero endpoints filter to zero")
	}
}

// In-flight lifecycle: llm-d's PreRequest / ResponseBody(StartOfStream,
// EndOfStream) against BLIS's charge / first token / OnRequestDone, over random
// sequences of single-profile and P/D requests; per-endpoint in-flight tokens
// must agree after every event.
func TestWide_InFlightLifecycle(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		add := rapid.Bool().Draw(rt, "addEstimatedOutput")
		llmd := inflightProducer(rt, add)
		cfg := sim.DefaultInFlightLoadConfig(sim.LLMDInFlightLoadProducer, version)
		cfg.AddEstimatedOutput = add
		blis := sim.NewConformanceLedger(cfg)
		type live struct {
			req     *fwksched.InferenceRequest
			started bool
		}
		var open []live
		ids := map[string]bool{}
		steps := rapid.IntRange(1, 30).Draw(rt, "steps")
		for step := 0; step < steps; step++ {
			op := rapid.IntRange(0, 2).Draw(rt, fmt.Sprint("op", step))
			if op == 0 || len(open) == 0 {
				c := drawWide(rt, 2, false)
				c.req.InFlight = cfg
				eps := wideEndpoints(c)
				id := fmt.Sprint("req", step)
				req := wideRequest(c, id)
				d := rapid.IntRange(0, len(eps)-1).Draw(rt, "decode")
				pd := rapid.Bool().Draw(rt, "pd")
				results := map[string]*fwksched.ProfileRunResult{"default": {TargetEndpoints: []fwksched.Endpoint{eps[d]}}}
				primary, prefillID := "default", ""
				if pd {
					p := (d + 1) % len(eps)
					results = map[string]*fwksched.ProfileRunResult{"decode": {TargetEndpoints: []fwksched.Endpoint{eps[d]}}, "prefill": {TargetEndpoints: []fwksched.Endpoint{eps[p]}}}
					primary, prefillID = "decode", eps[p].GetMetadata().ID.String()
				}
				sr := &fwksched.SchedulingResult{PrimaryProfileName: primary, ProfileResults: results}
				req.SchedulingResult = sr
				if err := llmd.PreRequest(context.Background(), req, sr); err != nil {
					rt.Fatal(err)
				}
				blis.Route(id, wideHarness(c), eps[d].GetMetadata().ID.String(), prefillID)
				open = append(open, live{req: req})
				for _, ep := range eps {
					ids[ep.GetMetadata().ID.String()] = true
				}
			} else {
				i := rapid.IntRange(0, len(open)-1).Draw(rt, fmt.Sprint("which", step))
				l := open[i]
				if !l.started {
					llmd.ResponseBody(context.Background(), l.req, &fwkrc.Response{RequestID: l.req.RequestID, StartOfStream: true}, nil)
					blis.FirstToken(l.req.RequestID)
					open[i].started = true
				} else {
					llmd.ResponseBody(context.Background(), l.req, &fwkrc.Response{RequestID: l.req.RequestID, EndOfStream: true}, nil)
					blis.End(l.req.RequestID)
					open = append(open[:i], open[i+1:]...)
				}
			}
			for id := range ids {
				if got, want := llmd.GetTokens(id), blis.Tokens(id); got != want {
					rt.Fatalf("step %d %s (add %v): llm-d %d tokens, BLIS %d", step, id, add, got, want)
				}
			}
		}
	})
}

// prefix-based-pd-decider with tier-weighted match below the cached count: the
// decider reads CachedBlockCount (the unweighted count), not MatchBlocks.
func TestWide_PrefixBasedPDDeciderCachedCount(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		c := drawWide(rt, 1, false)
		sel := rapid.IntRange(0, len(c.eps)-1).Draw(rt, "selected")
		e := c.eps[sel]
		nonCached := c.req.InputTokens - e.CachedBlocks*e.BlockSize
		nc := rapid.SampledFrom([]int{0, nonCached, nonCached + 1, rapid.IntRange(0, 2000).Draw(rt, "ncRandom")}).Draw(rt, "nonCachedTokens")
		if nc < 0 {
			nc = 0
		}
		d, err := disagg.NewPrefixBasedPDDecider(disagg.PrefixBasedPDDeciderConfig{NonCachedTokens: nc})
		if err != nil {
			rt.Fatal(err)
		}
		h := disagg.NewDisaggProfileHandler("decode", "prefill", "encode", d, nil)
		eps := wideEndpoints(c)
		next := h.Pick(context.Background(), wideRequest(c, "r"), map[string]fwksched.SchedulerProfile{"decode": nopProfile{}, "prefill": nopProfile{}},
			map[string]*fwksched.ProfileRunResult{"decode": {TargetEndpoints: []fwksched.Endpoint{eps[sel]}}})
		_, got := next["prefill"]
		if want := wideHarness(c).PDDecide(sim.LLMDPrefixPDDeciderConfig{NonCachedTokens: nc}, e.ID); got != want {
			rt.Fatalf("llm-d %v, BLIS %v (nc %d, input %d, cached %d, match %d, bs %d, hasPrefix %v)", got, want, nc, c.req.InputTokens, e.CachedBlocks, e.MatchBlocks, e.BlockSize, e.HasPrefixInfo)
		}
	})
}
