package conformance

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"pgregory.net/rapid"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/prefixcacheaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/tokenload"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"

	"github.com/inference-sim/inference-sim/sim"
)

// The deployed routers end to end: the two deployed colocated EndpointPickerConfigs
// (sim/testdata/epp, verbatim) with the knobs the v2 campaign moves patched into them --
// the affinity filter's peakPrefillThroughput, maxTTFTPenaltyMs and affinityThreshold, the
// concurrency guard's maxConcurrency, concurrencyMode, maxTokenConcurrency and headroom.
// BLIS ingests the patched YAML through its EPP loader; llm-d's plugins are built by their
// own factories from the same parameters and run as the YAML's scheduling profile. On
// random endpoint states at agentic magnitudes (prompts to ~100k tokens, backlogs to 300k)
// llm-d's pick must be among BLIS's tied maxima, and both must reject alike.

type deployedDoc struct {
	Plugins []struct {
		Type       string         `yaml:"type"`
		Name       string         `yaml:"name"`
		Parameters map[string]any `yaml:"parameters"`
	} `yaml:"plugins"`
	SchedulingProfiles []struct {
		Plugins []struct {
			PluginRef string   `yaml:"pluginRef"`
			Weight    *float64 `yaml:"weight"`
		} `yaml:"plugins"`
	} `yaml:"schedulingProfiles"`
}

func pluginName(typ, name string) string {
	if name == "" {
		return typ
	}
	return name
}

// patchDeployed sets the drawn knobs on the parsed YAML and returns the patched text.
func patchDeployed(t *rapid.T, raw []byte) ([]byte, deployedDoc) {
	var root map[string]any
	if err := yaml.Unmarshal(raw, &root); err != nil {
		t.Fatal(err)
	}
	for _, p := range root["plugins"].([]any) {
		pm := p.(map[string]any)
		params, _ := pm["parameters"].(map[string]any)
		if params == nil {
			params = map[string]any{}
		}
		switch pm["type"] {
		case "prefix-cache-affinity-filter":
			if v := rapid.SampledFrom([]float64{-1, 1, 1500, 15000, 60000, 1e9}).Draw(t, "peak"); v >= 0 {
				params["peakPrefillThroughput"] = v
			}
			if v := rapid.SampledFrom([]float64{-1, 0, 1, 4500, 18000}).Draw(t, "penalty"); v >= 0 {
				params["maxTTFTPenaltyMs"] = v
			}
			if v := rapid.SampledFrom([]float64{-1, 0.5, 0.8}).Draw(t, "affinityThreshold"); v >= 0 {
				params["affinityThreshold"] = v
			}
		case "concurrency-detector":
			params["maxConcurrency"] = rapid.IntRange(1, 12).Draw(t, "maxConcurrency")
			params["concurrencyMode"] = rapid.SampledFrom([]string{"requests", "tokens", "hybrid"}).Draw(t, "mode")
			params["maxTokenConcurrency"] = rapid.SampledFrom([]int{8000, 64000, 260000}).Draw(t, "maxTokenConcurrency")
			params["headroom"] = rapid.SampledFrom([]float64{0, 0.1, 1}).Draw(t, "headroom")
		}
		pm["parameters"] = params
	}
	out, err := yaml.Marshal(root)
	if err != nil {
		t.Fatal(err)
	}
	var doc deployedDoc
	if err := yaml.Unmarshal(out, &doc); err != nil {
		t.Fatal(err)
	}
	return out, doc
}

func asJSON(t *rapid.T, params map[string]any) *json.Decoder {
	if len(params) == 0 {
		return nil
	}
	b, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	return fwkplugin.StrictDecoder(b)
}

func TestConformance_DeployedRoutersWithCampaignKnobs(t *testing.T) {
	for _, fixture := range []string{"precise-gpu.yaml", "optimized-baseline-cpu-tier.yaml"} {
		raw, err := os.ReadFile("../../../sim/testdata/epp/" + fixture)
		if err != nil {
			t.Fatal(err)
		}
		t.Run(fixture, func(t *testing.T) {
			rapid.Check(t, func(rt *rapid.T) {
				patched, doc := patchDeployed(rt, raw)
				in, err := sim.ParseEndpointPickerConfig(patched, 64, version)
				if err != nil {
					rt.Fatalf("BLIS ingestion: %v\n%s", err, patched)
				}

				// One state: a 64-token-block prompt, per-producer matches.
				const bs = 64
				total := rapid.IntRange(0, 1600).Draw(rt, "totalBlocks")
				input := total*bs + rapid.IntRange(0, bs-1).Draw(rt, "tail")
				n := rapid.IntRange(1, 5).Draw(rt, "endpoints")
				var producers []string
				for _, p := range doc.Plugins {
					if p.Type == "approx-prefix-cache-producer" || p.Type == "precise-prefix-cache-producer" {
						producers = append(producers, pluginName(p.Type, p.Name))
					}
				}
				matches := map[string]map[string]sim.ConformanceMatch{}
				eps := make([]sim.ConformanceEndpoint, n)
				llmdEps := make([]fwksched.Endpoint, n)
				for i := range eps {
					id := fmt.Sprintf("pod%d", i)
					eps[i] = sim.ConformanceEndpoint{ID: "default/" + id,
						InFlightRequests: rapid.IntRange(0, 14).Draw(rt, fmt.Sprint("req", i)),
						InFlightTokens:   rapid.Int64Range(0, 300000).Draw(rt, fmt.Sprint("tok", i))}
					ep := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "default", Name: id}},
						&fwkdl.Metrics{UpdateTime: time.Now()}, nil)
					for _, name := range producers {
						m := rapid.IntRange(0, total).Draw(rt, fmt.Sprintf("match_%s_%d", name, i))
						if matches[name] == nil {
							matches[name] = map[string]sim.ConformanceMatch{}
						}
						matches[name][eps[i].ID] = sim.ConformanceMatch{MatchBlocks: m, TotalBlocks: total, BlockSize: bs, CachedBlocks: m}
						ep.Put(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name).String(),
							attrprefix.NewPrefixCacheMatchInfo(m, total, bs).WithCachedBlockCount(m))
					}
					ep.Put(attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName("inflight-load-producer").String(),
						&attrconcurrency.InFlightLoad{Requests: int64(eps[i].InFlightRequests), Tokens: eps[i].InFlightTokens})
					llmdEps[i] = ep
				}

				// llm-d: the YAML's plugins by their factories, run as its default profile.
				built := map[string]fwkplugin.Plugin{}
				for _, p := range doc.Plugins {
					var pl fwkplugin.Plugin
					var err error
					name := pluginName(p.Type, p.Name)
					switch p.Type {
					case "inflight-load-producer":
						pl, err = inflightload.InFlightLoadProducerFactory(name, asJSON(rt, p.Parameters), handle())
					case "concurrency-detector":
						pl, err = concurrency.ConcurrencyDetectorFactory(name, asJSON(rt, p.Parameters), handle())
					case "prefix-cache-affinity-filter":
						pl, err = prefixcacheaffinity.Factory(name, asJSON(rt, p.Parameters), nil)
					case "token-load-scorer":
						pl, err = tokenload.TokenLoadScorerFactory(name, asJSON(rt, p.Parameters), nil)
					case "prefix-cache-scorer":
						pl, err = prefix.PrefixCachePluginFactory(name, asJSON(rt, p.Parameters), handle())
					default:
						continue // producers whose attributes the state presets; plumbing
					}
					if err != nil {
						rt.Fatalf("llm-d %s: %v", name, err)
					}
					built[name] = pl
				}
				req := &fwksched.InferenceRequest{RequestID: "conformance", Body: llmdRequest(state{input: input}).Body}
				if err := built["inflight-load-producer"].(*inflightload.InFlightLoadProducer).Produce(context.Background(), req, llmdEps); err != nil {
					rt.Fatal(err)
				}
				prof := scheduling.NewSchedulerProfile()
				var filters []fwksched.Filter
				var scorers []*scheduling.WeightedScorer
				for _, ref := range doc.SchedulingProfiles[0].Plugins {
					switch pl := built[ref.PluginRef].(type) {
					case fwksched.Filter:
						filters = append(filters, pl)
					case fwksched.Scorer:
						w := 1.0
						if ref.Weight != nil {
							w = *ref.Weight
						}
						scorers = append(scorers, scheduling.NewWeightedScorer(pl, w))
					}
				}
				prof = prof.WithFilters(filters...).WithScorers(scorers...).WithPicker(maxscore.NewMaxScorePicker(1))
				res, runErr := prof.Run(context.Background(), req, llmdEps)

				h := sim.NewConformanceHarnessFromIngest(in, sim.ConformanceRequest{InputTokens: input}, eps, matches)
				maxima, rejected := h.Route(in.Pipeline.Filters, in.Scorers)
				if (runErr != nil) != rejected {
					rt.Fatalf("empty-result path: llm-d err=%v, BLIS rejected=%v\n%s", runErr, rejected, patched)
				}
				if rejected {
					return
				}
				picked := res.TargetEndpoints[0].GetMetadata().ID.String()
				sort.Strings(maxima)
				if i := sort.SearchStrings(maxima, picked); i == len(maxima) || maxima[i] != picked {
					rt.Fatalf("llm-d picked %s, BLIS maxima %v\n%s", picked, maxima, patched)
				}
			})
		})
	}
}
