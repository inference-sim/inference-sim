package embedtest

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/embed"
)

// requirementsRouter is a Router that answers only Requirements.
type requirementsRouter struct {
	embed.Router
	req embed.Requirements
}

func (r requirementsRouter) Requirements() embed.Requirements { return r.req }

// The harness plays the engine in the series the router's own extractor reads, not in vLLM's
// names, so a router whose mapping differs still sees a busy engine as busy.
func TestMetricsFollowTheExtractorsSeries(t *testing.T) {
	e := &env{s: &Suite{Model: "m"}, blocks: 16}
	e.r = requirementsRouter{req: embed.Requirements{Extractors: []embed.MetricsExtractor{{
		Name: "x", DefaultEngine: "other",
		Engines: []embed.EngineMapping{
			{Name: "unused", Series: map[string]string{"queuedRequestsSpec": "unused_queue"}},
			{Name: "other", Series: map[string]string{
				"queuedRequestsSpec":  "engine_queue_depth",
				"runningRequestsSpec": "engine_running",
				"kvUsageSpec":         `engine_kv_frac{pool="a"}`,
				"cacheInfoSpec":       "engine_cache_info",
			}},
		},
	}}}}
	got := e.metrics(true)
	for _, want := range []string{`engine_queue_depth{model_name="m"} 128`, `engine_running{model_name="m"} 256`,
		`engine_kv_frac{pool="a",model_name="m"} 1`, `engine_cache_info{block_size="16",num_gpu_blocks="4096"} 1`} {
		if !strings.Contains(got, want) {
			t.Errorf("metrics lack %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "vllm:num_requests_waiting") || strings.Contains(got, "unused_queue") {
		t.Errorf("metrics hold a series the extractor does not name:\n%s", got)
	}

	// With no series in Requirements, vLLM's names are the fallback.
	e.r = requirementsRouter{}
	if got = e.metrics(false); !strings.Contains(got, `vllm:num_requests_waiting{model_name="m"} 0`) {
		t.Errorf("no fallback to vLLM's series:\n%s", got)
	}
	e.r = nil
	if got = e.metrics(false); !strings.Contains(got, "vllm:kv_cache_usage_perc") {
		t.Errorf("no fallback without a router:\n%s", got)
	}
}

// A part whose configuration the Suite leaves nil is reported as skipped, with the field
// named, and not silently passed.
func TestNilConfigsAreReportedSkipped(t *testing.T) {
	_, skipped := run(Suite{New: func(context.Context, embed.Runtime, embed.Config) (embed.Router, error) {
		return nil, errors.New("not built")
	}})
	got := map[string]string{}
	for _, s := range skipped {
		got[s.part] = s.reason
	}
	if len(skipped) != 3 || !strings.Contains(got["unavailable"], "Suite.Unavailable") || !strings.Contains(got["evict"], "Suite.Evict") || !strings.Contains(got["unaudited"], "Suite.Unaudited") {
		t.Errorf("skipped parts: %+v", skipped)
	}
}
