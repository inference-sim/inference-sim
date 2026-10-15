package llmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// RoundTrip makes the runtime serve the router's outbound HTTP: the router's requests to model
// servers reach the simulated engines, never the network.
func (rt *desRuntime) RoundTrip(r *http.Request) (*http.Response, error) {
	if rt.server == nil {
		return nil, fmt.Errorf("llmd: no simulated model servers to serve %s %s", r.Method, r.URL)
	}
	return rt.server.serve(r)
}

// modelServers plays the HTTP endpoints of the simulated model servers that the router calls:
// the vLLM Prometheus endpoint and the vLLM render endpoint.
type modelServers struct {
	g *Gateway
	// rendered caches each instance's Prometheus text by the publish time it was rendered from.
	rendered map[cluster.InstanceID]renderedMetrics
}

type renderedMetrics struct {
	publishedAt int64
	text        []byte
}

func (m *modelServers) serve(r *http.Request) (*http.Response, error) {
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/metrics":
		inst := m.g.instanceAt(r.URL.Host)
		if inst == nil {
			return respond(r, http.StatusServiceUnavailable, "text/plain", []byte("no such model server\n")), nil
		}
		return respond(r, http.StatusOK, "text/plain; version=0.0.4", m.metricsText(inst)), nil
	case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/v1/completions/render"):
		return m.renderCompletions(r)
	case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/render"):
		// BLIS requests are token-ID completions; it has no tokenizer to render chat or other
		// text prompts.
		return respond(r, http.StatusNotImplemented, "text/plain", []byte("the simulated renderer serves token-ID completions only\n")), nil
	default:
		return respond(r, http.StatusNotFound, "text/plain", []byte("not served by the simulated model server\n")), nil
	}
}

// renderCompletions answers vLLM's /v1/completions/render for a token-ID prompt: the rendered
// prompt is the token IDs themselves.
func (m *modelServers) renderCompletions(r *http.Request) (*http.Response, error) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	var req struct {
		Prompt json.RawMessage `json:"prompt"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return respond(r, http.StatusBadRequest, "text/plain", []byte(err.Error())), nil
	}
	var single []uint32
	var batch [][]uint32
	switch {
	case json.Unmarshal(req.Prompt, &single) == nil:
		batch = [][]uint32{single}
	case json.Unmarshal(req.Prompt, &batch) == nil:
	default:
		return respond(r, http.StatusNotImplemented, "text/plain", []byte("the simulated renderer serves token-ID prompts only\n")), nil
	}
	type rendered struct {
		TokenIDs []uint32 `json:"token_ids"`
	}
	out := make([]rendered, len(batch))
	for i, ids := range batch {
		out[i] = rendered{TokenIDs: ids}
	}
	b, _ := json.Marshal(out)
	return respond(r, http.StatusOK, "application/json", b), nil
}

func respond(r *http.Request, code int, contentType string, body []byte) *http.Response {
	return &http.Response{
		Status:        fmt.Sprintf("%d %s", code, http.StatusText(code)),
		StatusCode:    code,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        http.Header{"Content-Type": []string{contentType}},
		Body:          io.NopCloser(bytes.NewReader(body)),
		ContentLength: int64(len(body)),
		Request:       r,
	}
}

// metricsText renders an instance's published engine metrics as vLLM's Prometheus exposition.
func (m *modelServers) metricsText(inst *cluster.InstanceSimulator) []byte {
	em := inst.EngineMetricsAt(m.g.host.Now())
	if c, ok := m.rendered[inst.ID()]; ok && c.publishedAt == em.PublishedAt {
		return c.text
	}
	model := m.g.servedModel(inst)
	var b bytes.Buffer
	gauge := func(name string, v any) {
		fmt.Fprintf(&b, "# TYPE %s gauge\n%s{model_name=%q} %v\n", name, name, model, v)
	}
	gauge("vllm:num_requests_waiting", em.Waiting)
	gauge("vllm:num_requests_running", em.Running)
	gauge("vllm:kv_cache_usage_perc", em.KVUsage)
	fmt.Fprintf(&b, "# TYPE vllm:num_preemptions_total counter\nvllm:num_preemptions_total{model_name=%q} %d\n", model, em.Preemptions)
	fmt.Fprintf(&b, "# TYPE vllm:cache_config_info gauge\nvllm:cache_config_info{block_size=\"%d\",num_gpu_blocks=\"%d\"} 1\n",
		m.g.host.BlockSizeTokens(), inst.TotalKVBlocks())
	if maxLoRA := inst.AdapterCapacity(); maxLoRA > 0 {
		// vLLM stamps the series with the time of the update; the extractor keeps the newest.
		fmt.Fprintf(&b, "# TYPE vllm:lora_requests_info gauge\nvllm:lora_requests_info{max_lora=\"%d\",running_lora_adapters=%q,waiting_lora_adapters=%q} %v\n",
			maxLoRA, strings.Join(em.RunningAdapters, ","), strings.Join(em.WaitingAdapters, ","),
			float64(toTime(em.PublishedAt).UnixNano())/1e9)
	}
	text := b.Bytes()
	m.rendered[inst.ID()] = renderedMetrics{publishedAt: em.PublishedAt, text: text}
	return text
}

// publishedSeries are the engine series the simulated model servers expose. A configuration that
// requires another series is refused at startup.
var publishedSeries = []string{
	"vllm:num_requests_waiting", "vllm:num_requests_running", "vllm:kv_cache_usage_perc",
	"vllm:num_preemptions_total", "vllm:cache_config_info", "vllm:lora_requests_info",
}

func isPublished(series string) bool {
	name := series
	if i := strings.IndexByte(series, '{'); i >= 0 {
		name = series[:i]
	}
	i := sort.SearchStrings(sortedSeries, name)
	return i < len(sortedSeries) && sortedSeries[i] == name
}

var sortedSeries = func() []string {
	s := append([]string(nil), publishedSeries...)
	sort.Strings(s)
	return s
}()
