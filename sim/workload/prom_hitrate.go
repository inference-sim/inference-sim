package workload

import (
	"fmt"
	"strconv"
	"strings"
)

// Prometheus metric family names for the vLLM KV-offload tiering hit-rate signal
// (unreleased vLLM, PR #48798; constants from kv_offload/tiering/base.py). These are
// the PRIMARY signal: a true multi-tier offload hit-rate.
const (
	PromTieringBlockHits    = "vllm:kv_offload_tiering_block_hits"
	PromTieringBlockQueries = "vllm:kv_offload_tiering_block_queries"
	PromTieringReadTime     = "vllm:kv_offload_tiering_read_time"
	PromTieringWriteTime    = "vllm:kv_offload_tiering_write_time"
)

// Prometheus metric family names for the released-vLLM GPU-only prefix-cache
// fallback. This is a WEAKER signal (GPU tier only, no offload tiers) and is tagged
// distinctly in the trace header so it is never conflated with a tiered hit-rate.
const (
	PromGPUPrefixCacheHits    = "vllm:gpu_prefix_cache_hits"
	PromGPUPrefixCacheQueries = "vllm:gpu_prefix_cache_queries"
	PromGPUPrefixCacheHitRate = "vllm:gpu_prefix_cache_hit_rate"
)

// Observed-hit-rate source tags recorded in TraceObservedKVMetrics.Source.
const (
	ObservedKVSourceTiered   = "tiered"
	ObservedKVSourceGPUCache = "gpu-prefix-cache-fallback"
)

// ParsePromMetrics parses Prometheus text-exposition output into a family→value map,
// summing the value across ALL label series of each metric family (so a per-tier
// counter such as `foo{tier="cpu"} 3` + `foo{tier="disk"} 4` collapses to foo→7).
//
// It tolerates `# HELP`/`# TYPE`/comment lines, blank lines, an optional trailing
// scrape timestamp (`name{...} value [timestamp]` — only the value is read), and the
// Prometheus-client convention of appending `_total` to counter names (a `_total`
// series is folded into its base family). Malformed value fields are skipped rather
// than aborting the whole scrape (R20: degenerate input is tolerated, not fatal).
func ParsePromMetrics(text string) map[string]float64 {
	sums := make(map[string]float64)
	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		name, rest := splitPromLine(line)
		if name == "" || rest == "" {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			continue
		}
		v, err := strconv.ParseFloat(fields[0], 64)
		if err != nil {
			continue // not a numeric sample line
		}
		// Fold the counter `_total` convention into the base family name so callers
		// can look up either spelling with one key.
		name = strings.TrimSuffix(name, "_total")
		sums[name] += v
	}
	return sums
}

// splitPromLine splits a Prometheus sample line into its metric name and the
// remainder (the value, plus any labels having been consumed). Handles the two
// shapes: `name value` and `name{label="x",...} value`.
func splitPromLine(line string) (name, rest string) {
	brace := strings.IndexByte(line, '{')
	space := strings.IndexByte(line, ' ')
	// Labels present and precede the first space → name ends at '{', rest starts
	// after the matching '}'.
	if brace >= 0 && (space < 0 || brace < space) {
		close := strings.IndexByte(line, '}')
		if close < 0 || close < brace {
			return "", ""
		}
		return line[:brace], strings.TrimSpace(line[close+1:])
	}
	if space < 0 {
		return "", ""
	}
	return line[:space], strings.TrimSpace(line[space:])
}

// DeriveObservedHitRate computes the observed KV-cache hit-rate over a measurement
// window from two Prometheus scrapes (start and end of the measured workload),
// returning a resolved TraceObservedKVMetrics block for the trace header (#1583).
//
// Precedence: the tiered offload family (PRIMARY) is used when its queries family is
// present; otherwise the GPU prefix-cache family (FALLBACK, weaker) is used. Within a
// family, all values are DELTAS (end − start) so cumulative counters yield the rate
// over the measured window.
//
// Errors (the CLI caller warns and omits the block — BC-11/BC-13, never fatal, never a
// bogus value):
//   - no recognized counter family present in the scrape;
//   - a counter went backwards (delta < 0 — e.g. a server restart mid-window);
//   - zero queries in the window (division guard, R11) with no gauge fallback.
func DeriveObservedHitRate(start, end map[string]float64, vllmCommit string) (*TraceObservedKVMetrics, error) {
	// PRIMARY: tiered offload family (present iff the queries family appears).
	if _, ok := end[PromTieringBlockQueries]; ok {
		queries, qErr := familyDelta(start, end, PromTieringBlockQueries)
		if qErr != nil {
			return nil, qErr
		}
		hits, hErr := familyDelta(start, end, PromTieringBlockHits)
		if hErr != nil {
			return nil, hErr
		}
		if queries <= 0 {
			return nil, fmt.Errorf("tiered KV counters show zero block_queries over the measured window (%.0f); cannot derive a hit-rate", queries)
		}
		if hits > queries {
			return nil, fmt.Errorf("tiered KV counters report more block_hits (%.0f) than block_queries (%.0f) over the window (hit-rate > 1 is impossible; likely a scrape/counter anomaly); refusing to record a bogus value", hits, queries)
		}
		readDelta, _ := familyDelta(start, end, PromTieringReadTime)
		writeDelta, _ := familyDelta(start, end, PromTieringWriteTime)
		return &TraceObservedKVMetrics{
			Source:         ObservedKVSourceTiered,
			HitRate:        hits / queries,
			BlockHits:      int64(hits + 0.5),
			BlockQueries:   int64(queries + 0.5),
			ReadTimeTotal:  readDelta,
			WriteTimeTotal: writeDelta,
			VLLMCommit:     vllmCommit,
		}, nil
	}

	// FALLBACK: GPU prefix-cache counters (delta), then the hit_rate gauge (end value).
	if _, ok := end[PromGPUPrefixCacheQueries]; ok {
		queries, qErr := familyDelta(start, end, PromGPUPrefixCacheQueries)
		if qErr != nil {
			return nil, qErr
		}
		hits, hErr := familyDelta(start, end, PromGPUPrefixCacheHits)
		if hErr != nil {
			return nil, hErr
		}
		if queries <= 0 {
			return nil, fmt.Errorf("GPU prefix-cache counters show zero queries over the measured window (%.0f); cannot derive a hit-rate", queries)
		}
		if hits > queries {
			return nil, fmt.Errorf("GPU prefix-cache counters report more hits (%.0f) than queries (%.0f) over the window (hit-rate > 1 is impossible); refusing to record a bogus value", hits, queries)
		}
		return &TraceObservedKVMetrics{
			Source:       ObservedKVSourceGPUCache,
			HitRate:      hits / queries,
			BlockHits:    int64(hits + 0.5),
			BlockQueries: int64(queries + 0.5),
			VLLMCommit:   vllmCommit,
		}, nil
	}
	if rate, ok := end[PromGPUPrefixCacheHitRate]; ok {
		// Gauge: a point-in-time rate, not a windowed delta. The weakest signal.
		return &TraceObservedKVMetrics{
			Source:     ObservedKVSourceGPUCache,
			HitRate:    rate,
			VLLMCommit: vllmCommit,
		}, nil
	}

	return nil, fmt.Errorf("no recognized KV cache counters found in /metrics (looked for %s and %s families)",
		PromTieringBlockQueries, PromGPUPrefixCacheQueries)
}

// familyDelta returns end[name] − start[name], erroring if the counter went backwards
// (a reset/restart makes a windowed delta meaningless). A missing start value is
// treated as 0 (the counter first appeared during the window).
func familyDelta(start, end map[string]float64, name string) (float64, error) {
	delta := end[name] - start[name]
	if delta < 0 {
		return 0, fmt.Errorf("counter %s went backwards over the measured window (start=%.0f end=%.0f, likely a server restart); cannot derive a windowed hit-rate",
			name, start[name], end[name])
	}
	return delta, nil
}
