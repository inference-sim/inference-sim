package cmd

import (
	"fmt"
	"sort"
	"strings"
	"time"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// parseSLODurationFlag parses comma-separated key=duration pairs (e.g.
// "critical=100ms,standard=500ms"). Empty input returns (nil, nil) — the
// caller treats this as "no signal at this tier" in the merger. Empty class
// keys, unparseable durations, and non-positive durations return an error
// so cmd/ wrappers can logrus.Fatalf with consistent context.
func parseSLODurationFlag(s string) (map[string]time.Duration, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	out := make(map[string]time.Duration)
	for _, pair := range strings.Split(s, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			return nil, fmt.Errorf("empty pair in %q (expected key=duration)", s)
		}
		parts := strings.SplitN(pair, "=", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid pair %q (expected key=duration)", pair)
		}
		key := strings.TrimSpace(parts[0])
		if key == "" {
			return nil, fmt.Errorf("empty class key in pair %q", pair)
		}
		d, err := time.ParseDuration(strings.TrimSpace(parts[1]))
		if err != nil {
			return nil, fmt.Errorf("unparseable duration in pair %q: %w", pair, err)
		}
		if d <= 0 {
			return nil, fmt.Errorf("non-positive duration in pair %q", pair)
		}
		out[key] = d
	}
	return out, nil
}

// mergeGoodputTargets composes the resolved per-class SLODimTargets map from
// up to three sources, with precedence: CLI > traceHeader > spec. Each
// dimension is merged independently; a class is included if it appears in ANY
// source. A nil source map means "no signal at that tier".
//
// Determinism (R2): the result map's iteration is up to the caller, but the
// classes considered come from a sorted union so the per-class merge order
// is stable for any future logging.
func mergeGoodputTargets(
	cliTTFT, cliITL, cliE2E map[string]time.Duration,
	traceHeader, spec map[string]workload.SLODimTargets,
) map[string]workload.SLODimTargets {
	classes := make(map[string]struct{})
	for k := range cliTTFT {
		classes[k] = struct{}{}
	}
	for k := range cliITL {
		classes[k] = struct{}{}
	}
	for k := range cliE2E {
		classes[k] = struct{}{}
	}
	for k := range traceHeader {
		classes[k] = struct{}{}
	}
	for k := range spec {
		classes[k] = struct{}{}
	}
	if len(classes) == 0 {
		return nil
	}

	// Sort the union for stable ordering of any future log/diagnostic output.
	keys := make([]string, 0, len(classes))
	for k := range classes {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	out := make(map[string]workload.SLODimTargets, len(keys))
	for _, cls := range keys {
		var t workload.SLODimTargets
		if v, ok := spec[cls]; ok {
			t = v
		}
		if v, ok := traceHeader[cls]; ok {
			if v.TTFTMs > 0 {
				t.TTFTMs = v.TTFTMs
			}
			if v.ITLMs > 0 {
				t.ITLMs = v.ITLMs
			}
			if v.E2EMs > 0 {
				t.E2EMs = v.E2EMs
			}
		}
		if d, ok := cliTTFT[cls]; ok {
			t.TTFTMs = float64(d) / float64(time.Millisecond)
		}
		if d, ok := cliITL[cls]; ok {
			t.ITLMs = float64(d) / float64(time.Millisecond)
		}
		if d, ok := cliE2E[cls]; ok {
			t.E2EMs = float64(d) / float64(time.Millisecond)
		}
		// Drop a class entirely when no dimension survived the merge — this
		// can only happen if a tier carried an explicit zero, which is
		// effectively "not gated" anyway.
		if t.TTFTMs == 0 && t.ITLMs == 0 && t.E2EMs == 0 {
			continue
		}
		out[cls] = t
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// emitGoodput populates MetricsOutput goodput fields from aggregated metrics,
// per-class injection counts, and the resolved targets. No-op when targets is
// empty (BC-3: byte-identical output when goodput is not configured).
//
// The per_class payload is built as map[string]map[string]any so it serializes
// cleanly through MetricsOutput.PerClass (typed as interface{} to avoid
// importing sim/cluster from sim).
func emitGoodput(
	output *sim.MetricsOutput,
	aggregated *sim.Metrics,
	injectedByClass map[string]int64,
	runtimeSec float64,
	targets map[string]workload.SLODimTargets,
) {
	if output == nil || aggregated == nil || len(targets) == 0 {
		return
	}
	results := cluster.BuildLatencyResults(aggregated)
	overall, perClass := cluster.SLOAttainmentMultiDim(results, injectedByClass, targets)

	totalGood := 0
	classKeys := make([]string, 0, len(perClass))
	for k := range perClass {
		classKeys = append(classKeys, k)
	}
	sort.Strings(classKeys)

	per := make(map[string]map[string]any, len(perClass))
	for _, cls := range classKeys {
		sca := perClass[cls]
		totalGood += sca.Good

		entry := map[string]any{
			"slo_attainment": safeRatio(int64(sca.Good), sca.Injected),
			"count":          sca.Injected,
		}
		if runtimeSec > 0 {
			entry["goodput_rps"] = float64(sca.Good) / runtimeSec
		} else {
			entry["goodput_rps"] = 0.0
		}

		// slo_attainment_by_dim: deterministic iteration via sorted keys.
		if len(sca.ByDim) > 0 {
			dimKeys := make([]string, 0, len(sca.ByDim))
			for d := range sca.ByDim {
				dimKeys = append(dimKeys, d)
			}
			sort.Strings(dimKeys)
			byDim := make(map[string]float64, len(sca.ByDim))
			for _, d := range dimKeys {
				byDim[d] = sca.ByDim[d]
			}
			entry["slo_attainment_by_dim"] = byDim
		}

		// Per-class TTFT P99, mean ITL, E2E P99, count from aggregated metrics.
		ttftP99, itlMean, e2eP99 := perClassLatencyStats(aggregated, cls)
		if ttftP99 > 0 {
			entry["ttft_p99_ms"] = ttftP99
		}
		if itlMean > 0 {
			entry["itl_mean_ms"] = itlMean
		}
		if e2eP99 > 0 {
			entry["e2e_p99_ms"] = e2eP99
		}

		per[cls] = entry
	}

	output.SLOAttainment = overall
	output.PerClass = per
	if runtimeSec > 0 {
		output.GoodputRPS = float64(totalGood) / runtimeSec
	}
}

// perClassLatencyStats computes per-class TTFT P99, mean ITL, and E2E P99
// from aggregated metrics. SLOClass is read from m.Requests; an empty class
// folds into "default" to mirror cluster.SLOAttainmentMultiDim's lookup.
//
// Returned values are in milliseconds (consistent with MetricsOutput's other
// latency fields). All accumulation iterates request IDs in sorted order
// (R2 / INV-6 determinism).
func perClassLatencyStats(m *sim.Metrics, cls string) (ttftP99, itlMean, e2eP99 float64) {
	if m == nil {
		return 0, 0, 0
	}
	ids := make([]string, 0, len(m.RequestE2Es))
	for id := range m.RequestE2Es {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	var ttftUs, e2eUs []float64
	var itlSumUs float64
	var itlCount int
	for _, id := range ids {
		rm, ok := m.Requests[id]
		if !ok {
			continue
		}
		c := rm.SLOClass
		if c == "" {
			c = "default"
		}
		if c != cls {
			continue
		}
		if e2e, ok := m.RequestE2Es[id]; ok {
			e2eUs = append(e2eUs, e2e)
		}
		if t, ok := m.RequestTTFTs[id]; ok {
			ttftUs = append(ttftUs, t)
		}
		if i, ok := m.RequestITLs[id]; ok {
			itlSumUs += i
			itlCount++
		}
	}

	sort.Float64s(ttftUs)
	sort.Float64s(e2eUs)
	ttftP99 = sim.CalculatePercentile(ttftUs, 99)
	e2eP99 = sim.CalculatePercentile(e2eUs, 99)
	if itlCount > 0 {
		itlMean = (itlSumUs / float64(itlCount)) / 1e3 // µs → ms
	}
	return ttftP99, itlMean, e2eP99
}

// safeRatio returns num/denom or 0 when denom is zero.
func safeRatio(num, denom int64) float64 {
	if denom == 0 {
		return 0
	}
	return float64(num) / float64(denom)
}

// emitObserveGoodput populates goodput fields on a MetricsOutput from real-server
// trace records (and optional ITL records). Used by `blis observe` to mirror the
// run/replay JSON shape over a record-driven (rather than DES-driven) data path.
//
// Numerator semantics (BC-2 on observe path):
//   - Denominator per class: count of dispatched records with Status ∈ {ok, error, timeout}.
//   - Numerator per class:  records with Status == "ok" whose TTFT/E2E (and ITL when
//     configured) meet every non-zero threshold for that class. error and timeout
//     records are counted in the denominator (consistent with INV-1) but never count
//     toward goodput.
//
// runtimeSec is wall-clock duration in seconds; goodput_rps = good / runtimeSec.
// No-op when targets is empty (BC-3).
func emitObserveGoodput(
	output *sim.MetricsOutput,
	records []workload.TraceRecord,
	itlRecords []workload.ITLRecord,
	runtimeSec float64,
	targets map[string]workload.SLODimTargets,
) {
	if output == nil || len(targets) == 0 {
		return
	}

	// Index ITL records by RequestID and compute per-request mean ITL in ms.
	itlMeanByRequest := make(map[int]float64)
	if len(itlRecords) > 0 {
		grouped := make(map[int][]workload.ITLRecord)
		for _, r := range itlRecords {
			grouped[r.RequestID] = append(grouped[r.RequestID], r)
		}
		for id, chunks := range grouped {
			if len(chunks) < 2 {
				continue
			}
			sort.Slice(chunks, func(i, j int) bool { return chunks[i].ChunkIndex < chunks[j].ChunkIndex })
			var sumDeltaUs float64
			var n int
			for i := 1; i < len(chunks); i++ {
				d := chunks[i].TimestampUs - chunks[i-1].TimestampUs
				if d > 0 {
					sumDeltaUs += float64(d)
					n++
				}
			}
			if n > 0 {
				itlMeanByRequest[id] = (sumDeltaUs / float64(n)) / 1000.0 // µs → ms
			}
		}
	}

	type acc struct {
		good      int
		ttftMet   int
		ttftCount int
		itlMet    int
		itlCount  int
		e2eMet    int
		e2eCount  int
		injected  int64
		// per-class latency accumulators (in ms)
		ttftMs []float64
		itlMs  []float64
		e2eMs  []float64
	}
	accs := make(map[string]*acc, len(targets))
	for cls := range targets {
		accs[cls] = &acc{}
	}

	classOf := func(rec workload.TraceRecord) (string, bool) {
		c := rec.SLOClass
		if c == "" {
			c = "default"
		}
		if _, ok := targets[c]; !ok {
			return c, false
		}
		return c, true
	}

	for _, rec := range records {
		if rec.Status != "ok" && rec.Status != "error" && rec.Status != "timeout" {
			continue
		}
		cls, gated := classOf(rec)
		if !gated {
			continue
		}
		a := accs[cls]
		a.injected++

		// Failed/timeout records count in denominator only.
		if rec.Status != "ok" {
			continue
		}
		ttftUs := rec.FirstChunkTimeUs - rec.SendTimeUs
		e2eUs := rec.LastChunkTimeUs - rec.SendTimeUs
		if ttftUs <= 0 || e2eUs <= 0 || e2eUs < ttftUs {
			continue
		}
		ttftMs := float64(ttftUs) / 1000.0
		e2eMs := float64(e2eUs) / 1000.0
		a.ttftMs = append(a.ttftMs, ttftMs)
		a.e2eMs = append(a.e2eMs, e2eMs)

		t := targets[cls]
		good := true
		if t.TTFTMs > 0 {
			a.ttftCount++
			if ttftMs <= t.TTFTMs {
				a.ttftMet++
			} else {
				good = false
			}
		}
		if t.E2EMs > 0 {
			a.e2eCount++
			if e2eMs <= t.E2EMs {
				a.e2eMet++
			} else {
				good = false
			}
		}
		if t.ITLMs > 0 {
			a.itlCount++
			if itlMean, ok := itlMeanByRequest[rec.RequestID]; ok {
				a.itlMs = append(a.itlMs, itlMean)
				if itlMean <= t.ITLMs {
					a.itlMet++
				} else {
					good = false
				}
			} else {
				good = false
			}
		} else if itlMean, ok := itlMeanByRequest[rec.RequestID]; ok {
			a.itlMs = append(a.itlMs, itlMean)
		}
		if good {
			a.good++
		}
	}

	classKeys := make([]string, 0, len(targets))
	for k := range targets {
		classKeys = append(classKeys, k)
	}
	sort.Strings(classKeys)

	per := make(map[string]map[string]any, len(targets))
	var totalGood int
	var totalInjected int64
	for _, cls := range classKeys {
		a := accs[cls]
		t := targets[cls]
		entry := map[string]any{
			"slo_attainment": safeRatio(int64(a.good), a.injected),
			"count":          a.injected,
		}
		if runtimeSec > 0 {
			entry["goodput_rps"] = float64(a.good) / runtimeSec
		} else {
			entry["goodput_rps"] = 0.0
		}
		byDim := map[string]float64{}
		if t.TTFTMs > 0 {
			byDim["ttft"] = safeFraction(a.ttftMet, a.ttftCount)
		}
		if t.ITLMs > 0 {
			byDim["itl"] = safeFraction(a.itlMet, a.itlCount)
		}
		if t.E2EMs > 0 {
			byDim["e2e"] = safeFraction(a.e2eMet, a.e2eCount)
		}
		if len(byDim) > 0 {
			entry["slo_attainment_by_dim"] = byDim
		}
		// Per-class latency stats (ms) from the trace records.
		if len(a.ttftMs) > 0 {
			sort.Float64s(a.ttftMs)
			entry["ttft_p99_ms"] = percentile(a.ttftMs, 99)
		}
		if len(a.itlMs) > 0 {
			sort.Float64s(a.itlMs)
			var sum float64
			for _, v := range a.itlMs {
				sum += v
			}
			entry["itl_mean_ms"] = sum / float64(len(a.itlMs))
		}
		if len(a.e2eMs) > 0 {
			sort.Float64s(a.e2eMs)
			entry["e2e_p99_ms"] = percentile(a.e2eMs, 99)
		}
		per[cls] = entry
		totalGood += a.good
		totalInjected += a.injected
	}

	output.SLOAttainment = safeRatio(int64(totalGood), totalInjected)
	output.PerClass = per
	if runtimeSec > 0 {
		output.GoodputRPS = float64(totalGood) / runtimeSec
	}
}

// percentile returns the p-th percentile of an already-sorted []float64
// using linear interpolation. Returns 0 when data is empty.
func percentile(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return sorted[0]
	}
	rank := p / 100.0 * float64(n-1)
	lo := int(rank)
	hi := lo + 1
	if hi >= n {
		return sorted[n-1]
	}
	frac := rank - float64(lo)
	return sorted[lo] + (sorted[hi]-sorted[lo])*frac
}

// safeFraction is the int variant of safeRatio.
func safeFraction(num, denom int) float64 {
	if denom == 0 {
		return 0
	}
	return float64(num) / float64(denom)
}

// resolveGoodputCLIFlags parses the three CLI string flags into duration maps.
// Returns nil maps when the flag is empty. On parse error it returns the
// per-flag error so callers can logrus.Fatalf with the right flag name.
func resolveGoodputCLIFlags(ttftFlag, itlFlag, e2eFlag string) (
	cliTTFT, cliITL, cliE2E map[string]time.Duration, err error,
) {
	cliTTFT, err = parseSLODurationFlag(ttftFlag)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("--slo-ttft: %w", err)
	}
	cliITL, err = parseSLODurationFlag(itlFlag)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("--slo-itl: %w", err)
	}
	cliE2E, err = parseSLODurationFlag(e2eFlag)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("--slo-e2e: %w", err)
	}
	return cliTTFT, cliITL, cliE2E, nil
}
