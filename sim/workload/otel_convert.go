package workload

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// OTelConvertOptions configures OTel-trace → TraceRecord conversion.
type OTelConvertOptions struct {
	// ContextGrowth is not read by ConvertOTelTrace itself; it is read by the
	// caller (the `blis convert otel` command) to choose the exported trace
	// header's session_context_growth value: "accumulate" (default) or
	// "independent". See TraceHeader.SessionContextGrowth in tracev2.go.
	ContextGrowth  string
	MaxThinkTimeUs int64 // cap on per-round arrival gap; 0 = no cap
	IncludeErrors  bool  // keep spans with status.code == 2
	MinRounds      int   // skip sessions with fewer usable calls (default 1)
}

// otelSpan is the subset of an OTel span we read. Unknown fields are ignored
// (standard encoding/json behavior — do NOT use DisallowUnknownFields here;
// real traces carry many attributes we don't model).
type otelSpan struct {
	SpanID     string         `json:"span_id"`
	TraceID    string         `json:"trace_id"`
	SessionID  string         `json:"session_id"`
	Name       string         `json:"name"`
	StartTime  string         `json:"start_time"`
	EndTime    string         `json:"end_time"`
	Status     otelStatus     `json:"status"`
	Attributes otelAttributes `json:"attributes"`
}

type otelStatus struct {
	Code int `json:"code"` // 1 = OK, 2 = ERROR
}

type otelAttributes struct {
	// Model is parsed for completeness but intentionally NOT propagated to
	// TraceRecord.Model (that field is routing-significant — see the
	// TraceRecord construction in ConvertOTelTrace). Kept here only to document
	// the schema; drop it entirely if a linter flags it as unused.
	Model        string `json:"gen_ai.request.model"`
	InputTokens  *int   `json:"gen_ai.usage.input_tokens"`
	OutputTokens *int   `json:"gen_ai.usage.output_tokens"`
	// Presence of this key (even empty) marks an LLM chat span when name doesn't.
	InputMessages *string `json:"gen_ai.input.messages"`
}

type otelTrace struct {
	Spans []otelSpan `json:"spans"`
}

// parseOTelTime parses an ISO-8601 timestamp that may be tz-aware ("...+00:00")
// or tz-naive ("..."). Naive timestamps are interpreted as UTC. Returns
// microseconds since the Unix epoch.
func parseOTelTime(s string) (int64, error) {
	// Try RFC3339 (tz-aware) first.
	if t, err := time.Parse(time.RFC3339Nano, s); err == nil {
		return t.UnixMicro(), nil
	}
	// Fall back to tz-naive: append Z (UTC).
	if t, err := time.Parse("2006-01-02T15:04:05.999999", s); err == nil {
		return t.UTC().UnixMicro(), nil
	}
	return 0, fmt.Errorf("unparseable timestamp %q", s)
}

// isLLMSpan reports whether a span is an LLM chat call.
func isLLMSpan(sp *otelSpan) bool {
	if strings.HasPrefix(sp.Name, "chat ") {
		return true
	}
	return sp.Attributes.InputMessages != nil
}

// sessionIDFromSpans extracts the session identifier from already-parsed OTel
// spans. Prefers the top-level-per-span session_id, then trace_id. Errors if
// neither is present on any span.
func sessionIDFromSpans(spans []otelSpan) (string, error) {
	for i := range spans {
		if spans[i].SessionID != "" {
			return spans[i].SessionID, nil
		}
		if spans[i].TraceID != "" {
			return spans[i].TraceID, nil
		}
	}
	return "", fmt.Errorf("no session_id or trace_id found in trace")
}

// OTelSessionID extracts the session identifier from a raw OTel trace.
// Prefers the top-level-per-span session_id, then trace_id. Errors if neither
// is present on any span.
func OTelSessionID(raw []byte) (string, error) {
	var tr otelTrace
	if err := json.Unmarshal(raw, &tr); err != nil {
		return "", fmt.Errorf("parsing OTel trace: %w", err)
	}
	return sessionIDFromSpans(tr.Spans)
}

// ConvertOTelTrace converts one OTel trace (a single agent session) into ordered
// TraceRecords with per-round input-token deltas. RoundIndex is 0..N in
// start-time order. RequestID is left 0; the caller assigns global IDs. Returns
// (nil, nil) when the session has fewer than opts.MinRounds usable LLM calls.
func ConvertOTelTrace(raw []byte, opts OTelConvertOptions) ([]TraceRecord, error) {
	minRounds := opts.MinRounds
	if minRounds < 1 {
		minRounds = 1
	}
	var tr otelTrace
	if err := json.Unmarshal(raw, &tr); err != nil {
		return nil, fmt.Errorf("parsing OTel trace: %w", err)
	}

	sessionID, sidErr := sessionIDFromSpans(tr.Spans)
	if sidErr != nil {
		return nil, sidErr
	}

	// Filter to usable LLM spans. Note: the recorded model name
	// (sp.Attributes.Model) is deliberately NOT captured here — see the
	// TraceRecord construction below for why it must not reach the record.
	type parsedSpan struct {
		startUs int64
		in, out int
		isError bool
	}
	var spans []parsedSpan
	for i := range tr.Spans {
		sp := &tr.Spans[i]
		if !isLLMSpan(sp) {
			continue
		}
		if sp.Status.Code == 2 && !opts.IncludeErrors {
			continue
		}
		if sp.Attributes.InputTokens == nil || sp.Attributes.OutputTokens == nil {
			continue // no ground-truth token counts → unusable
		}
		startUs, err := parseOTelTime(sp.StartTime)
		if err != nil {
			return nil, fmt.Errorf("session %q span %q: %w", sessionID, sp.SpanID, err)
		}
		spans = append(spans, parsedSpan{
			startUs: startUs,
			in:      *sp.Attributes.InputTokens,
			out:     *sp.Attributes.OutputTokens,
			isError: sp.Status.Code == 2,
		})
	}

	if len(spans) < minRounds {
		return nil, nil
	}

	// Order by start time (stable): overlapping/parallel spans serialize by start.
	sort.SliceStable(spans, func(i, j int) bool { return spans[i].startUs < spans[j].startUs })

	t0 := spans[0].startUs
	recs := make([]TraceRecord, 0, len(spans))
	prevIn, prevOut := 0, 0
	for round, sp := range spans {
		// Relative arrival time; cap the inter-round gap when requested.
		arrival := sp.startUs - t0
		if round > 0 && opts.MaxThinkTimeUs > 0 {
			prevArrival := recs[round-1].ArrivalTimeUs
			if arrival-prevArrival > opts.MaxThinkTimeUs {
				arrival = prevArrival + opts.MaxThinkTimeUs
			}
		}

		// Per-round input delta. Round 0 = full first prompt.
		inputDelta := sp.in
		if round > 0 {
			inputDelta = sp.in - prevIn - prevOut
			if inputDelta < 0 {
				inputDelta = 0 // clamp: non-monotone context (rare; e.g. context trimming)
			}
		}

		// Model is deliberately left empty. TraceRecord.Model is NOT passive
		// metadata: at replay it flows to req.Model, which buildRouterState
		// (sim/cluster/cluster_event.go:76-78) uses to FILTER routable instances
		// — `if req.Model != "" && inst.Model != req.Model { continue }`. Every
		// replay instance is configured with the --model flag (e.g. qwen/…),
		// while OTel traces record a different model (gpt-4o, claude-*). Writing
		// the recorded name here would leave zero routable instances → every
		// request silently dropped at routing (routingRejections++), zero
		// completions, no error. Leaving it empty makes requests inherit --model,
		// which is the intended behavior (all calls simulate under one model per
		// #1477). The recorded source model name is pure provenance metadata and
		// is dropped during conversion.
		status := "ok"
		if sp.isError {
			status = "error"
		}
		recs = append(recs, TraceRecord{
			SessionID:     sessionID,
			RoundIndex:    round,
			InputTokens:   inputDelta,
			OutputTokens:  sp.out,
			ArrivalTimeUs: arrival,
			// Model: intentionally empty — see comment above.
			Status: status,
		})
		prevIn, prevOut = sp.in, sp.out
	}
	return recs, nil
}
