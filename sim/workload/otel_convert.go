package workload

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/spec/trace"
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
	// ThinkFromArrivalGaps states no think time, so a replay derives it from the arrival gaps,
	// which bundle the service time into the think time. It exists only to reproduce the output
	// of versions before the think-time change. By default each follow-up round's think time is the gap between
	// the previous span's end and this span's start, as the trace records it; a round whose
	// previous span has no usable end states none.
	ThinkFromArrivalGaps bool
	// MaxContext bounds every round's recorded prompt plus output; 0 = no ceiling.
	// MaxContextMode says what a session with a round over it becomes (max_context.go).
	MaxContext     int
	MaxContextMode MaxContextMode
}

// OTelSession is one converted session with what the converter learned about it beyond its
// records: when it started, and the system message that opened its first call.
type OTelSession struct {
	Records []TraceRecord
	// StartUs is the first call's start, in microseconds since the epoch.
	StartUs int64
	// SystemText is the text of the first call's system messages; empty when it has none.
	SystemText string
	// SystemShare is the share of the first call's message characters that the system messages
	// are, from which a prefix length in tokens is estimated (the spans state tokens only for
	// the whole call).
	SystemShare float64
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
	// The recorded model name (gen_ai.request.model) is intentionally NOT parsed:
	// it must not reach TraceRecord.Model (see EncodeSessionToTraceRecords), and encoding/json simply ignores the key.
	InputTokens  *int `json:"gen_ai.usage.input_tokens"`
	OutputTokens *int `json:"gen_ai.usage.output_tokens"`
	// Presence of this key (even empty) marks an LLM chat span when name doesn't.
	InputMessages *string `json:"gen_ai.input.messages"`
}

// otelMessage is one input message: a role and its text, as a string or as parts.
type otelMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
	Parts   []struct {
		Content string `json:"content"`
		Text    string `json:"text"`
	} `json:"parts"`
}

func (m otelMessage) text() string {
	var s string
	if len(m.Content) > 0 && json.Unmarshal(m.Content, &s) == nil {
		return s
	}
	for _, p := range m.Parts {
		s += p.Content + p.Text
	}
	return s
}

// systemOf returns the text of the system messages in a call's input messages and the share of
// the messages' characters they are. Unparseable or empty messages give no system text.
func systemOf(messages *string) (string, float64) {
	if messages == nil || *messages == "" {
		return "", 0
	}
	var ms []otelMessage
	if json.Unmarshal([]byte(*messages), &ms) != nil {
		return "", 0
	}
	var sys string
	total := 0
	for _, m := range ms {
		t := m.text()
		total += len(t)
		if m.Role == "system" {
			sys += t
		}
	}
	if sys == "" || total == 0 {
		return "", 0
	}
	return sys, float64(len(sys)) / float64(total)
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
	sess, err := ConvertOTelSession(raw, opts)
	if err != nil || sess == nil {
		return nil, err
	}
	return sess.Records, nil
}

// ConvertOTelSession is ConvertOTelTrace returning the session with its start and its system
// message. It returns (nil, nil) when the session has fewer than opts.MinRounds usable calls.
func ConvertOTelSession(raw []byte, opts OTelConvertOptions) (*OTelSession, error) {
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
		startUs, endUs int64
		in, out        int
		isError        bool
		system         string
		systemShare    float64
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
		var endUs int64
		if sp.EndTime != "" {
			if e, err := parseOTelTime(sp.EndTime); err == nil && e > startUs {
				endUs = e
			}
		}
		sys, share := systemOf(sp.Attributes.InputMessages)
		spans = append(spans, parsedSpan{
			startUs: startUs,
			endUs:   endUs,
			in:      *sp.Attributes.InputTokens,
			out:     *sp.Attributes.OutputTokens,
			isError: sp.Status.Code == 2,
			system:  sys, systemShare: share,
		})
	}

	if len(spans) < minRounds {
		return nil, nil
	}

	// Order by start time (stable): overlapping/parallel spans serialize by start.
	sort.SliceStable(spans, func(i, j int) bool { return spans[i].startUs < spans[j].startUs })

	// Build the source-agnostic normalized session; the shared encoder
	// (EncodeSessionToTraceRecords) computes per-round input deltas and leaves
	// TraceRecord.Model empty (a recorded cross-model name would report every
	// request in the per-model metrics under a model the run never simulated; see
	// the encoder's doc comment and #1477). ThinkUs is left nil (not recorded, #1608): OTel has no
	// reliable response-complete time (end_time is often a placeholder), so per-round
	// think is derived from arrival gaps at replay rather than recorded here.
	t0 := spans[0].startUs
	rounds := make([]NormalizedRound, 0, len(spans))
	for i, sp := range spans {
		// Relative arrival time; cap the inter-round gap when requested.
		arrival := sp.startUs - t0
		if i > 0 && opts.MaxThinkTimeUs > 0 {
			prevArrival := rounds[i-1].ArrivalUs
			if arrival-prevArrival > opts.MaxThinkTimeUs {
				arrival = prevArrival + opts.MaxThinkTimeUs
			}
		}
		status := trace.StatusOK
		if sp.isError {
			status = trace.StatusError
		}
		round := NormalizedRound{
			InputTokensAbs: sp.in,
			OutputTokens:   sp.out,
			ArrivalUs:      arrival,
			Status:         status,
		}
		// The client thinks from the response it received to the next call it sends: the
		// previous span's end to this span's start (#2130).
		if i > 0 && !opts.ThinkFromArrivalGaps && spans[i-1].endUs > 0 && spans[i-1].endUs <= sp.startUs {
			think := sp.startUs - spans[i-1].endUs
			if opts.MaxThinkTimeUs > 0 && think > opts.MaxThinkTimeUs {
				think = opts.MaxThinkTimeUs
			}
			round.ThinkUs = &think
		}
		rounds = append(rounds, round)
	}
	if rounds = boundContext(rounds, opts.MaxContext, opts.MaxContextMode); len(rounds) < minRounds {
		return nil, nil
	}
	return &OTelSession{Records: EncodeSessionToTraceRecords(sessionID, rounds), StartUs: t0,
		SystemText: spans[0].system, SystemShare: spans[0].systemShare}, nil
}
