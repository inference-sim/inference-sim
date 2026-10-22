package workload

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/sirupsen/logrus"
)

// WekaConvertOptions configures Weka-session → TraceRecord conversion.
type WekaConvertOptions struct {
	// ContextGrowth is not read by ConvertWekaSession itself; it is read by the
	// caller (the `blis convert weka` command) to choose the exported trace
	// header's session_context_growth value: "accumulate" (default) or
	// "independent". See TraceHeader.SessionContextGrowth in tracev2.go.
	ContextGrowth string
	// MaxThinkTimeUs caps each recomputed inter-turn think gap; 0 = no cap.
	// Weka's gaps are genuine away-from-keyboard times, so uncapped (0) is
	// fidelity-preserving (unlike OTel's 15s default); a bounded-horizon
	// capacity run may set a cap so a single multi-minute away-gap does not
	// idle a pooled session for that long of sim time.
	MaxThinkTimeUs int64
	// MinRounds skips sessions with fewer usable main-agent turns (default 1).
	MinRounds int
	// DropSubagents converts a session containing type:"subagent" groups by dropping
	// them (their wall-clock folds into the next main turn's think gap). Unset, such a
	// session is refused (ErrWekaSubagentFanOut): the dropped child requests would put
	// no load, overlap or KV state on the engine, so the replay would under-load it.
	DropSubagents bool
}

// ErrWekaSubagentFanOut is returned for a session with sub-agent groups when
// DropSubagents is unset: their nested, concurrent requests are not modelled yet
// (#1477), and dropping them silently would under-represent the offered load.
var ErrWekaSubagentFanOut = errors.New("weka session contains sub-agent fan-out, which is not modelled (#1477)")

// CountWekaSubagents returns the number of type:"subagent" groups in one Weka session
// and the nested requests they carry.
func CountWekaSubagents(raw []byte) (groups, nested int, err error) {
	var s struct {
		Requests []struct {
			Type     string            `json:"type"`
			Requests []json.RawMessage `json:"requests"`
		} `json:"requests"`
	}
	if err := json.Unmarshal(raw, &s); err != nil {
		return 0, 0, fmt.Errorf("parsing weka session: %w", err)
	}
	for _, r := range s.Requests {
		if r.Type == "subagent" {
			groups++
			nested += len(r.Requests)
		}
	}
	return groups, nested, nil
}

// wekaSession is one JSONL line (one proxy session, a Weka v7 trace). Unknown fields
// are ignored (standard encoding/json behavior — do NOT use DisallowUnknownFields; real
// traces carry fields we don't model: models, hash_id_scope, totals). ConvertWekaSession
// reads only ID and the requests' timing and token counts; the AgentX replay
// (agentx_tree.go) also reads the hash-id, model and subagent fields.
type wekaSession struct {
	ID           string        `json:"id"`
	BlockSize    int           `json:"block_size"`
	ToolTokens   int           `json:"tool_tokens"`
	SystemTokens int           `json:"system_tokens"`
	Requests     []wekaRequest `json:"requests"`
}

// wekaRequest is one entry in a session's requests[]: a main-agent request
// (type "n" non-streaming / "s" streaming) or a sub-agent group (type
// "subagent"), which this converter refuses, or drops when asked to (#1477).
type wekaRequest struct {
	Type string  `json:"type"`     // "n", "s", or "subagent"
	T    float64 `json:"t"`        // seconds since trace start (arrival); a subagent's spawn time
	In   *int    `json:"in"`       // effective o200k input tokens; read directly (never derived from hash_ids)
	Out  *int    `json:"out"`      // output tokens (Anthropic-reported)
	API  float64 `json:"api_time"` // real end-to-end server time (sec); used to recompute pure think
	// Model is the recorded model name (claude-*). ConvertWekaSession never copies it to
	// a TraceRecord: at replay it would become Request.Model, which keys the per-model
	// metrics, and would match no simulated model. The AgentX chain detection
	// uses it only to tell agents apart (same-model rule).
	Model string `json:"model"`
	// HashIDs are the request's 64-token prompt blocks in trace-local ids.
	HashIDs []int64 `json:"hash_ids"`
	// Subagent-group fields (type "subagent" only).
	AgentID      string        `json:"agent_id"`
	DurationMs   *float64      `json:"duration_ms"`
	ToolTokens   int           `json:"tool_tokens"`
	SystemTokens int           `json:"system_tokens"`
	Requests     []wekaRequest `json:"requests"`
}

// ConvertWekaSession converts one Weka JSONL session into ordered TraceRecords
// with per-round input-token deltas. It filters requests[] to the linear
// main-agent stream (type:"subagent" groups are refused, or dropped under
// opts.DropSubagents), keeps main turns in
// stream order (a linear agent conversation is inherently sequential, unlike
// OTel's parallel spans — context accumulation depends on call order, not
// timestamp order), and recomputes each round's pure client think time as
// max(0, t_i − t_{i-1} − api_time_{i-1}) between consecutive main turns.
//
// RoundIndex is 0..N over main turns. RequestID is left 0 (the caller assigns
// global ids). Returns (nil, nil) when the session has fewer than
// opts.MinRounds usable main turns (e.g. an all-subagent session).
//
// Recomputing think (rather than trusting the recorded `think_time`) keeps the
// main chain clean when sub-agent entries are removed: a recorded `think_time`
// may reference a chronologically-preceding sub-agent inner request. The
// wall-clock a skipped sub-agent group ran is absorbed into the following main
// turn's think gap (the main agent really was blocked on the Task tool); PR-E
// models the fan-out explicitly.
func ConvertWekaSession(raw []byte, opts WekaConvertOptions) ([]TraceRecord, error) {
	minRounds := opts.MinRounds
	if minRounds < 1 {
		minRounds = 1
	}

	var s wekaSession
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, fmt.Errorf("parsing weka session: %w", err)
	}
	if s.ID == "" {
		return nil, fmt.Errorf("no session id (\"id\") in weka session")
	}
	if !opts.DropSubagents {
		if groups, nested, err := CountWekaSubagents(raw); err != nil {
			return nil, err
		} else if groups > 0 {
			return nil, fmt.Errorf("%w: session %q has %d sub-agent group(s) carrying %d nested request(s); "+
				"pass --drop-subagents to convert its main-agent stream only, under-loading the replay",
				ErrWekaSubagentFanOut, s.ID, groups, nested)
		}
	}

	// Filter to usable main-agent turns in stream order. `in`/`out` are read
	// directly from the record (never derived from len(hash_ids)×64, which would
	// over-count up to 63 tokens/request on the v5 dataset encoding). A main turn
	// missing either count is dropped (defensive — real data always carries them).
	type mainTurn struct {
		t, api float64
		in     int
		out    int
	}
	var turns []mainTurn
	for i := range s.Requests {
		r := &s.Requests[i]
		if r.Type == "subagent" {
			continue // opts.DropSubagents: refused above otherwise (#1477)
		}
		if r.In == nil || r.Out == nil {
			// No ground-truth token counts → unusable. Real Weka main turns always
			// carry in/out (0 drops observed across the 051926 dataset), so this is
			// defense-in-depth; log it (not a silent continue, R1) so a future
			// data-shape change that starts dropping turns is debuggable.
			logrus.Debugf("convert weka: session %q request %d dropped — main turn missing in/out", s.ID, i)
			continue
		}
		turns = append(turns, mainTurn{t: r.T, api: r.API, in: *r.In, out: *r.Out})
	}

	if len(turns) < minRounds {
		return nil, nil
	}

	// Build the source-agnostic normalized session; the shared encoder
	// (EncodeSessionToTraceRecords) computes per-round input deltas and leaves
	// TraceRecord.Model empty (routing safety). ThinkUs carries the recomputed
	// pure client think (#1478), preferred over arrival-gap think at replay.
	rounds := make([]NormalizedRound, 0, len(turns))
	for i, tn := range turns {
		// Round 0 has no predecessor → think is not recorded (nil, #1608). Rounds
		// 1..N carry the recomputed pure client think — including a recorded &0 when
		// the turn overlapped the previous response, which is now distinguishable
		// from "not recorded" (no longer degrades an all-overlap session to gaps).
		var thinkUs *int64
		if i > 0 {
			prev := turns[i-1]
			// Pure client think: wall-clock from the previous turn's RESPONSE
			// (its end = t_{i-1} + api_time_{i-1}) to this turn's arrival.
			gapUs := int64(math.Round((tn.t - prev.t - prev.api) * 1e6))
			if gapUs < 0 {
				gapUs = 0 // clamp (INV-3): overlapping turns — round i started before round i-1's response elapsed
			}
			if opts.MaxThinkTimeUs > 0 && gapUs > opts.MaxThinkTimeUs {
				gapUs = opts.MaxThinkTimeUs
			}
			thinkUs = &gapUs
		}
		rounds = append(rounds, NormalizedRound{
			InputTokensAbs: tn.in,
			OutputTokens:   tn.out,
			// Absolute per-session arrival (Weka `t` is seconds since trace
			// start, and each JSONL line is one session, so `t` is already
			// per-session-relative). Round 0 = the first main turn's `t` (may be
			// >0 when a skipped sub-agent group preceded it). Follow-up spacing at
			// closed-loop replay comes from ThinkUs, not this arrival.
			ArrivalUs: int64(math.Round(tn.t * 1e6)),
			ThinkUs:   thinkUs,
			Status:    "ok", // Weka errors are filtered at source
		})
	}
	return EncodeSessionToTraceRecords(s.ID, rounds), nil
}
