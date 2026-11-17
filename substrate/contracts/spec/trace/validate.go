package trace

import (
	"fmt"
	"regexp"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// Validate checks the header on its own.
func (h *Header) Validate() *validate.Problems {
	p := &validate.Problems{}
	if h.Version != Version {
		p.Field("trace_version", "must be %d, got %d: %s", Version, h.Version, versionMessage(h.Version))
	}
	switch {
	case h.TimeUnit == "":
		p.Field("time_unit", "required: names the unit of the data file's timestamp columns")
	case !h.TimeUnit.Valid():
		p.Field("time_unit", "%q is not one of %v", h.TimeUnit, AllTimeUnits())
	}
	switch {
	case h.Mode == "":
		p.Field("mode", "required: one of %v", AllModes())
	case h.Mode == "real":
		p.Field("mode", "\"real\" was written by blis observe, which is removed; a captured trace is converted with blis convert and has mode %q", ModeConverted)
	case !h.Mode.Valid():
		p.Field("mode", "%q is not one of %v", h.Mode, AllModes())
	}
	if g := h.SessionContextGrowth; g != "" && g != ContextAccumulate {
		p.Field("session_context_growth", "%q is not %q; omit it for rounds drawn independently", g, ContextAccumulate)
	}
	for _, class := range sortedKeys(h.GoodputSLOTargets) {
		if class == "" {
			p.Field("goodput_slo_targets", "an SLO class key must not be empty")
			continue
		}
		p.Merge("goodput_slo_targets."+class, h.GoodputSLOTargets[class].Validate())
	}
	if h.KVOffload != nil {
		p.Merge("kv_offload", h.KVOffload.Validate())
	}
	switch {
	case h.MaxNodesSpanned < 0:
		p.Field("max_nodes_spanned", "must not be negative")
	case h.MaxNodesSpanned == 1:
		p.Field("max_nodes_spanned", "is 1: an instance on one node spans none; omit the key")
	}
	h.validatePhases(p)
	if r := h.Router; r != nil && len(r.EPPFlags) == 0 {
		p.Field("router.epp_flags", "required: the router's command-line flags")
	}
	// A converted trace records no BLIS deployment, so it carries none of the deployment
	// inputs a replay would otherwise keep authoritative (inference-sim #2067).
	if h.Mode == ModeConverted {
		for key, set := range map[string]bool{"kv_offload": h.KVOffload != nil,
			"max_nodes_spanned": h.MaxNodesSpanned != 0, "router": h.Router != nil} {
			if set {
				p.Field(key, "a converted trace records no BLIS deployment; %s belongs to a trace blis run or blis replay wrote", key)
			}
		}
	}
	return p
}

// validatePhases checks the phase list on its own: each name is a lowercase slug, used once;
// the start is a known one; the numbers are not negative.
func (h *Header) validatePhases(p *validate.Problems) {
	seen := map[string]bool{}
	for i, ph := range h.Phases {
		at := fmt.Sprintf("phases[%d]", i)
		switch {
		case ph.Name == "":
			p.Field(at+".name", "required: the label the phase's records carry")
		case !phaseSlug.MatchString(ph.Name):
			p.Field(at+".name", "%q is not a lowercase slug (letters, digits, - and _)", ph.Name)
		case seen[ph.Name]:
			p.Field(at+".name", "%q names two phases", ph.Name)
		}
		seen[ph.Name] = true
		switch ph.Start {
		case "", StartAfterDrain, StartAt:
		default:
			p.Field(at+".start", "%q is not %q or %q", ph.Start, StartAfterDrain, StartAt)
		}
		if ph.OffsetUs < 0 {
			p.Field(at+".offset_us", "must not be negative, got %d", ph.OffsetUs)
		}
		if ph.Start == StartAt && i > 0 && ph.OffsetUs < h.Phases[i-1].OffsetUs && h.Phases[i-1].Start == StartAt {
			p.Field(at+".offset_us", "is %d, before the previous phase's start at %d: phases are in order", ph.OffsetUs, h.Phases[i-1].OffsetUs)
		}
		if ph.Concurrency < 0 {
			p.Field(at+".concurrency", "must not be negative, got %d", ph.Concurrency)
		}
	}
}

var phaseSlug = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)

// Validate checks one SLO class's targets.
func (t SLODimTargets) Validate() *validate.Problems {
	p := &validate.Problems{}
	for _, f := range []struct {
		name string
		v    float64
	}{{"ttft_ms", t.TTFTMs}, {"itl_ms", t.ITLMs}, {"e2e_ms", t.E2EMs}} {
		if p.FiniteField(f.name, f.v) && f.v < 0 {
			p.Field(f.name, "must not be negative")
		}
	}
	if t.TTFTMs == 0 && t.ITLMs == 0 && t.E2EMs == 0 {
		p.Errorf("sets no target on any of ttft_ms, itl_ms, e2e_ms; omit the class instead of declaring it empty")
	}
	return p
}

// Validate checks an offload configuration's numbers.
func (o *KVOffload) Validate() *validate.Problems {
	p := &validate.Problems{}
	for _, f := range []struct {
		name string
		v    int64
	}{{"cpu_bytes_to_use", o.CPUBytesToUse}, {"per_block_bytes", o.PerBlockBytes}, {"block_size", o.BlockSize},
		{"blocks_per_chunk", o.BlocksPerChunk}, {"tokens_per_hash", o.TokensPerHash}} {
		if f.v < 0 {
			p.Field(f.name, "must not be negative")
		}
	}
	for i, t := range o.Tiers {
		at := fmt.Sprintf("secondary_tiers[%d]", i)
		if t.Type == "" {
			p.Field(at+".type", "required")
		}
		for _, f := range []struct {
			name string
			v    float64
		}{{"read_bandwidth", t.ReadBandwidth}, {"write_bandwidth", t.WriteBandwidth}, {"base_latency", t.BaseLatency},
			{"single_transfer_fraction", t.SingleTransferFraction}, {"latency_jitter_stddev", t.LatencyJitterStddev}} {
			if p.FiniteField(at+"."+f.name, f.v) && f.v < 0 {
				p.Field(at+"."+f.name, "must not be negative")
			}
		}
		for _, f := range []struct {
			name string
			v    int64
		}{{"n_read_threads", t.NReadThreads}, {"n_write_threads", t.NWriteThreads},
			{"saturation_queue_depth", t.SaturationQueueDepth}, {"capacity_bytes", t.CapacityBytes}} {
			if f.v < 0 {
				p.Field(at+"."+f.name, "must not be negative")
			}
		}
	}
	return p
}

// Validate checks one record, as inference-sim's trace loader did when it parsed a row.
func (r *Record) Validate() *validate.Problems {
	p := &validate.Problems{}
	for _, f := range []struct {
		name string
		v    int
	}{{"request_id", r.RequestID}, {"round_index", r.RoundIndex}, {"prefix_length", r.PrefixLength},
		{"input_tokens", r.InputTokens}, {"output_tokens", r.OutputTokens}, {"text_tokens", r.TextTokens},
		{"image_tokens", r.ImageTokens}, {"audio_tokens", r.AudioTokens}, {"video_tokens", r.VideoTokens}} {
		if f.v < 0 {
			p.Field(f.name, "must not be negative, got %d", f.v)
		}
	}
	for _, f := range []struct {
		name string
		v    int64
	}{{"deadline_us", r.DeadlineUs}, {"arrival_time_us", r.ArrivalTimeUs}, {"first_chunk_time_us", r.FirstChunkTimeUs},
		{"last_chunk_time_us", r.LastChunkTimeUs}, {"slo_target_us", r.SLOTargetUs}} {
		if f.v < 0 {
			p.Field(f.name, "must not be negative, got %d", f.v)
		}
	}
	if p.FiniteField("reason_ratio", r.ReasonRatio) && (r.ReasonRatio < 0 || r.ReasonRatio > 1) {
		p.Field("reason_ratio", "must lie in [0, 1], got %v", r.ReasonRatio)
	}
	if r.DeadlineUs > 0 && r.ArrivalTimeUs > 0 && r.DeadlineUs < r.ArrivalTimeUs {
		p.Field("deadline_us", "is %d, before arrival_time_us %d", r.DeadlineUs, r.ArrivalTimeUs)
	}
	if r.LastChunkTimeUs != 0 && r.LastChunkTimeUs < r.FirstChunkTimeUs {
		p.Field("last_chunk_time_us", "is %d, before first_chunk_time_us %d", r.LastChunkTimeUs, r.FirstChunkTimeUs)
	}
	if !r.Status.Valid() {
		p.Field("status", "%q is not one of %v", r.Status, AllStatuses())
	}
	if r.ThinkTimeUs != nil && *r.ThinkTimeUs < 0 {
		p.Field("think_time_us", "must not be negative, got %d", *r.ThinkTimeUs)
	}
	if r.InputTokensReset != nil && *r.InputTokensReset < 0 {
		p.Field("input_tokens_reset", "must not be negative, got %d", *r.InputTokensReset)
	}
	return p
}

// Trace is a header and its records.
type Trace struct {
	Header  Header
	Records []Record
}

// Validate checks the header, every record, and what holds between records: request
// identifiers are unique, and within a session the rounds are numbered once each.
func (t *Trace) Validate() *validate.Problems {
	p := &validate.Problems{}
	p.Merge("header", t.Header.Validate())
	ids := map[int]bool{}
	phased := map[string]int{}
	known := map[string]bool{}
	for _, ph := range t.Header.Phases {
		known[ph.Name] = true
	}
	rounds := map[string]map[int]bool{}
	for i, r := range t.Records {
		at := fmt.Sprintf("records[%d]", i)
		p.Merge(at, r.Validate())
		if ids[r.RequestID] {
			p.Field(at+".request_id", "%d appears twice", r.RequestID)
		}
		ids[r.RequestID] = true
		switch {
		case len(known) == 0 && r.Phase != "":
			p.Field(at+".phase", "%q: the header lists no phases", r.Phase)
		case len(known) > 0 && r.Phase == "":
			p.Field(at+".phase", "required: the header lists phases, so every request names one")
		case len(known) > 0 && !known[r.Phase]:
			p.Field(at+".phase", "%q is not a phase of the header", r.Phase)
		}
		phased[r.Phase]++
		if r.SessionID == "" {
			continue
		}
		if rounds[r.SessionID] == nil {
			rounds[r.SessionID] = map[int]bool{}
		}
		if rounds[r.SessionID][r.RoundIndex] {
			p.Field(at+".round_index", "session %q has round %d twice", r.SessionID, r.RoundIndex)
		}
		rounds[r.SessionID][r.RoundIndex] = true
	}
	for i, ph := range t.Header.Phases {
		if phased[ph.Name] == 0 {
			p.Field(fmt.Sprintf("header.phases[%d]", i), "phase %q names no request: a phase without requests is refused", ph.Name)
		}
	}
	return p
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j] < out[j-1]; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}
