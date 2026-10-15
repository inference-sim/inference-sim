package workload

import "github.com/inference-sim/inference-sim/sim"

// This file ports aiperf 0.12.0's ConversationReconstructor (dataset/loader/weka_synth_buf.py
// init_turn_0, advance_turn, truncate_synth_buf_at_block) at the granularity the replay
// needs: which recorded hash blocks and synthetic tail tokens each role-tagged message of a
// turn holds. aiperf sends every turn's FULL message list (DELTAS_WITH_RESPONSES with
// reset_context), so the wire prompt of turn k is these messages rendered through the chat
// template, with the tree's cache-bust marker at the start of the first user message.
// Rendering the template is what bounds cross-stream KV sharing the way the real wire does:
// two conversations that share hash blocks share tokens only up to their first differing
// message boundary.

// agxRun is a stretch of a message's content, as token offsets [a, b): of the concatenated
// 64-token blocks of turn's recorded hash ids, or of turn's synthetic tail. Offsets are in
// tokens so a cut lands exactly where aiperf's token-level slice does.
type agxRun struct {
	turn int32
	a, b int32
	tail bool
}

func (r agxRun) tokens() int32 { return r.b - r.a }

// agxSeg is a RoleSegment: role, block accounting (block_start, block_count) and content.
type agxSeg struct {
	role       byte // 's' system, 'u' user, 'a' assistant
	blockStart int32
	blockCount int32
	runs       []agxRun
}

func (s *agxSeg) tokens() int32 {
	n := int32(0)
	for _, r := range s.runs {
		n += r.tokens()
	}
	return n
}

// keepTokens truncates the segment's content to its first n tokens.
func (s *agxSeg) keepTokens(n int32) {
	var out []agxRun
	for _, r := range s.runs {
		if n <= 0 {
			break
		}
		l := r.tokens()
		if l <= n {
			out = append(out, r)
			n -= l
			continue
		}
		r.b = r.a + n
		out = append(out, r)
		n = 0
	}
	s.runs = out
}

// agxRecon is one stream's reconstructor state.
type agxRecon struct {
	segs []agxSeg
	turn int32 // last turn applied; -1 before turn 0
}

func newAgxRecon() *agxRecon { return &agxRecon{turn: -1} }

// region is the content of turn k's new region: hash ids [lcp, covered) of k's list, then
// its synth tail.
func agxRegion(k int32, lcp, covered, tail int32) []agxRun {
	var rs []agxRun
	if covered > lcp {
		rs = append(rs, agxRun{turn: k, a: lcp * agxBlockSize, b: covered * agxBlockSize})
	}
	if tail > 0 {
		rs = append(rs, agxRun{turn: k, a: 0, b: tail, tail: true})
	}
	return rs
}

// splitRuns splits runs at token offset n.
func splitRuns(rs []agxRun, n int32) (head, rest []agxRun) {
	for _, r := range rs {
		l := r.tokens()
		switch {
		case n <= 0:
			rest = append(rest, r)
		case l <= n:
			head = append(head, r)
			n -= l
		default:
			h, t := r, r
			h.b, t.a = r.a+n, r.a+n
			head, rest = append(head, h), append(rest, t)
			n = 0
		}
	}
	return head, rest
}

func agxGeometry(tn *agxTurn) (covered, tail int32) {
	full := tn.in / agxBlockSize
	covered = min(tn.nHash, full)
	tail = tn.in%agxBlockSize + max(0, (full-tn.nHash)*agxBlockSize)
	return covered, tail
}

// advance applies turn k (k == r.turn+1) of conv.
func (r *agxRecon) advance(conv *agxConv, k int32) {
	tn := &conv.turns[k]
	covered, tail := agxGeometry(tn)
	if k == 0 {
		// init_turn_0 with no tool/system prefix (BuildAgentXTrace refuses one): one user
		// message holding the covered blocks and the synth tail.
		r.segs = []agxSeg{{role: 'u', blockStart: 0, blockCount: covered, runs: agxRegion(0, 0, covered, tail)}}
		r.turn = 0
		return
	}
	lcp := tn.lcp
	r.truncate(lcp)
	newBlocks := max(0, covered-lcp)
	region := agxRegion(k, lcp, covered, tail)
	prevOut := conv.turns[k-1].rawOut
	asst := int32(0)
	if prevOut > 0 {
		asst = (prevOut + agxBlockSize - 1) / agxBlockSize
	}
	hasUser := false
	for _, s := range r.segs {
		if s.role == 'u' {
			hasUser = true
		}
	}
	if !hasUser {
		asst = 0 // context-loss rule
	}
	asst = min(asst, newBlocks)
	if tn.asstCap >= 0 {
		asst = min(asst, tn.asstCap)
	}
	if tail == 0 && asst == newBlocks && asst > 0 {
		asst-- // every turn ends with a user message
	}
	head, rest := splitRuns(region, asst*agxBlockSize)
	if asst > 0 {
		r.segs = append(r.segs, agxSeg{role: 'a', blockStart: lcp, blockCount: asst, runs: head})
	}
	if len(rest) > 0 {
		r.segs = append(r.segs, agxSeg{role: 'u', blockStart: lcp + asst, blockCount: newBlocks - asst, runs: rest})
	}
	r.turn = k
}

// truncate is truncate_synth_buf_at_block.
func (r *agxRecon) truncate(target int32) {
	if target <= 0 {
		r.segs = r.segs[:0]
		return
	}
	cursor := int32(0)
	for i := range r.segs {
		s := &r.segs[i]
		if cursor+s.blockCount < target {
			cursor += s.blockCount
			continue
		}
		if cursor+s.blockCount == target {
			if over := s.tokens() - s.blockCount*agxBlockSize; over > 0 {
				s.keepTokens(s.tokens() - over)
			}
			r.segs = r.segs[:i+1]
			return
		}
		if cursor == target {
			r.segs = r.segs[:i]
			return
		}
		kept := target - cursor
		s.blockCount = kept
		s.keepTokens(min(s.tokens(), kept*agxBlockSize))
		r.segs = r.segs[:i+1]
		return
	}
}

// The chat template the wire prompt is rendered with: DeepSeek-V3-family token counts
// (vLLM's tokenizer chat template for deepseek-v3 / v4): one BOS, a role header per message
// (user "<｜User｜>", assistant "<｜Assistant｜>" ... "<｜end▁of▁sentence｜>"), the generation
// prompt, and aiperf's "[rid:<12 hex>]\n\n" cache-bust marker, about 10 tokens, at the head
// of the first user message. Template tokens take reserved ids below agxContentBase.
const (
	agxTokBOS       = sim.TokenID(1)
	agxTokUser      = sim.TokenID(2)
	agxTokAssistant = sim.TokenID(3)
	agxTokEOS       = sim.TokenID(4)
	agxTokSystem    = sim.TokenID(5)
	agxMarkerTokens = 10
	agxContentBase  = 1 << 20
)

// agxTemplateOverhead is the template and marker tokens a message list adds to its content.
func agxTemplateOverhead(segs []agxSeg) int {
	n := 2 // BOS + generation prompt
	marker := false
	for _, s := range segs {
		switch s.role {
		case 'a':
			n += 2
		default:
			n++
			if s.role == 'u' && !marker {
				n += agxMarkerTokens
				marker = true
			}
		}
	}
	return n
}
