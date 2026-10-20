package workload

import (
	"hash/fnv"
	"os"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestAgentXMeasureTemplateAndSharing reports, over a corpus (AGENTX_CORPUS, with
// AGENTX_MEASURE=1), what rendering prompts through the chat template changes:
//   - the template and marker tokens per request, against the recorded content;
//   - how many traces the max-context filter keeps whose wire prompt would still overflow;
//   - the prefix reuse one tree instance offers, in 64-token blocks of the prompt as sent,
//     split into blocks the request's own stream sent before and blocks only OTHER streams
//     (parent <-> subagent <-> flat chains) sent, with and without the template: hash-id
//     identity alone (no template) credits cross-stream matches the real wire ends at the
//     first differing message boundary.
//
// Every request of every trace is rendered once, in recorded start order, against an
// unbounded cache of the blocks the tree's earlier requests sent.
func TestAgentXMeasureTemplateAndSharing(t *testing.T) {
	path := os.Getenv("AGENTX_CORPUS")
	if path == "" || os.Getenv("AGENTX_MEASURE") == "" {
		t.Skip("set AGENTX_CORPUS and AGENTX_MEASURE=1")
	}
	c, err := LoadAgentXCorpus(path, "", 0, true)
	if err != nil {
		t.Fatal(err)
	}
	type acc struct{ tokens, own, cross int64 }
	stats := map[string]map[bool]*acc{} // kind -> template -> acc
	for _, k := range []string{"parent", "subagent", "all"} {
		stats[k] = map[bool]*acc{false: {}, true: {}}
	}
	var overheadPct, msgs []float64
	var contentTok, overheadTok int64
	limits := []int{131072, 200000, 256000}
	overflowTraces := make([]int, len(limits))
	keptTraces := make([]int, len(limits))
	for _, tr := range c.Traces {
		type item struct {
			ts   float64
			conv int32
			turn int32
		}
		var items []item
		for ci := range tr.convs {
			for k := range tr.convs[ci].turns {
				items = append(items, item{tr.convs[ci].turns[k].tsMs, int32(ci), int32(k)})
			}
		}
		sort.SliceStable(items, func(a, b int) bool {
			if items[a].ts != items[b].ts {
				return items[a].ts < items[b].ts
			}
			if items[a].conv != items[b].conv {
				return items[a].conv < items[b].conv
			}
			return items[a].turn < items[b].turn
		})
		recons := make([]*agxRecon, len(tr.convs))
		for i := range recons {
			recons[i] = newAgxRecon()
		}
		// seen[template] = blocks the tree sent; own[template][conv] = blocks conv sent.
		seen := map[bool]map[uint64]bool{false: {}, true: {}}
		own := map[bool][]map[uint64]bool{false: make([]map[uint64]bool, len(tr.convs)), true: make([]map[uint64]bool, len(tr.convs))}
		for _, b := range []bool{false, true} {
			for i := range own[b] {
				own[b][i] = map[uint64]bool{}
			}
		}
		wirePeak := 0
		for _, it := range items {
			cv := &tr.convs[it.conv]
			r := recons[it.conv]
			for k := r.turn + 1; k <= it.turn; k++ {
				r.advance(cv, k)
			}
			tn := &cv.turns[it.turn]
			ov := agxTemplateOverhead(r.segs)
			contentTok += int64(tn.in)
			overheadTok += int64(ov)
			overheadPct = append(overheadPct, 100*float64(ov)/float64(max(tn.in, 1)))
			msgs = append(msgs, float64(len(r.segs)))
			wirePeak = max(wirePeak, int(tn.in)+ov+int(tn.maxTokens))
			kind := "parent"
			if cv.scope != 0 {
				kind = "subagent"
			}
			for _, tmpl := range []bool{false, true} {
				toks := agxRender(tr, 1, it.conv, r.segs, tn.in, tmpl)
				own, cross := agxPrefixReuse(seen[tmpl], own[tmpl][it.conv], toks)
				for _, k := range []string{kind, "all"} {
					a := stats[k][tmpl]
					a.tokens += int64(len(toks))
					a.own += own
					a.cross += cross
				}
			}
		}
		for i, l := range limits {
			if tr.Peak <= l {
				keptTraces[i]++
				if wirePeak > l {
					overflowTraces[i]++
				}
			}
		}
	}
	sort.Float64s(overheadPct)
	sort.Float64s(msgs)
	q := func(v []float64, p float64) float64 { return v[int(p*float64(len(v)-1))] }
	t.Logf("requests %d; messages per prompt median %.0f p90 %.0f; template+marker tokens: %.3f%% of content token-weighted, per request median %.3f%% p90 %.3f%%",
		len(msgs), q(msgs, 0.5), q(msgs, 0.9), 100*float64(overheadTok)/float64(contentTok), q(overheadPct, 0.5), q(overheadPct, 0.9))
	for i, l := range limits {
		t.Logf("max_context_length %d: %d traces kept, %d of them have a request whose wire prompt + max(out,1) exceeds it", l, keptTraces[i], overflowTraces[i])
	}
	for _, k := range []string{"subagent", "parent", "all"} {
		for _, tmpl := range []bool{false, true} {
			a := stats[k][tmpl]
			model := "hash-id identity (no template)"
			if tmpl {
				model = "template (message boundaries)"
			}
			t.Logf("%-8s %-31s reuse own stream %5.2f%%  other streams %5.2f%%  total %5.2f%%", k, model,
				100*float64(a.own)/float64(a.tokens), 100*float64(a.cross)/float64(a.tokens), 100*float64(a.own+a.cross)/float64(a.tokens))
		}
	}
}

// agxPrefixReuse walks a prompt's 64-token blocks (prefix-chained keys, as a KV cache keys
// them) against the blocks the tree's earlier requests sent. The matched prefix is the
// request's reuse; a matched block its own stream sent before counts as own-stream reuse,
// one only other streams sent as cross-stream reuse. It then records the prompt's blocks.
func agxPrefixReuse(seen map[uint64]bool, own map[uint64]bool, toks []sim.TokenID) (ownTok, crossTok int64) {
	var key uint64 = 1469598103934665603
	matching := true
	var keys []uint64
	for b := 0; b+agxBlockSize <= len(toks); b += agxBlockSize {
		h := fnv.New64a()
		var buf [8]byte
		for i := 0; i < 8; i++ {
			buf[i] = byte(key >> (8 * i))
		}
		_, _ = h.Write(buf[:])
		for _, tk := range toks[b : b+agxBlockSize] {
			_, _ = h.Write([]byte{byte(tk), byte(tk >> 8), byte(tk >> 16), byte(tk >> 24)})
		}
		key = h.Sum64()
		keys = append(keys, key)
		if matching && seen[key] {
			if own[key] {
				ownTok += agxBlockSize
			} else {
				crossTok += agxBlockSize
			}
		} else {
			matching = false
		}
	}
	for _, k := range keys {
		seen[k] = true
		own[k] = true
	}
	return ownTok, crossTok
}
