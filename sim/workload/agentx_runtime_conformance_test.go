package workload

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// aiperfRuntime is gen_aiperf_runtime.py's dump: every request aiperf 0.12.0's own timing
// stack sent, in virtual time, with a fixed service time per request.
type aiperfRuntime struct {
	Concurrency int     `json:"concurrency"`
	DurationS   float64 `json:"duration_s"`
	GraceS      float64 `json:"grace_s"`
	Lanes       []struct {
		Trace   string `json:"trace"`
		TStarMs string `json:"t_star_ms"`
	} `json:"lanes"`
	Sends           []agxSend `json:"sends"`
	ProfilingStartS *float64  `json:"profiling_start_s"`
	ProfilingEndS   *float64  `json:"profiling_requests_end_s"`
}

type agxSend struct {
	T       float64  `json:"t"`
	Phase   string   `json:"phase"`
	Tree    int      `json:"tree"`
	Conv    string   `json:"conv"`
	Turn    int      `json:"turn"`
	Depth   int      `json:"depth"`
	In      int      `json:"in"`
	Out     int      `json:"out"`
	Msgs    int      `json:"msgs"`
	MsgsSHA string   `json:"msgs_sha"`
	Done    *float64 `json:"done"`
}

// agxContentLen is the prompt's message content, without template and marker tokens.
func agxContentLen(toks []sim.TokenID) int {
	n := 0
	for _, t := range toks {
		if t >= agxContentBase {
			n++
		}
	}
	return n - agxMarkerTokens
}

// agxConformanceService is gen_aiperf_runtime.py's service_s on the prompt's content
// length: 0.5 + in/200000 + out/100 s, in microseconds.
func agxConformanceService(r *sim.Request) int64 {
	s := 0.5 + float64(agxContentLen(r.InputTokens))/200000.0 + float64(len(r.OutputTokens))/100.0
	return int64(math.Round(s * 1e6))
}

// blockIndex maps a trace's hash-block tokens back to every (hash id, offset) they appear
// at (single 30-bit tokens collide; a 64-token block does not).
type blockIndex map[sim.TokenID][][2]int64

var blockIndexes = map[*AgentXTrace]blockIndex{}

func blockIndexFor(tr *AgentXTrace) blockIndex {
	if ix, ok := blockIndexes[tr]; ok {
		return ix
	}
	ix := blockIndex{}
	for _, h := range tr.trieHash {
		for i := int32(0); i < agxBlockSize; i++ {
			k := agxBlockToken(tr, h, i)
			ix[k] = append(ix[k], [2]int64{h, int64(i)})
		}
	}
	blockIndexes[tr] = ix
	return ix
}

// decodePrompt reads a sent prompt's ACTUAL tokens back into gen_aiperf_runtime.py's
// canonical message string: it checks the template (BOS, a role header per message, EOS
// after each assistant message, the generation prompt) and the tree's marker at the head of
// the first user message, and renders each message's content as hash-block stretches and
// tail runs.
func decodePrompt(t agxTB, in *agxInstance, toks []sim.TokenID) (canon string, msgs, content int) {
	if len(toks) < 2 || toks[0] != agxTokBOS || toks[len(toks)-1] != agxTokAssistant {
		t.Fatalf("prompt does not start with BOS and end with the generation prompt")
	}
	ix := blockIndexFor(in.trace)
	marker := agxMix(in.salt, 0x6d61726b6572)
	var parts []string
	var items []string
	var lastH, lastEnd int64 = -1, -1
	tailRun := 0
	flush := func() {
		if tailRun > 0 {
			items = append(items, fmt.Sprintf("t%d", tailRun))
			tailRun = 0
		}
	}
	role := byte(0)
	seenUser := false
	body := toks[1 : len(toks)-1]
	finishMsg := func() {
		if role == 0 {
			return
		}
		flush()
		if lastH >= 0 {
			items[len(items)-1] += fmt.Sprintf("%d", lastEnd)
			lastH = -1
		}
		parts = append(parts, fmt.Sprintf("%c:%s", role, strings.Join(items, ",")))
		items = nil
		msgs++
	}
	for i := 0; i < len(body); i++ {
		tok := body[i]
		switch tok {
		case agxTokUser, agxTokAssistant, agxTokSystem:
			finishMsg()
			role = map[sim.TokenID]byte{agxTokUser: 'u', agxTokAssistant: 'a', agxTokSystem: 's'}[tok]
			if role == 'u' && !seenUser {
				seenUser = true
				for j := 0; j < agxMarkerTokens; j++ {
					if body[i+1+j] != agxToken(marker, uint64(j)) {
						t.Fatalf("first user message does not open with the tree's marker")
					}
				}
				i += agxMarkerTokens
			}
			continue
		case agxTokEOS:
			if role != 'a' {
				t.Fatalf("EOS outside an assistant message")
			}
			continue
		}
		content++
		// Continue the current stretch when the token is its next one; otherwise start a
		// stretch at the candidate whose following tokens match.
		if lastH >= 0 && lastEnd < agxBlockSize && tok == agxBlockToken(in.trace, lastH, int32(lastEnd)) {
			lastEnd++
			continue
		}
		// Of the candidates whose next tokens (up to 3, within the block) all match, take the
		// one that verifies the most: a candidate at a block's last offsets has little or no
		// lookahead, so a 30-bit collision there would otherwise win over the true block's
		// first token (seen on the full corpus at C=128).
		var hi [2]int64
		ok, best := false, int64(-1)
		for _, cand := range ix[tok] {
			match, verified := true, int64(0)
			for j := int64(1); j < 4 && i+int(j) < len(body) && cand[1]+j < agxBlockSize; j++ {
				if body[i+int(j)] != agxBlockToken(in.trace, cand[0], int32(cand[1]+j)) {
					match = false
					break
				}
				verified++
			}
			if match && verified > best {
				hi, ok, best = cand, true, verified
			}
		}
		if !ok {
			if lastH >= 0 {
				items[len(items)-1] += fmt.Sprintf("%d", lastEnd)
				lastH = -1
			}
			tailRun++
			continue
		}
		flush()
		if lastH >= 0 {
			items[len(items)-1] += fmt.Sprintf("%d", lastEnd)
		}
		items = append(items, fmt.Sprintf("h%d.%d-", hi[0], hi[1]))
		lastH, lastEnd = hi[0], hi[1]+1
	}
	finishMsg()
	return strings.Join(parts, "|"), msgs, content
}

// sendsOf turns the harness's issues into the dump's shape, decoding each prompt from the
// tokens BLIS actually sent.
func sendsOf(t agxTB, h *agxHarness) []agxSend {
	insts := map[int]*agxInstance{}
	for _, in := range h.x.instances {
		insts[in.id] = in
	}
	var out []agxSend
	for _, is := range h.issued {
		d := is.req.Agentic
		canon, msgs, content := decodePrompt(t, insts[d.Tree], is.req.InputTokens)
		sum := sha256.Sum256([]byte(canon))
		phase := "profile"
		if is.phase == sim.PhaseWarmup {
			phase = "warmup"
		}
		depth := 0
		if d.Kind != "root" {
			depth = 1
		}
		var done *float64
		if is.doneAt >= 0 {
			v := float64(is.doneAt) / 1e6
			done = &v
		}
		out = append(out, agxSend{
			T: float64(is.at) / 1e6, Phase: phase, Tree: d.Tree, Conv: d.Stream, Turn: d.Turn, Depth: depth,
			In: content, Out: is.out, Msgs: msgs, MsgsSHA: hex.EncodeToString(sum[:])[:16], Done: done,
		})
	}
	return out
}

func sortSends(s []agxSend) {
	key := func(x agxSend) (int64, int, string, int, string) {
		return int64(math.Round(x.T * 1e6)), x.Tree, x.Conv, x.Turn, x.Phase
	}
	sort.SliceStable(s, func(i, j int) bool {
		a1, a2, a3, a4, a5 := key(s[i])
		b1, b2, b3, b4, b5 := key(s[j])
		switch {
		case a1 != b1:
			return a1 < b1
		case a2 != b2:
			return a2 < b2
		case a3 != b3:
			return a3 < b3
		case a4 != b4:
			return a4 < b4
		}
		return a5 < b5
	})
}

func loadAiperfRuntime(t *testing.T, path string) *aiperfRuntime {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var ref aiperfRuntime
	if err := json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	return &ref
}

// compareRuntimeToAiperf runs BLIS's replay under the dump's service time and checks
// lanes, every send (time, tree, stream, turn, phase, content length, message structure
// decoded from the sent tokens, max_tokens, completion time or cancellation) and the
// profiling window against aiperf's.
func compareRuntimeToAiperf(t *testing.T, c *AgentXCorpus, ref *aiperfRuntime) {
	t.Helper()
	cfg := AgentXReplayConfig{Concurrency: ref.Concurrency, Seed: 42, DurationUs: int64(ref.DurationS * 1e6),
		GraceUs: int64(ref.GraceS * 1e6), IdleCapUs: 10e6, MinRatio: 0, MaxRatio: 1}
	h := runAgentX(t, c, cfg, agxConformanceService)
	if len(h.x.lanes) != len(ref.Lanes) {
		t.Fatalf("%d lanes, aiperf %d", len(h.x.lanes), len(ref.Lanes))
	}
	for i, l := range h.x.lanes {
		want, _ := strconv.ParseFloat(ref.Lanes[i].TStarMs, 64)
		if l.trace.ID != ref.Lanes[i].Trace || l.tStarMs != want {
			t.Errorf("lane %d: %s %x, aiperf %s %s", i, l.trace.ID, l.tStarMs, ref.Lanes[i].Trace, ref.Lanes[i].TStarMs)
		}
	}
	got, want := sendsOf(t, h), append([]agxSend(nil), ref.Sends...)
	sortSends(got)
	sortSends(want)
	if len(got) != len(want) {
		t.Errorf("%d sends, aiperf %d", len(got), len(want))
	}
	bad := 0
	maxDT := 0.0
	near := func(a, b float64) bool { maxDT = math.Max(maxDT, math.Abs(a-b)); return math.Abs(a-b) <= 2e-6 }
	for i := 0; i < min(len(got), len(want)); i++ {
		g, w := got[i], want[i]
		tw := g
		tw.T, tw.Done = w.T, w.Done
		doneOK := (g.Done == nil) == (w.Done == nil) && (g.Done == nil || near(*g.Done, *w.Done))
		if tw != w || !near(g.T, w.T) || !doneOK {
			bad++
			if bad <= 10 {
				t.Errorf("send %d:\n  blis   %+v done=%v\n  aiperf %+v done=%v", i, g, ptrF(g.Done), w, ptrF(w.Done))
			}
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d sends differ", bad, len(want))
	}
	start, end := h.x.Window()
	if ref.ProfilingStartS != nil && !near(float64(start)/1e6, *ref.ProfilingStartS) {
		t.Errorf("profiling start %d us, aiperf %v s", start, *ref.ProfilingStartS)
	}
	if ref.ProfilingEndS != nil && !near(float64(end)/1e6, *ref.ProfilingEndS) {
		t.Errorf("profiling completion %d us, aiperf %v s", end, *ref.ProfilingEndS)
	}
	t.Logf("C=%d: %d sends match aiperf; max time difference %.1f us", ref.Concurrency, len(want), maxDT*1e6)
}

func ptrF(p *float64) any {
	if p == nil {
		return nil
	}
	return *p
}

// TestAgentXRuntimeMatchesAiperf is the differential conformance test: aiperf 0.12.0's
// own warm-up, profiling, branch, join, barrier, idle-cap, recycle and grace logic
// (gen_aiperf_runtime.py drives it in virtual time behind a stub transport) and BLIS's
// replay, under the same per-request service time on the six-trace slice, send the same
// requests (stream, turn, phase, prompt length and hash-block layout, max_tokens) at the
// same times, and leave the same requests unfinished at the grace end.
func TestAgentXRuntimeMatchesAiperf(t *testing.T) {
	c, err := LoadAgentXCorpus("testdata/agentx/weka_v7_slice.jsonl", "", 0, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, conc := range []int{1, 4, 9} {
		t.Run(fmt.Sprintf("slice/C=%d", conc), func(t *testing.T) {
			compareRuntimeToAiperf(t, c, loadAiperfRuntime(t, fmt.Sprintf("testdata/agentx/aiperf_runtime_slice_c%d.json", conc)))
		})
	}
	// The synthetic shapes: partial and hashed-partial blocks, truncated hash lists and
	// pull-backs (the prompt structure), a barrier release at the instant of an immediate
	// spawn, a lane that recycles until the window end, background branches.
	syn, err := LoadAgentXCorpus("testdata/agentx/weka_v7_synthetic.jsonl", "", 0, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, conc := range []int{1, 2, 3, 5} {
		t.Run(fmt.Sprintf("synthetic/C=%d", conc), func(t *testing.T) {
			compareRuntimeToAiperf(t, syn, loadAiperfRuntime(t, fmt.Sprintf("testdata/agentx/aiperf_runtime_synthetic_c%d.json", conc)))
		})
	}
}

// TestAgentXRuntimeMatchesAiperfOnFullCorpus is the same over the full corpus, gated on
// AGENTX_CORPUS and AGENTX_AIPERF_RUNTIME (a comma-separated list of dumps).
func TestAgentXRuntimeMatchesAiperfOnFullCorpus(t *testing.T) {
	corpus, dumps := os.Getenv("AGENTX_CORPUS"), os.Getenv("AGENTX_AIPERF_RUNTIME")
	if corpus == "" || dumps == "" {
		t.Skip("set AGENTX_CORPUS and AGENTX_AIPERF_RUNTIME to compare the full corpus")
	}
	c, err := LoadAgentXCorpus(corpus, "", 0, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range strings.Split(dumps, ",") {
		t.Run(p, func(t *testing.T) { compareRuntimeToAiperf(t, c, loadAiperfRuntime(t, p)) })
	}
}
