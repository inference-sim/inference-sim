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
	Sends []agxSend `json:"sends"`
}

type agxSend struct {
	T         float64 `json:"t"`
	Phase     string  `json:"phase"`
	Conv      string  `json:"conv"`
	Turn      int     `json:"turn"`
	Depth     int     `json:"depth"`
	In        int     `json:"in"`
	Out       int     `json:"out"`
	NBlocks   int     `json:"nblocks"`
	BlocksSHA string  `json:"blocks_sha"`
	Cancelled bool    `json:"cancelled"`
}

// agxConformanceService is gen_aiperf_runtime.py's service_s: 0.5 + in/200000 + out/100 s.
func agxConformanceService(r *sim.Request) int64 {
	s := 0.5 + float64(len(r.InputTokens))/200000.0 + float64(len(r.OutputTokens))/100.0
	return int64(math.Round(s * 1e6))
}

// sendsOf turns the harness's issues into the dump's shape, reading each prompt's leading
// hash-block layout back from the trace (the blocks the prompt was built from).
func sendsOf(h *agxHarness) []agxSend {
	var out []agxSend
	for _, is := range h.issued {
		d := is.req.Agentic
		var tr *AgentXTrace
		for _, in := range h.x.instances {
			if in.id == d.Tree {
				tr = in.trace
			}
		}
		var conv int32
		for ci := range tr.convs {
			if tr.convs[ci].sessionID == d.Stream {
				conv = int32(ci)
			}
		}
		tn := tr.convs[conv].turns[d.Turn]
		covered := min(int(tn.nHash), int(tn.in)/agxBlockSize)
		blocks := tr.hashPrefix(tn.hashNode, tn.nHash, covered, nil)
		parts := make([]string, len(blocks))
		for i, b := range blocks {
			parts[i] = strconv.FormatInt(b, 10)
		}
		sum := sha256.Sum256([]byte(strings.Join(parts, ",")))
		phase := "profile"
		if is.phase == sim.PhaseWarmup {
			phase = "warmup"
		}
		depth := 0
		if d.Kind != "root" {
			depth = 1
		}
		out = append(out, agxSend{
			T: float64(is.at) / 1e6, Phase: phase, Conv: d.Stream, Turn: d.Turn, Depth: depth,
			In: is.in, Out: is.out, NBlocks: len(blocks), BlocksSHA: hex.EncodeToString(sum[:])[:16],
			Cancelled: is.doneAt < 0,
		})
	}
	return out
}

func sortSends(s []agxSend) {
	sort.SliceStable(s, func(i, j int) bool {
		if math.Abs(s[i].T-s[j].T) > 1e-3 {
			return s[i].T < s[j].T
		}
		if s[i].Conv != s[j].Conv {
			return s[i].Conv < s[j].Conv
		}
		return s[i].Turn < s[j].Turn
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
// lanes and every send against aiperf's.
func compareRuntimeToAiperf(t *testing.T, c *AgentXCorpus, ref *aiperfRuntime) {
	t.Helper()
	cfg := AgentXReplayConfig{Concurrency: ref.Concurrency, Seed: 42, DurationUs: int64(ref.DurationS * 1e6),
		GraceUs: int64(ref.GraceS * 1e6), IdleCapUs: 10e6, MinRatio: 0, MaxRatio: 1}
	h := runAgentX(t, c, cfg, agxConformanceService)
	for i, l := range h.x.lanes {
		want, _ := strconv.ParseFloat(ref.Lanes[i].TStarMs, 64)
		if l.trace.ID != ref.Lanes[i].Trace || l.tStarMs != want {
			t.Errorf("lane %d: %s %x, aiperf %s %s", i, l.trace.ID, l.tStarMs, ref.Lanes[i].Trace, ref.Lanes[i].TStarMs)
		}
	}
	got, want := sendsOf(h), append([]agxSend(nil), ref.Sends...)
	sortSends(got)
	sortSends(want)
	if len(got) != len(want) {
		t.Errorf("%d sends, aiperf %d", len(got), len(want))
	}
	bad := 0
	maxDT := 0.0
	for i := 0; i < min(len(got), len(want)); i++ {
		g, w := got[i], want[i]
		tw := g
		tw.T = w.T
		maxDT = math.Max(maxDT, math.Abs(g.T-w.T))
		if tw != w || math.Abs(g.T-w.T) > 2e-6 {
			bad++
			if bad <= 10 {
				t.Errorf("send %d:\n  blis   %+v\n  aiperf %+v", i, g, w)
			}
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d sends differ", bad, len(want))
	}
	t.Logf("C=%d: %d sends match aiperf; max issue-time difference %.1f us", ref.Concurrency, len(want), maxDT*1e6)
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
		t.Run(fmt.Sprintf("C=%d", conc), func(t *testing.T) {
			compareRuntimeToAiperf(t, c, loadAiperfRuntime(t, fmt.Sprintf("testdata/agentx/aiperf_runtime_slice_c%d.json", conc)))
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
