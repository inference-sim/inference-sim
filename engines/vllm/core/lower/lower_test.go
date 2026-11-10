// Original to blis-vllm.

package lower

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

// vllm_lower.json is vLLM v0.32.0's own answers for every case below, printed by
// oracle/lower/extract.py, which runs the release's functions as written. CI regenerates the file
// and fails if it differs (ci.yml, oracle-tier2), so these tests compare the port with the release
// and not with a copy of it.
type oracleFile struct {
	Commit    string   `json:"commit"`
	Breakable []string `json:"breakable"`
	Capture   []struct {
		Case struct {
			Name           string `json:"name"`
			Mode           string `json:"mode"`
			GraphMode      string `json:"graph_mode"`
			MaxNumSeqs     int    `json:"max_num_seqs"`
			MaxBatched     int    `json:"max_num_batched_tokens"`
			SpecTokens     int    `json:"spec_tokens"`
			Blackwell      bool   `json:"blackwell"`
			EnforceEager   bool   `json:"enforce_eager"`
			MaxCapture     *int   `json:"max_capture"`
			Sizes          []int  `json:"sizes"`
			PerformanceMod string `json:"performance_mode"`
			Varlen         bool   `json:"varlen"`
		} `json:"case"`
		Resolved struct {
			Max   int   `json:"max"`
			Sizes []int `json:"sizes"`
		} `json:"resolved"`
		Dispatch []struct {
			Reqs    int  `json:"reqs"`
			Tokens  int  `json:"tokens"`
			Uniform *int `json:"uniform"`
			Got     struct {
				Mode        string `json:"mode"`
				Tokens      int    `json:"tokens"`
				Reqs        *int   `json:"reqs"`
				Uniform     *int   `json:"uniform"`
				MaxQueryLen *int   `json:"max_query_len"`
			} `json:"got"`
		} `json:"dispatch"`
	} `json:"capture"`
	DecodeLen []struct {
		Spec *int `json:"spec"`
		Par  bool `json:"parallel_drafting"`
		V2   bool `json:"v2"`
		Got  int  `json:"got"`
	} `json:"max_decode_query_len"`
	DPSync []struct {
		Case struct {
			Name       string `json:"name"`
			Mode       string `json:"mode"`
			Seqs       int    `json:"seqs"`
			MNBT       int    `json:"mnbt"`
			SpecTokens int    `json:"spec_tokens"`
			Ranks      []struct {
				Reqs   int  `json:"reqs"`
				Tokens int  `json:"tokens"`
				MQL    *int `json:"mql"`
				Dummy  bool `json:"dummy"`
				Unif   *int `json:"uniform"`
			} `json:"ranks"`
		} `json:"case"`
		Results []struct {
			Rank   int    `json:"rank"`
			Mode   string `json:"mode"`
			Tokens int    `json:"tokens"`
			Reqs   *int   `json:"reqs"`
			Sync   *struct {
				Eager  bool  `json:"eager"`
				Reqs   int   `json:"reqs"`
				Across []int `json:"across"`
			} `json:"sync"`
		} `json:"results"`
	} `json:"dp_sync"`
	Context []struct {
		Steps        int `json:"steps"`
		MaxNumTokens int `json:"max_num_tokens"`
		Reqs         int `json:"reqs"`
		Got          int `json:"got"`
	} `json:"graph_context_tokens"`
}

func loadOracle(t *testing.T) oracleFile {
	t.Helper()
	raw, err := os.ReadFile("testdata/vllm_lower.json")
	if err != nil {
		t.Fatal(err)
	}
	var o oracleFile
	if err := json.Unmarshal(raw, &o); err != nil {
		t.Fatal(err)
	}
	if o.Commit != "5df63a7203cdb9328d824a27b682e77c7b9928be" {
		t.Fatalf("the oracle file is for commit %s, not the pinned release", o.Commit)
	}
	return o
}

func TestCaptureSizesAreVLLMs(t *testing.T) {
	o := loadOracle(t)
	if len(o.Capture) == 0 {
		t.Fatal("no cases")
	}
	for _, c := range o.Capture {
		t.Run(c.Case.Name, func(t *testing.T) {
			mode, err := ParseGraphMode(c.Case.Mode)
			if err != nil {
				t.Fatal(err)
			}
			got, err := CaptureSizes(CaptureInput{Mode: mode, EnforceEager: c.Case.EnforceEager,
				MaxCaptureSize: c.Case.MaxCapture, Sizes: c.Case.Sizes, MaxNumSeqs: c.Case.MaxNumSeqs,
				MaxNumBatchedTokens: c.Case.MaxBatched, SpecTokens: c.Case.SpecTokens, Blackwell: c.Case.Blackwell,
				Interactivity: c.Case.PerformanceMod == "interactivity"})
			if err != nil {
				t.Fatal(err)
			}
			if got.Max != c.Resolved.Max || !slices.Equal(got.Sizes, c.Resolved.Sizes) {
				t.Errorf("max %d sizes %v; vLLM resolves max %d sizes %v", got.Max, got.Sizes, c.Resolved.Max, c.Resolved.Sizes)
			}
		})
	}
}

func TestDispatchIsVLLMs(t *testing.T) {
	o := loadOracle(t)
	checked := 0
	for _, c := range o.Capture {
		if len(c.Dispatch) == 0 {
			continue
		}
		t.Run(c.Case.Name, func(t *testing.T) {
			gm, _ := ParseGraphMode(c.Case.GraphMode)
			dql := 1 + c.Case.SpecTokens
			m, err := NewGraphManager(ManagerInput{Mode: gm, Capture: Capture{Max: c.Resolved.Max, Sizes: c.Resolved.Sizes},
				MaxNumReqs: c.Case.MaxNumSeqs, DecodeQueryLen: dql, VarlenDecode: c.Case.Varlen})
			if err != nil {
				t.Fatal(err)
			}
			for _, p := range c.Dispatch {
				var mql *int
				if c.Case.Varlen && p.Uniform != nil {
					mql = p.Uniform
				}
				got := m.Dispatch(p.Reqs, p.Tokens, p.Uniform, mql)
				w := p.Got
				bad := got.Mode.String() != w.Mode || got.Tokens != w.Tokens || !sameInt(got.Reqs, w.Reqs) ||
					!sameInt(got.Uniform, w.Uniform) || !sameInt(got.MaxQueryLen, w.MaxQueryLen)
				if bad {
					t.Errorf("reqs %d tokens %d uniform %v: got %v %d reqs %v uniform %v bound %v; vLLM %s %d reqs %v uniform %v bound %v",
						p.Reqs, p.Tokens, show(p.Uniform), got.Mode, got.Tokens, show(got.Reqs), show(got.Uniform), show(got.MaxQueryLen),
						w.Mode, w.Tokens, show(w.Reqs), show(w.Uniform), show(w.MaxQueryLen))
				}
				checked++
			}
		})
	}
	if checked < 1000 {
		t.Errorf("only %d dispatch probes ran; the oracle file shrank", checked)
	}
}

func sameInt(a, b *int) bool { return (a == nil) == (b == nil) && (a == nil || *a == *b) }

func show(p *int) any {
	if p == nil {
		return "None"
	}
	return *p
}

func TestMaxDecodeQueryLenIsVLLMs(t *testing.T) {
	for _, c := range loadOracle(t).DecodeLen {
		if got := MaxDecodeQueryLen(c.Spec, c.Par, c.V2); got != c.Got {
			t.Errorf("spec %v parallel_drafting %v v2 %v: %d; vLLM %d", show(c.Spec), c.Par, c.V2, got, c.Got)
		}
	}
}

func TestGraphContextTokensAreVLLMs(t *testing.T) {
	for _, c := range loadOracle(t).Context {
		if got := GraphContextTokens(c.Steps, c.MaxNumTokens, c.Reqs); got != c.Got {
			t.Errorf("steps %d max %d reqs %d: %d; vLLM %d", c.Steps, c.MaxNumTokens, c.Reqs, got, c.Got)
		}
	}
}

func TestBreakableArchitecturesAreVLLMs(t *testing.T) {
	o := loadOracle(t)
	if got := BreakableArchitectures(); !slices.Equal(got, o.Breakable) {
		t.Errorf("breakable architectures %v; vLLM %v", got, o.Breakable)
	}
	if !Breakable("DeepseekV41ForCausalLM") || Breakable("LlamaForCausalLM") {
		t.Error("Breakable misreads the list")
	}
}

func TestCaptureRefusals(t *testing.T) {
	five := 5
	for _, c := range []struct {
		name string
		in   CaptureInput
	}{
		{"stated max and sizes disagree", CaptureInput{Mode: GraphPiecewise, MaxCaptureSize: &five, Sizes: []int{8, 16},
			MaxNumSeqs: 8, MaxNumBatchedTokens: 64}},
		{"empty stated sizes", CaptureInput{Mode: GraphPiecewise, Sizes: []int{}, MaxNumSeqs: 8, MaxNumBatchedTokens: 64}},
		{"no batched tokens", CaptureInput{Mode: GraphPiecewise, MaxNumSeqs: 8}},
	} {
		if _, err := CaptureSizes(c.in); err == nil {
			t.Errorf("%s: no error", c.name)
		}
	}
	// The closest input that is accepted: the stated maximum is the largest stated size.
	eight := 16
	if _, err := CaptureSizes(CaptureInput{Mode: GraphPiecewise, MaxCaptureSize: &eight, Sizes: []int{8, 16},
		MaxNumSeqs: 8, MaxNumBatchedTokens: 64}); err != nil {
		t.Errorf("a consistent pair was refused: %v", err)
	}
}

// SyncDP against vLLM's sync_cudagraph_and_dp_padding, run unmodified with its all-reduce supplied
// the ranks' tensors. The cases cover the rules the issue names: equal and unequal uniform decodes,
// a prefill rank that takes every rank off the decode graph, one rank past the capture ceiling that
// takes every rank eager, a decode-only mode with a mixed rank, ranks that disagree on the uniform
// count, an idle rank running the dummy batch, and every rank idle.
func TestSyncDPIsVLLMs(t *testing.T) {
	o := loadOracle(t)
	if len(o.DPSync) < 10 {
		t.Fatalf("only %d data-parallel cases in the oracle file", len(o.DPSync))
	}
	for _, c := range o.DPSync {
		t.Run(c.Case.Name, func(t *testing.T) {
			gm, _ := ParseGraphMode(c.Case.Mode)
			dql := 1 + c.Case.SpecTokens
			cap, err := CaptureSizes(CaptureInput{Mode: gm, MaxNumSeqs: c.Case.Seqs, MaxNumBatchedTokens: c.Case.MNBT,
				SpecTokens: c.Case.SpecTokens})
			if err != nil {
				t.Fatal(err)
			}
			m, err := NewGraphManager(ManagerInput{Mode: gm, Capture: cap, MaxNumReqs: c.Case.Seqs, DecodeQueryLen: dql})
			if err != nil {
				t.Fatal(err)
			}
			var ranks []RankBatch
			for _, k := range c.Case.Ranks {
				r := RankBatch{Reqs: k.Reqs, Tokens: k.Tokens, HasPrefill: k.MQL == nil && k.Tokens > 0,
					DecodeEligible: k.Unif != nil}
				if k.MQL != nil {
					r.MaxQueryLen = *k.MQL
				} else {
					r.MaxQueryLen = k.Tokens
				}
				if k.Dummy {
					r = DummyBatch(dql)
				}
				ranks = append(ranks, r)
			}
			got, syncs, err := SyncDP(m, ranks)
			if err != nil {
				t.Fatal(err)
			}
			if len(c.Results) == 0 {
				for i, l := range got {
					if l.Tokens != 0 {
						t.Errorf("rank %d launches %d tokens; every rank is idle", i, l.Tokens)
					}
				}
				return
			}
			for _, w := range c.Results {
				if got[w.Rank].Mode.String() != w.Mode || got[w.Rank].Tokens != w.Tokens {
					t.Errorf("rank %d: %v %d tokens; vLLM %s %d", w.Rank, got[w.Rank].Mode, got[w.Rank].Tokens, w.Mode, w.Tokens)
				}
				if w.Sync != nil && (syncs[w.Rank].Eager != w.Sync.Eager || syncs[w.Rank].Reqs != w.Sync.Reqs) {
					t.Errorf("rank %d sync: eager %v reqs %d; vLLM eager %v reqs %d", w.Rank, syncs[w.Rank].Eager, syncs[w.Rank].Reqs,
						w.Sync.Eager, w.Sync.Reqs)
				}
			}
		})
	}
}

// One lighter rank is padded to the group's launch: the property that the price of a lighter rank carries the
// group's padding, stated on the plan side as the launched token count of each rank.
func TestALighterRankLaunchesTheGroupsShape(t *testing.T) {
	cap, _ := CaptureSizes(CaptureInput{Mode: GraphFullAndPiecewise, MaxNumSeqs: 256, MaxNumBatchedTokens: 8192})
	m, _ := NewGraphManager(ManagerInput{Mode: GraphFullAndPiecewise, Capture: cap, MaxNumReqs: 256, DecodeQueryLen: 1})
	decode := func(n int) RankBatch { return RankBatch{Reqs: n, Tokens: n, MaxQueryLen: 1, DecodeEligible: true} }
	got, _, err := SyncDP(m, []RankBatch{decode(3), decode(20)})
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != got[1] || got[0].Tokens < 20 || got[0].Mode != GraphFull {
		t.Errorf("launches %+v; a 3-request rank joins a 20-request rank's FULL graph", got)
	}
}

// The drafter's eager context store is decided on the graph's padded request count, not on the
// requests the step has: nine requests pad to the graph of ten, whose
// ten full verifies hold the 58 target tokens, so nothing is stored eagerly; two tokens past what
// the padded graph holds, the surplus is.
func TestTheEagerContextStoreFollowsThePaddedGraph(t *testing.T) {
	const steps = 5
	cap, err := CaptureSizes(CaptureInput{Mode: GraphFullDecodeOnly, MaxNumSeqs: 64, MaxNumBatchedTokens: 8192, SpecTokens: steps})
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewGraphManager(ManagerInput{Mode: GraphFullDecodeOnly, Capture: cap, MaxNumReqs: 64, DecodeQueryLen: 1 + steps})
	if err != nil {
		t.Fatal(err)
	}
	pad, _ := Drafter(m, DrafterInput{Steps: steps, MaxNumTokens: 8192, Reqs: 9, TargetTokens: 1})
	if pad.Launch.Mode != GraphFull || pad.Launch.Tokens%(1+steps) != 0 {
		t.Fatalf("nine requests launch %+v", pad.Launch)
	}
	graphReqs := pad.Launch.Tokens / (1 + steps)
	if graphReqs <= 9 {
		t.Fatalf("nine requests did not pad (graph of %d requests); the case needs a padded graph", graphReqs)
	}
	held := graphReqs * (1 + steps)
	for _, c := range []struct{ target, want int }{{held, 0}, {held - 1, 0}, {held + 2, 2}} {
		got, err := Drafter(m, DrafterInput{Steps: steps, MaxNumTokens: 8192, Reqs: 9, TargetTokens: c.target})
		if err != nil {
			t.Fatal(err)
		}
		if got.EagerContextRows != c.want {
			t.Errorf("target %d tokens: %d eager rows, want %d", c.target, got.EagerContextRows, c.want)
		}
	}
	// By the unpadded count the 9 requests hold 54 rows, so a 58-token batch would be charged 4.
	if unpadded := 58 - 9*(1+steps); unpadded != 4 {
		t.Fatalf("arithmetic of the example: %d", unpadded)
	}
	got, _ := Drafter(m, DrafterInput{Steps: steps, MaxNumTokens: 8192, Reqs: 9, TargetTokens: 58})
	if got.EagerContextRows != 0 {
		t.Errorf("58 target tokens on nine requests: %d eager rows; the padded graph holds %d", got.EagerContextRows, held)
	}
}

// The drafter of a data-parallel pool is as wide as the group's agreement, not as its own batch.
func TestTheDrafterReusesTheTargetsAgreement(t *testing.T) {
	const steps = 3
	cap, _ := CaptureSizes(CaptureInput{Mode: GraphFullDecodeOnly, MaxNumSeqs: 64, MaxNumBatchedTokens: 8192, SpecTokens: steps})
	m, _ := NewGraphManager(ManagerInput{Mode: GraphFullDecodeOnly, Capture: cap, MaxNumReqs: 64, DecodeQueryLen: 1 + steps})
	alone, _ := Drafter(m, DrafterInput{Steps: steps, MaxNumTokens: 8192, Reqs: 3, TargetTokens: 12})
	grouped, _ := Drafter(m, DrafterInput{Steps: steps, MaxNumTokens: 8192, Reqs: 3, TargetTokens: 12,
		Sync: &Sync{Reqs: 40, Tokens: 40}})
	if grouped.Launch.Tokens <= alone.Launch.Tokens || grouped.Launch.Tokens < 40*(1+steps) {
		t.Errorf("alone %+v, in a group whose widest rank has 40 requests %+v", alone.Launch, grouped.Launch)
	}
	eager, _ := Drafter(m, DrafterInput{Steps: steps, MaxNumTokens: 8192, Reqs: 3, TargetTokens: 12,
		Sync: &Sync{Reqs: 40, Eager: true}})
	if eager.Launch.Mode != GraphNone || eager.EagerContextRows != 12 {
		t.Errorf("ranks that agreed to run eager: %+v", eager)
	}
}

// The decode threshold is the engine's, derived from the speculative config, so a verify of up to
// 1+K tokens is a decode and a chunk of K+2 launches the prefill kernels.
func TestADecodeIsAsWideAsAVerify(t *testing.T) {
	k := 5
	if got := MaxDecodeQueryLen(&k, false, true); got != 6 {
		t.Errorf("K=5: %d", got)
	}
	if got := MaxDecodeQueryLen(nil, false, true); got != 1 {
		t.Errorf("no speculation: %d; the cost model's constant 8 is not vLLM's threshold", got)
	}
	if got := MaxDecodeQueryLen(&k, true, false); got != 11 {
		t.Errorf("V1 runner with a parallel drafter: %d, want 1+K+K", got)
	}
}
