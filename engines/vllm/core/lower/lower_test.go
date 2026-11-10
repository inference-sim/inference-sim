// Original to blis-vllm.

package lower

import (
	"encoding/json"
	"os"
	"slices"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/units"
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
			KDA            bool   `json:"kda"`
		} `json:"case"`
		Resolved struct {
			Max   int     `json:"max"`
			Sizes []int   `json:"sizes"`
			Error *string `json:"error"`
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
			Varlen     bool   `json:"varlen"`
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
	Engram struct {
		Resolve []struct {
			Case struct {
				Stated  *bool `json:"stated"`
				Offload bool  `json:"cpu_offload"`
				THP     bool  `json:"use_thp"`
				DP      int   `json:"dp"`
				Elastic bool  `json:"elastic"`
			} `json:"case"`
			Resolved bool    `json:"resolved"`
			Error    *string `json:"error"`
		} `json:"resolve"`
		Shard []struct {
			Case struct {
				TP     int  `json:"tp"`
				TPRank int  `json:"tp_rank"`
				EDP    int  `json:"edp"`
				DPRank int  `json:"dp_rank"`
				Shared bool `json:"shared"`
			} `json:"case"`
			Got [2]int `json:"got"`
		} `json:"shard"`
		ParallelSize []struct {
			TP     int  `json:"tp"`
			DP     int  `json:"dp"`
			Got    int  `json:"got"`
			Across bool `json:"across"`
		} `json:"parallel_size"`
	} `json:"engram"`
	Drafter []struct {
		Case struct {
			Name   string `json:"name"`
			Steps  int    `json:"steps"`
			Reqs   int    `json:"reqs"`
			Target int    `json:"target"`
			Sync   *struct {
				Reqs  int  `json:"reqs"`
				Eager bool `json:"eager"`
			} `json:"sync"`
		} `json:"case"`
		Desc struct {
			Mode   string `json:"mode"`
			Tokens int    `json:"tokens"`
			Reqs   *int   `json:"reqs"`
		} `json:"desc"`
		ContextRows *int `json:"graph_context_rows"`
	} `json:"drafter"`
	DCPPreferences map[string]struct {
		Backend    string `json:"comm_backend"`
		QReplicate bool   `json:"q_replicate"`
	} `json:"dcp_preferences"`
	DBO struct {
		Split []struct {
			Enable  bool `json:"enable_dbo"`
			UBatch  int  `json:"ubatch_size"`
			Decode  int  `json:"decode"`
			Prefill int  `json:"prefill"`
			Use     bool `json:"use"`
			Num     int  `json:"num"`
			Steps   []struct {
				Tokens  int  `json:"tokens"`
				Uniform bool `json:"uniform"`
				Split   bool `json:"split"`
			} `json:"steps"`
		} `json:"split"`
		Defaults []struct {
			StatedBackend *string `json:"stated_backend"`
			StatedQ       *bool   `json:"stated_q"`
			ModelBackend  string  `json:"model_backend"`
			ModelQ        bool    `json:"model_q"`
			Backend       string  `json:"backend"`
			Q             bool    `json:"q"`
		} `json:"defaults"`
	} `json:"dbo"`
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
			if (err != nil) != (c.Resolved.Error != nil) {
				t.Fatalf("error %v; vLLM's %v", err, show2(c.Resolved.Error))
			}
			if err != nil {
				return
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
		c := c
		t.Run(c.Case.Name, func(t *testing.T) {
			gm, _ := ParseGraphMode(c.Case.GraphMode)
			dql := 1 + c.Case.SpecTokens
			m, err := NewGraphManager(ManagerInput{Mode: gm, Capture: Capture{Max: c.Resolved.Max, Sizes: c.Resolved.Sizes},
				MaxNumReqs: c.Case.MaxNumSeqs, DecodeQueryLen: dql, VarlenDecode: c.Case.Varlen, UseKDARecoverSSM: c.Case.KDA})
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

func show2(s *string) string {
	if s == nil {
		return "none"
	}
	return *s
}

func sameInt(a, b *int) bool { return (a == nil) == (b == nil) && (a == nil || *a == *b) }

func show(p *int) any {
	if p == nil {
		return "None"
	}
	return *p
}

func TestMaxDecodeQueryLenIsVLLMs(t *testing.T) {
	o := loadOracle(t)
	if len(o.DecodeLen) < 20 {
		t.Fatalf("only %d decode-length cases in the oracle file", len(o.DecodeLen))
	}
	for _, c := range o.DecodeLen {
		if got := MaxDecodeQueryLen(c.Spec, c.Par, c.V2); got != c.Got {
			t.Errorf("spec %v parallel_drafting %v v2 %v: %d; vLLM %d", show(c.Spec), c.Par, c.V2, got, c.Got)
		}
	}
}

func TestGraphContextTokensAreVLLMs(t *testing.T) {
	o := loadOracle(t)
	if len(o.Context) < 36 {
		t.Fatalf("only %d graph-context cases in the oracle file", len(o.Context))
	}
	for _, c := range o.Context {
		if got := GraphContextTokens(c.Steps, c.MaxNumTokens, c.Reqs); got != c.Got {
			t.Errorf("steps %d max %d reqs %d: %d; vLLM %d", c.Steps, c.MaxNumTokens, c.Reqs, got, c.Got)
		}
	}
}

func TestBreakableArchitecturesAreVLLMs(t *testing.T) {
	o := loadOracle(t)
	if len(o.Breakable) < 20 {
		t.Fatalf("only %d breakable architectures in the oracle file", len(o.Breakable))
	}
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
			m, err := NewGraphManager(ManagerInput{Mode: gm, Capture: cap, MaxNumReqs: c.Case.Seqs, DecodeQueryLen: dql,
				VarlenDecode: c.Case.Varlen})
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
			// An oracle case with no result is not a pass: vLLM's answer is what the case is for.
			if len(c.Results) == 0 {
				t.Fatalf("the oracle file holds no result for %s; regenerate it (oracle/lower/extract.py)", c.Case.Name)
			}
			// vLLM returns the agreement for every rank or for none (None), so the agreements are a slice of
			// one per rank or nil, and a nil is the answer for one rank and for ranks that all have no work.
			wantNone := c.Results[0].Sync == nil
			for _, w := range c.Results {
				if (w.Sync == nil) != wantNone {
					t.Fatalf("the oracle's ranks disagree on whether vLLM returned an agreement: %+v", c.Results)
				}
			}
			if wantNone {
				if syncs != nil {
					t.Errorf("syncs %+v; vLLM returns no agreement (None), so the agreements are nil", syncs)
				}
			} else if len(syncs) != len(ranks) {
				t.Fatalf("%d agreements for %d ranks; vLLM returns one for each", len(syncs), len(ranks))
			}
			for _, w := range c.Results {
				if got[w.Rank].Mode.String() != w.Mode || got[w.Rank].Tokens != w.Tokens {
					t.Errorf("rank %d: %v %d tokens; vLLM %s %d", w.Rank, got[w.Rank].Mode, got[w.Rank].Tokens, w.Mode, w.Tokens)
				}
				if w.Sync != nil {
					s := syncs[w.Rank]
					if s.Eager != w.Sync.Eager || s.Reqs != w.Sync.Reqs || s.Tokens != w.Sync.Across[w.Rank] {
						t.Errorf("rank %d sync: eager %v reqs %d tokens %d; vLLM eager %v reqs %d tokens %d", w.Rank, s.Eager, s.Reqs, s.Tokens,
							w.Sync.Eager, w.Sync.Reqs, w.Sync.Across[w.Rank])
					}
				}
			}
		})
	}
}

// Every rank idle is no agreement, and the agreements are nil, not a slice of zero agreements a caller
// would take for one that was reached; the launches are the ranks' own, with no tokens. Mutation check:
// returning the preallocated slice fails it.
func TestSyncDPAllIdleReturnsNoAgreement(t *testing.T) {
	cap, err := CaptureSizes(CaptureInput{Mode: GraphFullAndPiecewise, MaxNumSeqs: 256, MaxNumBatchedTokens: 8192})
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewGraphManager(ManagerInput{Mode: GraphFullAndPiecewise, Capture: cap, MaxNumReqs: 256, DecodeQueryLen: 1})
	if err != nil {
		t.Fatal(err)
	}
	for n := 1; n <= 3; n++ {
		launches, syncs, err := SyncDP(m, make([]RankBatch, n))
		if err != nil {
			t.Fatal(err)
		}
		if syncs != nil {
			t.Errorf("%d idle rank(s): agreements %+v, want nil", n, syncs)
		}
		if len(launches) != n {
			t.Fatalf("%d idle rank(s): %d launches", n, len(launches))
		}
		for i, l := range launches {
			if l.Tokens != 0 {
				t.Errorf("%d idle rank(s): rank %d launches %d tokens", n, i, l.Tokens)
			}
		}
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
	pad, err := Drafter(m, DrafterInput{Steps: steps, MaxNumTokens: 8192, Reqs: 9, TargetTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
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
	got, err := Drafter(m, DrafterInput{Steps: steps, MaxNumTokens: 8192, Reqs: 9, TargetTokens: 58})
	if err != nil {
		t.Fatal(err)
	}
	if got.EagerContextRows != 0 {
		t.Errorf("58 target tokens on nine requests: %d eager rows; the padded graph holds %d", got.EagerContextRows, held)
	}
}

// The drafter pass is vLLM's: its launch and the context rows its graph stores come from
// _build_uniform_batch_dp_sync and dispatch_cg_and_sync_dp, run as written. The cases include a
// target whose ranks agreed to run eager, for which vLLM still replays the drafter's FULL graph
// (the rebuilt agreement carries eager=False).
func TestTheDrafterPassIsVLLMs(t *testing.T) {
	o := loadOracle(t)
	if len(o.Drafter) < 6 {
		t.Fatalf("%d drafter cases in the oracle file", len(o.Drafter))
	}
	for _, c := range o.Drafter {
		t.Run(c.Case.Name, func(t *testing.T) {
			steps := c.Case.Steps
			cap, err := CaptureSizes(CaptureInput{Mode: GraphFullDecodeOnly, MaxNumSeqs: 64, MaxNumBatchedTokens: 8192, SpecTokens: steps})
			if err != nil {
				t.Fatal(err)
			}
			m, err := NewGraphManager(ManagerInput{Mode: GraphFullDecodeOnly, Capture: cap, MaxNumReqs: 64, DecodeQueryLen: 1 + steps})
			if err != nil {
				t.Fatal(err)
			}
			in := DrafterInput{Steps: steps, MaxNumTokens: 8192, Reqs: c.Case.Reqs, TargetTokens: c.Case.Target}
			if c.Case.Sync != nil {
				in.Sync = &Sync{Reqs: c.Case.Sync.Reqs, Eager: c.Case.Sync.Eager}
			}
			got, err := Drafter(m, in)
			if err != nil {
				t.Fatal(err)
			}
			if got.Launch.Mode.String() != c.Desc.Mode || got.Launch.Tokens != c.Desc.Tokens {
				t.Errorf("launch %v %d tokens; vLLM %s %d", got.Launch.Mode, got.Launch.Tokens, c.Desc.Mode, c.Desc.Tokens)
			}
			if c.ContextRows == nil {
				t.Fatalf("vLLM runs no FULL graph for %s; the case needs one", c.Case.Name)
			}
			if want := max(0, c.Case.Target-*c.ContextRows); got.EagerContextRows != want {
				t.Errorf("%d eager context rows; the graph vLLM replays stores %d of %d target tokens, leaving %d",
					got.EagerContextRows, *c.ContextRows, c.Case.Target, want)
			}
		})
	}
}

// A pool of data-parallel ranks sizes the drafter by the group's agreement, however it was
// reached, and a batch the agreement does not cover is vLLM's assert.
func TestTheDrafterIsAsWideAsTheAgreement(t *testing.T) {
	const steps = 3
	cap, err := CaptureSizes(CaptureInput{Mode: GraphFullDecodeOnly, MaxNumSeqs: 64, MaxNumBatchedTokens: 8192, SpecTokens: steps})
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewGraphManager(ManagerInput{Mode: GraphFullDecodeOnly, Capture: cap, MaxNumReqs: 64, DecodeQueryLen: 1 + steps})
	if err != nil {
		t.Fatal(err)
	}
	alone, err := Drafter(m, DrafterInput{Steps: steps, MaxNumTokens: 8192, Reqs: 3, TargetTokens: 12})
	if err != nil {
		t.Fatal(err)
	}
	grouped, err := Drafter(m, DrafterInput{Steps: steps, MaxNumTokens: 8192, Reqs: 3, TargetTokens: 12,
		Sync: &Sync{Reqs: 40, Tokens: 40}})
	if err != nil {
		t.Fatal(err)
	}
	if grouped.Launch.Tokens <= alone.Launch.Tokens || grouped.Launch.Tokens < 40*(1+steps) {
		t.Errorf("alone %+v, in a group whose widest rank has 40 requests %+v", alone.Launch, grouped.Launch)
	}
	if _, err := Drafter(m, DrafterInput{Steps: steps, MaxNumTokens: 8192, Reqs: 41, TargetTokens: 12,
		Sync: &Sync{Reqs: 40}}); err == nil {
		t.Error("a drafter batch wider than the agreement was accepted")
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

func TestEngramSharingIsVLLMs(t *testing.T) {
	o := loadOracle(t)
	if len(o.Engram.Resolve) < 12 {
		t.Fatalf("%d Engram cases", len(o.Engram.Resolve))
	}
	for _, c := range o.Engram.Resolve {
		got, err := EngramShared(c.Case.Stated, c.Case.Offload, c.Case.THP, c.Case.DP, c.Case.Elastic)
		if got != c.Resolved || (err != nil) != (c.Error != nil) {
			t.Errorf("%+v: shared %v err %v; vLLM shared %v err %v", c.Case, got, err, c.Resolved, c.Error)
		}
	}
	if len(o.Engram.ParallelSize) < 18 {
		t.Fatalf("only %d Engram parallel-size cases in the oracle file", len(o.Engram.ParallelSize))
	}
	for _, c := range o.Engram.ParallelSize {
		if got := EngramParallelSize(c.TP, c.DP, c.Across); got != c.Got {
			t.Errorf("tp %d dp %d across %v: %d; vLLM %d", c.TP, c.DP, c.Across, got, c.Got)
		}
	}
}

// The shard an Engram table's rank holds is vLLM's _get_shard_info, run as written.
func TestEngramShardIsVLLMs(t *testing.T) {
	shards := loadOracle(t).Engram.Shard
	if len(shards) < 50 {
		t.Fatalf("only %d Engram shard cases in the oracle file", len(shards))
	}
	for _, c := range shards {
		s, r := EngramShard(c.Case.TP, c.Case.TPRank, c.Case.EDP, c.Case.DPRank, c.Case.Shared)
		if s != c.Got[0] || r != c.Got[1] {
			t.Errorf("%+v: shard %d of %d; vLLM %d of %d", c.Case, r, s, c.Got[1], c.Got[0])
		}
	}
}

// A node holds one copy of a shared table and one for each rank of an unshared one. Four tensor
// ranks over a table of eight hash heads, two data-parallel replicas of four ranks on one node of
// eight: shared, a rank's shard (two heads) is counted once for the node; unshared, a rank's shard
// is cut over tensor and data parallelism (one head) and counted for each of the node's eight
// ranks. Head sizes differ, so the expected figures are summed from the heads the shard holds.
func TestASharedTableCountsOncePerNode(t *testing.T) {
	heads := []int{101, 103, 107, 109, 113, 127, 131, 137}
	const dim, block = 64, 32
	node := func(ranksOnNode, tp, engramDP int, shared bool) units.Bytes {
		t.Helper()
		var p plan.PoolPlan
		sh, r := EngramShard(tp, 0, engramDP, 0, shared)
		tab, err := EngramTable("engram", heads, dim, block, sh, r, shared, true)
		if err != nil {
			t.Fatal(err)
		}
		p.Weights = []plan.ResidentTensor{tab}
		m, err := plan.MemoryOf(p, ranksOnNode)
		if err != nil {
			t.Fatal(err)
		}
		return m.Host.Value()
	}
	bytesOf := func(h ...int) units.Bytes {
		var n int
		for _, x := range h {
			n += x
		}
		return units.Bytes(n) * (dim + dim/block)
	}
	// Shared, tp 4: rank 0 holds heads 0 and 1, and the node counts them once.
	if got, want := node(8, 4, 2, true), bytesOf(101, 103); got != want {
		t.Errorf("a shared shard on an 8-rank node: %d bytes, want %d once", got, want)
	}
	// Unshared, tp 4 x dp 2: rank 0 holds head 0, counted for each of the node's 8 ranks.
	if got, want := node(8, 4, 2, false), 8*bytesOf(101); got != want {
		t.Errorf("an unshared shard on an 8-rank node: %d bytes, want 8 x %d", got, bytesOf(101))
	}
	if _, err := EngramTable("engram", heads, dim, block, 16, 0, false, true); err == nil {
		t.Error("16 shards of 8 hash heads were accepted; vLLM refuses a layout that leaves ranks without heads")
	}
	tab, err := EngramTable("engram", heads, dim, block, 4, 0, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if !tab.Shared || tab.Where != plan.ResidentHost {
		t.Errorf("a shared table is a host figure that counts once: %+v", tab)
	}
	// Without cpu_offload vLLM allocates the table on the GPU (engram.py:1045-1052), and a table
	// there cannot be shared between replicas.
	gpu, err := EngramTable("engram", heads, dim, block, 4, 0, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if gpu.Where != plan.ResidentGPU {
		t.Errorf("a table with cpu_offload off is on the GPU: %+v", gpu)
	}
	if _, err := EngramTable("engram", heads, dim, block, 4, 0, true, false); err == nil {
		t.Error("a shared table without cpu_offload was accepted")
	}
}

// The draft-logits buffer is (max_num_seqs, K, vocab) in the head dtype for DeepSeek-V4.1 at K=5:
// 129,280 x 5 x 2 bytes a slot, 1,323,827,200 bytes, about 1.2 GiB at 1,024 slots; greedy has none.
func TestTheProbabilisticDraftSamplerChargesItsLogits(t *testing.T) {
	b, ok, err := DraftLogits(true, 1024, 5, 129280, 2)
	if err != nil || !ok {
		t.Fatal(ok, err)
	}
	if want := units.Bytes(1024) * 5 * 129280 * 2; b.Bytes != want || b.Basis != plan.BasisRank || b.Host {
		t.Errorf("%+v, want %d bytes per rank on the GPU", b, want)
	}
	if gib := float64(b.Bytes) / (1 << 30); gib < 1.2 || gib > 1.3 {
		t.Errorf("%.2f GiB, the issue's figure is about 1.2", gib)
	}
	if _, ok, _ := DraftLogits(false, 1024, 5, 129280, 2); ok {
		t.Error("greedy sampling allocated a buffer")
	}
	small, _, _ := DraftLogits(true, 512, 5, 129280, 2)
	if small.Bytes >= b.Bytes {
		t.Error("the buffer does not grow with max_num_seqs")
	}
	f32, _, _ := DraftLogits(true, 1024, 5, 129280, 4)
	if f32.Bytes != 2*b.Bytes {
		t.Error("a float32 cache is not twice a bfloat16 one")
	}
}

// A rank with no tokens among ranks with work would be priced as if it had agreed to run eager;
// vLLM gives such a rank the dummy batch, so the input is an error. One rank has no agreement.
func TestSyncDPRefusesAnIdleRankBesideBusyOnes(t *testing.T) {
	cap, err := CaptureSizes(CaptureInput{Mode: GraphFullAndPiecewise, MaxNumSeqs: 256, MaxNumBatchedTokens: 8192})
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewGraphManager(ManagerInput{Mode: GraphFullAndPiecewise, Capture: cap, MaxNumReqs: 256, DecodeQueryLen: 1})
	if err != nil {
		t.Fatal(err)
	}
	busy := RankBatch{Reqs: 4, Tokens: 4, MaxQueryLen: 1, DecodeEligible: true}
	if _, _, err := SyncDP(m, []RankBatch{{}, busy}); err == nil {
		t.Error("an idle rank beside a busy one was accepted")
	}
	if _, syncs, err := SyncDP(m, []RankBatch{busy}); err != nil || syncs != nil {
		t.Errorf("one rank: syncs %v, err %v; vLLM returns no agreement at dp_size 1", syncs, err)
	}
}

func TestDualBatchOverlapSplitsAsVLLMDoes(t *testing.T) {
	o := loadOracle(t)
	if len(o.DBO.Split) < 24 {
		t.Fatalf("%d DBO cases", len(o.DBO.Split))
	}
	for _, c := range o.DBO.Split {
		d := DBO{Enable: c.Enable, UBatchSize: c.UBatch, DecodeThreshold: c.Decode, PrefillThreshold: c.Prefill}
		want := 1
		if c.Use {
			want = c.Num
		}
		if got := d.Ubatches(); got != want {
			t.Errorf("%+v: %d ubatches; vLLM %d", c, got, want)
		}
		for _, st := range c.Steps {
			if got := d.Splits(st.Tokens, st.Uniform); got != st.Split {
				t.Errorf("enable %v ubatch %d thresholds %d/%d, %d tokens uniform %v: split %v; vLLM %v",
					c.Enable, c.UBatch, c.Decode, c.Prefill, st.Tokens, st.Uniform, got, st.Split)
			}
		}
	}
}

func TestDCPDefaultsAreVLLMs(t *testing.T) {
	o := loadOracle(t)
	if len(o.DBO.Defaults) < 36 {
		t.Fatalf("%d DCP cases", len(o.DBO.Defaults))
	}
	for _, c := range o.DBO.Defaults {
		sb := ""
		if c.StatedBackend != nil {
			sb = *c.StatedBackend
		}
		got, err := DCPResolve(sb, c.StatedQ, DCP{Backend: c.ModelBackend, QReplicate: c.ModelQ})
		if err != nil || got.Backend != c.Backend || got.QReplicate != c.Q {
			t.Errorf("%+v: %+v %v; vLLM %s %v", c, got, err, c.Backend, c.Q)
		}
	}
	// A model's preference is what its hook passes to set_dcp_defaults, read from vLLM's source by
	// the oracle: every architecture it lists has that preference here, and every other one the
	// stock one the function's own defaults give. Nothing below restates a value.
	o2 := loadOracle(t)
	if len(o2.DCPPreferences) == 0 {
		t.Fatal("the oracle lists no model that states a DCP preference")
	}
	stock := DCPModelPreference("an architecture that states none")
	for arch, want := range o2.DCPPreferences {
		got := DCPModelPreference(arch)
		if got.Backend != want.Backend || got.QReplicate != want.QReplicate {
			t.Errorf("%s: preference %+v; vLLM's hook passes %s, %v", arch, got, want.Backend, want.QReplicate)
		}
		if got == stock {
			t.Errorf("%s: its preference equals the stock one, so the table cannot tell it from a model that states none", arch)
		}
	}
	for _, arch := range []string{"LlamaForCausalLM", "DeepseekV32ForCausalLM", ""} {
		if _, stated := o2.DCPPreferences[arch]; !stated && DCPModelPreference(arch) != stock {
			t.Errorf("%s states no preference in vLLM but resolves to %+v, not the stock %+v", arch, DCPModelPreference(arch), stock)
		}
	}
	if _, err := DCPResolve("nccl", nil, DCP{Backend: "ag_rs"}); err == nil {
		t.Error("a backend vLLM does not name was accepted")
	}
}
