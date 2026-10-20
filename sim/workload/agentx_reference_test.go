package workload

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
)

// aiperfReference is the dump gen_aiperf_reference.py writes by running aiperf 0.12.0's
// own loader and TrajectorySource over a corpus.
type aiperfReference struct {
	Pool          []string `json:"pool"`
	Conversations []struct {
		SessionID string `json:"session_id"`
		IsRoot    bool   `json:"is_root"`
		Branches  []struct {
			ID         string   `json:"id"`
			Children   []string `json:"children"`
			Background bool     `json:"background"`
			StartMs    float64  `json:"start_ms"`
		} `json:"branches"`
		Turns []struct {
			Ts        float64  `json:"ts"`
			Delay     *float64 `json:"delay"`
			API       *float64 `json:"api"`
			MaxTokens int      `json:"max_tokens"`
			BranchIDs []string `json:"branch_ids"`
			Prereqs   []string `json:"prereqs"`
			Preds     [][]any  `json:"preds"`
		} `json:"turns"`
	} `json:"conversations"`
	Lanes map[string]struct {
		Lanes []struct {
			Trace   string `json:"trace"`
			TStarMs string `json:"t_star_ms"`
			States  []struct {
				Conv     string  `json:"conv"`
				Next     int     `json:"next"`
				OffsetMs string  `json:"offset_ms"`
				Waiting  bool    `json:"waiting"`
				Join     *int    `json:"join"`
				Branch   *string `json:"branch"`
				Depth    int     `json:"depth"`
			} `json:"states"`
			Boundaries [][]any `json:"boundaries"`
		} `json:"lanes"`
		NextRecycle string `json:"next_recycle"`
	} `json:"lanes"`
}

func loadAiperfReference(t *testing.T, path string) *aiperfReference {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var r aiperfReference
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return &r
}

// compareTreesToAiperf checks every conversation, turn, branch, join and replay barrier of
// the rebuilt corpus against aiperf's own reconstruction, field by field.
func compareTreesToAiperf(t *testing.T, c *AgentXCorpus, ref *aiperfReference) {
	t.Helper()
	if len(c.Traces) != len(ref.Pool) {
		t.Fatalf("pool size %d, aiperf %d", len(c.Traces), len(ref.Pool))
	}
	type convRef struct {
		tr *AgentXTrace
		ci int32
	}
	byID := map[string]convRef{}
	for i, tr := range c.Traces {
		if tr.ID != ref.Pool[i] {
			t.Fatalf("pool[%d] = %s, aiperf %s", i, tr.ID, ref.Pool[i])
		}
		for ci := range tr.convs {
			byID[tr.convs[ci].sessionID] = convRef{tr, int32(ci)}
		}
	}
	if len(byID) != len(ref.Conversations) {
		t.Errorf("conversation count %d, aiperf %d", len(byID), len(ref.Conversations))
	}
	bad := 0
	fail := func(format string, a ...any) {
		bad++
		if bad <= 30 {
			t.Errorf(format, a...)
		}
	}
	for _, rc := range ref.Conversations {
		cr, ok := byID[rc.SessionID]
		if !ok {
			fail("missing conversation %s", rc.SessionID)
			continue
		}
		conv := &cr.tr.convs[cr.ci]
		if len(conv.turns) != len(rc.Turns) {
			fail("%s: %d turns, aiperf %d", rc.SessionID, len(conv.turns), len(rc.Turns))
			continue
		}
		if cr.ci == 0 {
			if len(cr.tr.branches) != len(rc.Branches) {
				fail("%s: %d branches, aiperf %d", rc.SessionID, len(cr.tr.branches), len(rc.Branches))
			} else {
				for bi, rb := range rc.Branches {
					b := cr.tr.branches[bi]
					var kids []string
					for _, k := range b.children {
						kids = append(kids, cr.tr.convs[k].sessionID)
					}
					if b.id != rb.ID || b.background != rb.Background || b.startMs != rb.StartMs || fmt.Sprint(kids) != fmt.Sprint(rb.Children) {
						fail("%s branch %d: got %s %v %v %v, aiperf %s %v %v %v", rc.SessionID, bi, b.id, b.background, b.startMs, kids, rb.ID, rb.Background, rb.StartMs, rb.Children)
					}
				}
			}
		}
		for k, rt := range rc.Turns {
			tn := conv.turns[k]
			if tn.tsMs != rt.Ts || int(tn.maxTokens) != rt.MaxTokens {
				fail("%s turn %d: ts %v max %d, aiperf %v %d", rc.SessionID, k, tn.tsMs, tn.maxTokens, rt.Ts, rt.MaxTokens)
			}
			if (rt.Delay != nil) != tn.hasDelay || (rt.Delay != nil && *rt.Delay != tn.delayMs) {
				fail("%s turn %d: delay %v/%v, aiperf %v", rc.SessionID, k, tn.delayMs, tn.hasDelay, derefF(rt.Delay))
			}
			api := 0.0
			if rt.API != nil {
				api = *rt.API
			}
			if math.Abs(api-tn.apiMs) > 0 {
				fail("%s turn %d: api %v, aiperf %v", rc.SessionID, k, tn.apiMs, api)
			}
			var spawns, gates []string
			for _, b := range tn.spawns {
				spawns = append(spawns, cr.tr.branches[b].id)
			}
			for _, b := range tn.gates {
				gates = append(gates, cr.tr.branches[b].id)
			}
			if fmt.Sprint(spawns) != fmt.Sprint(rt.BranchIDs) || fmt.Sprint(gates) != fmt.Sprint(rt.Prereqs) {
				fail("%s turn %d: spawns %v gates %v, aiperf %v %v", rc.SessionID, k, spawns, gates, rt.BranchIDs, rt.Prereqs)
			}
			var preds [][]any
			for _, p := range tn.preds {
				preds = append(preds, []any{cr.tr.convs[p.conv].sessionID, float64(p.turn)})
			}
			if fmt.Sprint(preds) != fmt.Sprint(rt.Preds) {
				fail("%s turn %d: preds %v, aiperf %v", rc.SessionID, k, preds, rt.Preds)
			}
		}
	}
	if bad > 0 {
		t.Errorf("%d mismatches against aiperf", bad)
	}
}

// TestAgentXTreesMatchAiperfOnSlice: the committed six-trace slice (subagents, flat worker
// chains, background branches) rebuilds exactly as aiperf 0.12.0 rebuilds it.
func TestAgentXTreesMatchAiperfOnSlice(t *testing.T) {
	c, err := LoadAgentXCorpus("testdata/agentx/weka_v7_slice.jsonl", "", 1048576, true)
	if err != nil {
		t.Fatal(err)
	}
	ref := loadAiperfReference(t, "testdata/agentx/aiperf_reference_slice.json")
	compareTreesToAiperf(t, c, ref)
	compareLanesToAiperf(t, c, ref)
}

// TestAgentXTreesMatchAiperfOnSynthetic: the hand-built shapes the real corpus does not
// exercise (gen_synthetic_corpus.py) rebuild exactly as aiperf rebuilds them.
func TestAgentXTreesMatchAiperfOnSynthetic(t *testing.T) {
	c, err := LoadAgentXCorpus("testdata/agentx/weka_v7_synthetic.jsonl", "", 1048576, true)
	if err != nil {
		t.Fatal(err)
	}
	ref := loadAiperfReference(t, "testdata/agentx/aiperf_reference_synthetic.json")
	compareTreesToAiperf(t, c, ref)
	compareLanesToAiperf(t, c, ref)
}

// TestAgentXTreesMatchAiperfOnFullCorpus runs the same comparison on the full corpus when
// AGENTX_CORPUS and AGENTX_AIPERF_REFERENCE name the corpus and a gen_aiperf_reference.py
// dump of it (both too large to commit).
func TestAgentXTreesMatchAiperfOnFullCorpus(t *testing.T) {
	corpus, refPath := os.Getenv("AGENTX_CORPUS"), os.Getenv("AGENTX_AIPERF_REFERENCE")
	if corpus == "" || refPath == "" {
		t.Skip("set AGENTX_CORPUS and AGENTX_AIPERF_REFERENCE to compare the full corpus")
	}
	c, err := LoadAgentXCorpus(corpus, "", 1048576, true)
	if err != nil {
		t.Fatal(err)
	}
	ref := loadAiperfReference(t, refPath)
	compareTreesToAiperf(t, c, ref)
	compareLanesToAiperf(t, c, ref)
}

func derefF(p *float64) any {
	if p == nil {
		return nil
	}
	return *p
}

// compareLanesToAiperf checks, for every concurrency the reference dumped, each lane's
// trace, t* (bit for bit), live stream states and the next recycled root.
func compareLanesToAiperf(t *testing.T, c *AgentXCorpus, ref *aiperfReference) {
	t.Helper()
	if len(ref.Lanes) == 0 {
		t.Fatal("reference has no lanes")
	}
	for cs, rl := range ref.Lanes {
		conc, err := strconv.Atoi(cs)
		if err != nil {
			t.Fatal(err)
		}
		s := &agxSampler{pool: c.Traces}
		lanes, _ := agxBuildLanes(s, conc, 42, 0, 1)
		if len(lanes) != len(rl.Lanes) {
			t.Fatalf("C=%d: %d lanes, aiperf %d", conc, len(lanes), len(rl.Lanes))
		}
		for i, l := range lanes {
			r := rl.Lanes[i]
			want, _ := strconvParseHex(r.TStarMs)
			if l.trace.ID != r.Trace || l.tStarMs != want {
				t.Errorf("C=%d lane %d: %s t*=%x, aiperf %s %s", conc, i, l.trace.ID, l.tStarMs, r.Trace, r.TStarMs)
				continue
			}
			if len(l.states) != len(r.States) {
				t.Errorf("C=%d lane %d: %d states, aiperf %d", conc, i, len(l.states), len(r.States))
				continue
			}
			for j, st := range l.states {
				rs := r.States[j]
				off, _ := strconvParseHex(rs.OffsetMs)
				join := -1
				if rs.Join != nil {
					join = *rs.Join
				}
				branch := ""
				if st.branch >= 0 {
					branch = l.trace.branches[st.branch].id
				}
				rb := ""
				if rs.Branch != nil {
					rb = *rs.Branch
				}
				if l.trace.convs[st.conv].sessionID != rs.Conv || int(st.next) != rs.Next || st.offsetMs != off ||
					st.waiting != rs.Waiting || int(st.joinTurn) != join || branch != rb || (st.conv > 0) != (rs.Depth > 0) {
					t.Errorf("C=%d lane %d state %d: %+v, aiperf %+v", conc, i, j, st, rs)
				}
			}
			if len(l.completed) != len(r.Boundaries) {
				t.Errorf("C=%d lane %d: %d resume boundaries, aiperf %d", conc, i, len(l.completed), len(r.Boundaries))
			}
		}
		if nr := s.nextRecycle(); nr.ID != rl.NextRecycle {
			t.Errorf("C=%d: next recycle %s, aiperf %s", conc, nr.ID, rl.NextRecycle)
		}
	}
}

func strconvParseHex(s string) (float64, error) { return strconv.ParseFloat(s, 64) }
