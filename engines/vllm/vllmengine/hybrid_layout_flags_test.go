// Original to blis-vllm.

package vllmengine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// withEngineLines is the scenario with extra lines added to the head of every pool's engine
// block, as a user adds a flag to a deployment file.
func withEngineLines(t *testing.T, scenario, lines string) map[string]string {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", scenario))
	if err != nil {
		t.Fatal(err)
	}
	const anchor = "    engine:\n"
	if !strings.Contains(string(body), anchor) {
		t.Fatalf("%s has no engine block", scenario)
	}
	path := filepath.Join(t.TempDir(), scenario)
	if err := os.WriteFile(path, []byte(strings.ReplaceAll(string(body), anchor, anchor+lines)), 0o644); err != nil {
		t.Fatal(err)
	}
	r := roots(t, scenario)
	r[RootScenario] = path
	return r
}

func newEngine(r map[string]string) (engine.Engine, error) {
	Register()
	return engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: r})
}

// engineKVBlocks is the engine's KV pool in blocks, the null block included.
func engineKVBlocks(t *testing.T, r map[string]string) int64 {
	t.Helper()
	e, err := newEngine(r)
	if err != nil {
		t.Fatal(err)
	}
	return e.Snapshot().KVBlocksTotal + 1
}

// secondAdmission sends one prompt twice, the second time after the first has finished, and
// returns the second request's first batch share.
func secondAdmission(t *testing.T, r map[string]string, prompt int) engine.Scheduled {
	t.Helper()
	a, b := req("a", prompt, 4), req("b", prompt, 4)
	copy(b.Prompt, a.Prompt)
	tr, err := enginetest.Drive(func() (engine.Engine, error) { return newEngine(r) },
		[]enginetest.Arrival{{At: 0, Req: a}, {At: 10_000_000, Req: b}}, enginetest.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.Finished) != 2 {
		t.Fatalf("finished %d of 2 requests", len(tr.Finished))
	}
	for _, s := range tr.Steps {
		for _, sc := range s.Batch.Reqs {
			if sc.ID == "b" {
				return sc
			}
		}
	}
	t.Fatal("request b never ran")
	return engine.Scheduled{}
}

// swa_bounded_replay: false on DeepSeek-V4.1 under the V2 runner (DSpark needs it) leaves the
// sliding-window caches prefix-cacheable with no replay window (SlidingWindowMLASpec.prefix_cacheable
// and prefix_replay_tokens, vllm/v1/kv_cache_interface.py:968-974@5df63a72). The default, bounded
// replay, recomputes the last window (128 tokens) of every hit: a repeated 1000-token prompt
// hits 960 tokens either way, but the bounded engine computes the 128 before the hit again and
// the unbounded one only the 40 past it. The pool holds the same blocks (only the sliding-window
// groups' prefix caching changes, not their pages). The prefix-hit and layout rules are those the
// tier-2 scenarios dashboard-deepseek-v4.1-flash-layers (unbounded) and -replay (bounded) pin.
func TestSWABoundedReplayOffRunsTheUnboundedLayout(t *testing.T) {
	bounded := withEngineLines(t, "d034-032.yaml", "")
	off := withEngineLines(t, "d034-032.yaml", "      swa_bounded_replay: false\n")
	on, got := secondAdmission(t, bounded, 1000), secondAdmission(t, off, 1000)
	if on.CachedTokens != 960 || on.Computed != 832 || on.Tokens != 168 {
		t.Errorf("default (bounded replay): %+v, want a hit of 960 replayed from 832 (168 tokens run)", on)
	}
	if got.CachedTokens != 960 || got.Computed != 960 || got.Tokens != 40 {
		t.Errorf("swa_bounded_replay false: %+v, want a hit of 960 with nothing replayed (40 tokens run)", got)
	}
	if a, b := engineKVBlocks(t, bounded), engineKVBlocks(t, off); a != b {
		t.Errorf("swa_bounded_replay false changed the pool from %d to %d blocks", a, b)
	}
}

// disable_hybrid_kv_cache_manager: true unifies a hybrid model's layers to one full-attention
// group (unify_hybrid_kv_cache_specs, vllm/v1/core/kv_cache_utils.py:1799-1822@5df63a72), whose
// block holds every layer's page: gpt-oss-120b's 36 layers instead of the 18 of a hybrid group,
// so the pool holds half the blocks; the 62 layers of Muse-Spark (16 full, 46 windows) instead of
// the 16 of its widest group. Prefix hits of a model whose windows were promoted are full
// attention's.
func TestDisablingTheHybridManagerUnifiesTheKVGroups(t *testing.T) {
	for _, c := range []struct {
		scenario      string
		layers, group int64
	}{
		{"d036-032.yaml", 36, 18},            // gpt-oss-120b: two groups of 18
		{"muse-gb200-nvl-tep8.yaml", 62, 16}, // Muse-Spark: a group of 16 full, three of 16 windows
	} {
		t.Run(c.scenario, func(t *testing.T) {
			hybrid := engineKVBlocks(t, withEngineLines(t, c.scenario, ""))
			unified := engineKVBlocks(t, withEngineLines(t, c.scenario, "      disable_hybrid_kv_cache_manager: true\n"))
			// blocks = available / (layers a block holds pages of), with the null block counted.
			if want := hybrid * c.group / c.layers; unified < want-1 || unified > want+1 {
				t.Errorf("unified pool %d blocks, hybrid %d: want about %d (a block spans %d layers, not %d)",
					unified, hybrid, want, c.layers, c.group)
			}
		})
	}
	// A repeated prompt hits as full attention does, and the engine runs it.
	off := withEngineLines(t, "d036-032.yaml", "      disable_hybrid_kv_cache_manager: true\n")
	if sc := secondAdmission(t, off, 1000); sc.CachedTokens != 992 {
		t.Errorf("gpt-oss-120b with the hybrid manager off: %+v, want a full-attention hit of 992 tokens (62 blocks of 16)", sc)
	}
}

// vLLM v0.32.0 does not start a DeepSeek V4 or V4.1 engine with the hybrid manager off: the
// promotion of the sliding-window MLA specs raises TypeError, so the factory refuses the
// deployment with that reason and says what to do, and a user's flag is neither dropped nor
// priced as a layout vLLM never builds. Without the flag the same pools build.
func TestDisablingTheHybridManagerOnDeepSeekIsVLLMsOwnStartupFailure(t *testing.T) {
	for _, scenario := range []string{"d034-032.yaml", "d010-032.yaml"} {
		t.Run(scenario, func(t *testing.T) {
			if _, err := newEngine(withEngineLines(t, scenario, "")); err != nil {
				t.Fatalf("without the flag: %v", err)
			}
			_, err := newEngine(withEngineLines(t, scenario, "      disable_hybrid_kv_cache_manager: true\n"))
			if err == nil {
				t.Fatal("built, want vLLM's start-up failure")
			}
			for _, frag := range []string{"vLLM does not start", "TypeError", "bounded_replay", "drop disable_hybrid_kv_cache_manager"} {
				if !strings.Contains(err.Error(), frag) {
					t.Errorf("the refusal lacks %q: %v", frag, err)
				}
			}
		})
	}
}

// A P/D pair of Muse-Spark pools with the hybrid manager off runs through NIXL as one
// full-attention group: the prefill hands off its 8192-token prompt, the decoder pulls all of it
// and emits (vLLM keeps the hybrid manager off when a user states it, and NIXL then needs no HMA,
// nixl/base_scheduler.py:95-100@5df63a72).
func TestAPDPairWithTheHybridManagerOffRuns(t *testing.T) {
	r := withEngineLines(t, "muse-h100-roce-1p1d.yaml", "      disable_hybrid_kv_cache_manager: true\n")
	var engines [2]engine.Engine
	for i := range engines {
		spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: r, Pool: i}
		var err error
		if engines[i], err = engine.New(spec); err != nil {
			t.Fatal(err)
		}
	}
	p, d := engines[0], engines[1]
	r0 := req("a", 8192, 32)
	pre := r0
	pre.RemoteDecode, pre.MaxOutputTokens = true, 1
	if err := p.Add(pre, 0); err != nil {
		t.Fatal(err)
	}
	now := engine.Time(0)
	var held []engine.Handoff
	for len(held) == 0 {
		b, err := p.Schedule(now)
		if err != nil || b.Empty() {
			t.Fatalf("prefill schedule: %v, empty %v", err, b.Empty())
		}
		now += p.StepTime(b)
		res, err := p.Complete(b.ID, engine.Outcome{}, now)
		if err != nil {
			t.Fatal(err)
		}
		held = res.Handoffs
	}
	dec := r0
	at := plan.Endpoint{Node: 0, Rack: 0}
	dec.RemotePrefill, dec.Peer = true, &at
	if err := d.Add(dec, now); err != nil {
		t.Fatal(err)
	}
	b, err := d.Schedule(now)
	if err != nil || len(b.TransfersStarted) != 1 || b.TransfersStarted[0].Tokens != 8192 {
		t.Fatalf("decode schedule: %v, transfers %+v, want one pull of 8192 tokens", err, b.TransfersStarted)
	}
	now += 5000
	if err := d.TransferDone(b.TransfersStarted[0].ID, true, now); err != nil {
		t.Fatal(err)
	}
	if err := p.TransferDone(held[0].Transfer, true, now); err != nil {
		t.Fatal(err)
	}
	emitted := 0
	for steps := 0; emitted < 32 && steps < 200; steps++ {
		b, err := d.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		if b.Empty() {
			now += 1000
			continue
		}
		now += d.StepTime(b)
		res, err := d.Complete(b.ID, engine.Outcome{}, now)
		if err != nil {
			t.Fatal(err)
		}
		emitted += res.Emitted["a"]
	}
	if emitted != 32 {
		t.Fatalf("the decoder emitted %d of 32 tokens", emitted)
	}
}
