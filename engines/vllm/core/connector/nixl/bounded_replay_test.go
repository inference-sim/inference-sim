// Original to blis-vllm.

package nixl

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-vllm/core/kv"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// A decoder beside DeepSeek V4.1's KV cache layout: eight-odd sliding-window groups of 32-token
// blocks whose 128-token window is kept out of prefix caching and replayed after a hit
// (bounded replay), compressed caches of 128-token blocks, and rings. The layers are the ones the
// scheduler oracle runs (oracle/scenarios/dashboard-deepseek-v4.1-flash-replay.json).

const flashBlock = 128

// storeLoad is a connector that moves blocks as a prefix-cache store does: it restores no
// bounded-replay window, which vLLM's default get_loaded_kv_cache_group_ids states
// (kv_connector/v1/base.py:490-500@5df63a72).
type storeLoad struct{ *Scheduler }

func (storeLoad) LoadedGroups(_ *sched.Request, _ []int) []int { return nil }

func flashLayers(t *testing.T) []kvLayerRun {
	t.Helper()
	var sc struct {
		Config struct {
			KVLayers []kvLayerRun `json:"kv_layers"`
		} `json:"config"`
	}
	readJSON(t, "../../../oracle/scenarios/dashboard-deepseek-v4.1-flash-replay.json", &sc)
	return sc.Config.KVLayers
}

// flashSched is a scheduler over the V4.1 flash layers with a NIXL connector of the given role,
// built as core/vllm builds an engine's (Backoff, Groups).
func flashSched(t *testing.T, consumer bool, wrap func(*Scheduler) sched.Connector) (*sched.Scheduler, *Scheduler) {
	t.Helper()
	groups, err := kv.GroupLayers(scenarioLayers(t, flashLayers(t)), kv.GroupOptions{})
	if err != nil {
		t.Fatal(err)
	}
	s, err := sched.New(sched.Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 512, BlockSize: flashBlock, NumBlocks: 400,
		MaxModelLen: 1024, EnableChunkedPrefill: true, Policy: sched.FCFS, EnablePrefixCaching: true,
		ReserveFullSequence: true, KVGroups: groups})
	if err != nil {
		t.Fatal(err)
	}
	cfg := Default()
	cfg.PrefillBackoff = Backoff(sched.Spec{}, s.KV.PrefixReplayTokens() > 0)
	for g := 0; g < s.KV.NumGroups(); g++ {
		grp := Group{BlockSize: s.KV.GroupBlockSize(g)}
		if gs := s.KV.GroupSpec(g); gs.Kind == kv.SlidingWindow || gs.Kind == kv.SlidingWindowMLA {
			grp.Window = gs.Window
		}
		cfg.Groups = append(cfg.Groups, grp)
	}
	conn := New(cfg)
	var c sched.Connector = conn
	if wrap != nil {
		c = wrap(conn)
	}
	if err := s.SetConnector(c, sched.ConnectorConfig{Consumer: consumer, Producer: !consumer}); err != nil {
		t.Fatal(err)
	}
	return s, conn
}

func promptOf(n int) []int64 {
	p := make([]int64, n)
	for i := range p {
		p[i] = int64(1000 + i)
	}
	return p
}

// pullThrough admits a remote-prefilled request of the given prompt length on the decoder, reports
// its pull landed, and returns what the decoder scheduled at admission and after the landing.
func pullThrough(t *testing.T, s *sched.Scheduler, conn *Scheduler, prompt int) (admit, after sched.Output, r *sched.Request) {
	t.Helper()
	r = &sched.Request{ID: "d", Prompt: promptOf(prompt), MaxTokens: 4, EOS: eosToken,
		Transfer: &sched.TransferParams{DoRemotePrefill: true, HasRemote: true, RemoteBlocks: 1}}
	if err := s.Add(r); err != nil {
		t.Fatal(err)
	}
	admit = s.Schedule()
	conn.BuildMeta()
	if got := s.WaitingForRemoteKV(); !reflect.DeepEqual(got, []string{"d"}) {
		t.Fatalf("waiting for remote KV %v, want [d]", got)
	}
	if len(admit.Scheduled) != 0 {
		t.Fatalf("the decoder computed %v while the pull was in flight", admit.Scheduled)
	}
	s.UpdateWith(admit, nil, sched.TransferOutcome{FinishedRecving: []string{"d"}})
	after = s.Schedule()
	return admit, after, r
}

// NIXL restores every group's blocks of the request's own prompt, the bounded-replay windows
// included, so the decoder replays nothing: it loads all the prompt but the one token the
// prefiller's backoff left (the prefiller does not compute the last token of a prompt under
// bounded replay, nixl/base_scheduler.py:379-394@5df63a72), and computes that token alone.
func TestDecoderBesideBoundedReplayLoadsThePromptAndReplaysNothing(t *testing.T) {
	for _, prompt := range []int{456, 300, 100} {
		s, conn := flashSched(t, true, nil)
		admit, after, r := pullThrough(t, s, conn, prompt)
		if got := admit; got.Scheduled == nil {
			t.Fatalf("prompt %d: no schedule output", prompt)
		}
		if got := after.Scheduled["d"]; got != 1 {
			t.Errorf("prompt %d: the decoder computed %d tokens after the load, want 1 (the prompt's last)", prompt, got)
		}
		if r.ReplayStart != 0 {
			t.Errorf("prompt %d: replay starts at %d, want none: the load carried the window", prompt, r.ReplayStart)
		}
	}
}

// A load that carries no window (what a prefix-cache store restores) is trimmed to a block
// boundary and its last window is recomputed: with a 455-token load that is 384 tokens kept, 128
// replayed, and the 72 beyond them computed.
func TestALoadWithoutTheWindowIsTrimmedAndReplayed(t *testing.T) {
	s, conn := flashSched(t, true, func(c *Scheduler) sched.Connector { return storeLoad{c} })
	_, after, r := pullThrough(t, s, conn, 456)
	if r.ReplayStart != 256 {
		t.Errorf("replay starts at %d, want 256 (the 384-token hit less its 128-token window)", r.ReplayStart)
	}
	if got := after.Scheduled["d"]; got != 456-256 {
		t.Errorf("the decoder computed %d tokens, want %d (the replayed window and the tail)", got, 456-256)
	}
}

// The prefiller computes one prompt token fewer than it was sent when any group replays, and the
// decoder pulls that many fewer: a model with no bounded-replay group keeps the whole prompt.
func TestPrefillBackoffFloorsAtOneBesideBoundedReplay(t *testing.T) {
	if got := Backoff(sched.Spec{}, true); got != 1 {
		t.Errorf("backoff beside a replay window is %d, want 1", got)
	}
	if got := Backoff(sched.Spec{}, false); got != 0 {
		t.Errorf("backoff with no replay window is %d, want 0", got)
	}
	if got := Backoff(sched.Spec{PrefillLookahead: 4}, true); got != 3 {
		t.Errorf("backoff with a 4-token lookahead is %d, want 3", got)
	}
	s, conn := flashSched(t, false, nil)
	r := &sched.Request{ID: "p", Prompt: promptOf(456), MaxTokens: 1, EOS: eosToken,
		Transfer: &sched.TransferParams{DoRemoteDecode: true}}
	if err := s.Add(r); err != nil {
		t.Fatal(err)
	}
	if len(r.Prompt) != 455 {
		t.Fatalf("the prefiller kept %d prompt tokens, want 455", len(r.Prompt))
	}
	out := s.Schedule()
	conn.BuildMeta()
	if got := out.Scheduled["p"]; got != 455 {
		t.Errorf("the prefiller computed %d tokens, want 455", got)
	}
}

// The blocks a pull reads and a finished prefill advertises are the connector's clipped lists:
// every block of a full-attention group, the last cdiv(window, block size) + 1 of a sliding window
// (the manager allocates for the whole sequence and frees the out-of-window blocks late), and the
// ring's one.
func TestTransferredBlocksFollowTheConnectorsClipRule(t *testing.T) {
	cfg := Config{Groups: []Group{{}, {Window: 128, BlockSize: 32}, {Window: 128, BlockSize: 32}}}
	cfg.Groups[0].BlockSize = 128
	ids := func(n int) []int {
		out := make([]int, n)
		for i := range out {
			out[i] = i + 1
		}
		return out
	}
	got := cfg.ClipBlocks([][]int{ids(9), ids(15), ids(3)})
	want := [][]int{ids(9), ids(15)[10:], ids(3)}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("clipped %v, want %v", got, want)
	}

	// On the V4.1 layers: a decoder's pull asks the worker for the unhashed blocks it allocated,
	// clipped, and the prefiller advertises the clipped blocks it holds.
	s, conn := flashSched(t, true, nil)
	pullThrough(t, s, conn, 456)
	counts, ok := conn.Pulled("d")
	if !ok || len(counts) != s.KV.NumGroups() {
		t.Fatalf("pulled %v (%v), want one count per KV cache group", counts, ok)
	}
	for g, n := range counts {
		spec := s.KV.GroupSpec(g)
		switch spec.Kind {
		case kv.SlidingWindowMLA:
			if limit := (spec.Window+s.KV.GroupBlockSize(g)-1)/s.KV.GroupBlockSize(g) + 1; n == 0 || n > limit {
				t.Errorf("group %d (window %d): %d blocks pulled, want 1 to %d", g, spec.Window, n, limit)
			}
		case kv.CircularBuffer:
			if n != 1 {
				t.Errorf("group %d (ring): %d blocks pulled, want 1", g, n)
			}
		default:
			if want := (455 + s.KV.GroupBlockSize(g) - 1) / s.KV.GroupBlockSize(g); n != want {
				t.Errorf("group %d (full): %d blocks pulled, want all %d", g, n, want)
			}
		}
	}

	p, pconn := flashSched(t, false, nil)
	r := &sched.Request{ID: "p", Prompt: promptOf(456), MaxTokens: 1, EOS: eosToken,
		Transfer: &sched.TransferParams{DoRemoteDecode: true}}
	if err := p.Add(r); err != nil {
		t.Fatal(err)
	}
	out := p.Schedule()
	pconn.BuildMeta()
	p.Update(out, map[string][]int64{"p": {eosToken}})
	sent, ok := pconn.Sent("p")
	if !ok || len(sent) != p.KV.NumGroups() {
		t.Fatalf("sent %v (%v), want one count per KV cache group", sent, ok)
	}
	for g, n := range sent {
		if spec := p.KV.GroupSpec(g); spec.Kind == kv.SlidingWindowMLA && n > (spec.Window+31)/32+1 {
			t.Errorf("group %d: %d window blocks advertised, want at most %d", g, n, (spec.Window+31)/32+1)
		}
	}
}

// A connector that does not state which groups its load restores, beside a bounded-replay
// window, is refused: the scheduler cannot tell whether the window is replayed.
func TestPackedGroupsNeedAConnectorThatStatesItsLoadedGroups(t *testing.T) {
	groups, err := kv.GroupLayers(scenarioLayers(t, flashLayers(t)), kv.GroupOptions{})
	if err != nil {
		t.Fatal(err)
	}
	s, err := sched.New(sched.Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 512, BlockSize: flashBlock, NumBlocks: 400,
		MaxModelLen: 1024, EnableChunkedPrefill: true, Policy: sched.FCFS, EnablePrefixCaching: true, KVGroups: groups})
	if err != nil {
		t.Fatal(err)
	}
	err = s.SetConnector(noLoadedGroups{New(Default())}, sched.ConnectorConfig{Consumer: true})
	if err == nil {
		t.Errorf("a connector stating no loaded groups beside a bounded-replay window: %v, want a refusal", err)
	}
}

// noLoadedGroups hides the NIXL scheduler's GroupLoader.
type noLoadedGroups struct{ s *Scheduler }

func (n noLoadedGroups) OnNewRequest(r *sched.Request) { n.s.OnNewRequest(r) }
func (n noLoadedGroups) MatchedTokens(r *sched.Request, c int) (int, bool) {
	return n.s.MatchedTokens(r, c)
}
func (n noLoadedGroups) AfterAlloc(r *sched.Request, b sched.AllocatedBlocks, e int) {
	n.s.AfterAlloc(r, b, e)
}
func (n noLoadedGroups) Finished(r *sched.Request, ids []int) bool { return n.s.Finished(r, ids) }
func (n noLoadedGroups) AfterOutput(o sched.TransferOutcome)       { n.s.AfterOutput(o) }
func (n noLoadedGroups) FinishedAllGroups(r *sched.Request, ids [][]int) bool {
	return n.s.FinishedAllGroups(r, ids)
}
func (n noLoadedGroups) ConnectorModels() sched.Capabilities { return n.s.ConnectorModels() }
