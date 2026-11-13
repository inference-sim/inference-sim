// Original to blis-vllm.

package kernelmodel

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// DeepSeek-V4.1 at vLLM v0.32.0 runs the layers after the last KV layer on each request's last 128
// scheduled tokens in steps with no graph (decoder_replay.go). These tests read the price the
// pool's batch gets, so the rows the engine states show up only as time.

const replayScenario = `kind: Scenario
name: replay
engine_version: "%s"
model: deepseek-v4.1-flash
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory]
cluster:
  hardware: h200
  nodes: 1
  gpus_per_node: 8
---
kind: Deployment
name: replay
pools:
  - role: colocated
    nodes: 1
    parallel:
      tp: %d
      pp: 1
      dp: %d
      enable_expert_parallel: false
    engine:
      block_size: 64
      kv_cache_dtype: fp8
      max_num_seqs: 256
      max_num_batched_tokens: 8192
      max_model_len: 1048576
      gpu_memory_utilization: 0.8
      engram_config:
        cpu_offload: true
      speculative_config: {method: dspark, num_speculative_tokens: 5}
%s`

// openReplay opens DeepSeek-V4.1-Flash on H200 at tensor parallel 8 under the given release, with
// extra lines in its engine block.
func openReplay(t testing.TB, release, extra string) (*Model, error) {
	t.Helper()
	return openReplayAt(t, release, 8, 1, extra)
}

func openReplayAt(t testing.TB, release string, tp, dp int, extra string) (*Model, error) {
	t.Helper()
	dir := t.TempDir()
	body := fmt.Sprintf(replayScenario, release, tp, dp, extra)
	if err := os.WriteFile(filepath.Join(dir, "replay.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	r := repos()
	r.Scenarios = dir
	return OpenPool("replay.yaml", r, 0)
}

func mustReplay(t testing.TB, release, extra string) *Model {
	t.Helper()
	m, err := openReplay(t, release, extra)
	if err != nil {
		t.Fatalf("%s %q: %v", release, extra, err)
	}
	return m
}

func replayChunk(scheduled, computed int) ReqShape {
	return ReqShape{Scheduled: scheduled, Computed: computed, PromptLen: scheduled + computed}
}

func replayDecodes(n int) []ReqShape {
	out := make([]ReqShape, n)
	for i := range out {
		// Under DSpark at K=5 a decode verifies six tokens.
		out[i] = ReqShape{Scheduled: 6, Computed: 100_000 + i, PromptLen: 90_000}
	}
	return out
}

// replayOff is the same pool with bounded replay switched off, which is the pool priced as vLLM 0.31.0
// ran it: every layer on every scheduled token.
const replayOff = "      swa_bounded_replay: false\n"

// A prefill chunk is cheaper than the same chunk on the pool that runs every layer on every token,
// by the layers after the last KV layer on the rows they do not run; a pool on the earlier release
// prices it as it always did.
func TestAPrefillChunkIsPricedOnTheTrailingRowsOfTheLayersAfterTheLastKVLayer(t *testing.T) {
	trimmed, full := mustReplay(t, "0.32.0", ""), mustReplay(t, "0.32.0", replayOff)
	before := mustReplay(t, "0.31.0", "")
	big := []ReqShape{replayChunk(8192, 0)}
	a, b := trimmed.StepTimeOfShapes(big), full.StepTimeOfShapes(big)
	if a >= b || float64(a) > 0.85*float64(b) {
		t.Errorf("an 8192-token chunk prices %d us with the decoder replay and %d us without; the 19 later layers are half the stack", a, b)
	}
	if v := before.StepTimeOfShapes(big); v != b {
		t.Errorf("the 0.31.0 pool prices the chunk at %d us, the 0.32.0 pool with replay off %d us; neither runs the replay", v, b)
	}
}

// The layers after the last KV layer cost 128 rows a request, not the chunk: the saving grows with
// the chunk, and a long context behind the chunk costs the kept rows far less than it costs the
// layers run on every row.
func TestTheSavingGrowsWithTheChunkAndTheContextBehindIt(t *testing.T) {
	trimmed, full := mustReplay(t, "0.32.0", ""), mustReplay(t, "0.32.0", replayOff)
	price := func(m *Model, c ReqShape) int64 { return m.StepTimeOfShapes([]ReqShape{c}) }
	saved := func(c ReqShape) float64 { return float64(price(full, c) - price(trimmed, c)) }
	s2k, s8k := saved(replayChunk(2048, 0)), saved(replayChunk(8192, 0))
	if s2k <= 0 || s8k/s2k < 2.5 || s8k/s2k > 5 {
		t.Errorf("the saving is %.0f us for 2048 tokens and %.0f us for 8192; it should grow about 4x with the rows skipped", s2k, s8k)
	}
	// Behind a 300 thousand token prefix every row the full pool runs scores that context; the kept
	// rows score it for 128 rows only, so the context adds much less to the trimmed price.
	near, far := replayChunk(8192, 0), replayChunk(8192, 300_000)
	addedTrimmed := price(trimmed, far) - price(trimmed, near)
	addedFull := price(full, far) - price(full, near)
	if addedTrimmed <= 0 || float64(addedTrimmed) > 0.8*float64(addedFull) {
		t.Errorf("a 300000-token context adds %d us to the trimmed chunk and %d us to the full one", addedTrimmed, addedFull)
	}
}

// A request that has a cached prefix schedules the hit's 128-token rewind and the tokens after it,
// and keeps the last 128 of those; each request in a step keeps its own.
func TestACachedPrefixRequestKeepsItsOwnLastRows(t *testing.T) {
	trimmed, full := mustReplay(t, "0.32.0", ""), mustReplay(t, "0.32.0", replayOff)
	hit := replayChunk(128+3000, 300_000-128)
	small := replayChunk(90, 0)
	step := []ReqShape{hit, small}
	if a, b := trimmed.StepTimeOfShapes(step), full.StepTimeOfShapes(step); a >= b {
		t.Errorf("a step of a cached-prefix chunk and a 90-token request prices %d us with the replay and %d us without", a, b)
	}
	// The 90-token request keeps all 90 rows: a step of it alone is not changed.
	if a, b := trimmed.StepTimeOfShapes([]ReqShape{small}), full.StepTimeOfShapes([]ReqShape{small}); a != b {
		t.Errorf("a 90-token request alone prices %d us with the replay and %d us without; it trims no row", a, b)
	}
	// Two requests of 4096 keep 256 rows between them where one of 8192 keeps 128: the same
	// scheduled tokens, and the replay saves less on the two.
	saved := func(step []ReqShape) int64 { return full.StepTimeOfShapes(step) - trimmed.StepTimeOfShapes(step) }
	two, one := []ReqShape{replayChunk(4096, 0), replayChunk(4096, 0)}, []ReqShape{replayChunk(8192, 0)}
	if saved(two) >= saved(one) {
		t.Errorf("two 4096-token requests save %d us, one 8192-token request %d us; the two keep twice the rows", saved(two), saved(one))
	}
}

// A decode step keeps every row of every request, so the replay changes nothing for it. A small
// chunk that a piecewise graph covers, below the trim threshold, keeps the whole batch too.
func TestADecodeStepAndASmallGraphedStepAreNotTrimmed(t *testing.T) {
	trimmed, full := mustReplay(t, "0.32.0", ""), mustReplay(t, "0.32.0", replayOff)
	for name, step := range map[string][]ReqShape{
		"sixteen decodes":                   replayDecodes(16),
		"a 300-token chunk (graphed)":       {replayChunk(300, 0)},
		"a chunk and decodes under the cap": append(replayDecodes(8), replayChunk(400, 0)),
	} {
		if a, b := trimmed.StepTimeOfShapes(step), full.StepTimeOfShapes(step); a != b {
			t.Errorf("%s: %d us with the replay, %d us without", name, a, b)
		}
	}
	// The same decodes beside a chunk that makes the step eager do trim the chunk, not the decodes.
	mixed := append(replayDecodes(8), replayChunk(4096, 0))
	if a, b := trimmed.StepTimeOfShapes(mixed), full.StepTimeOfShapes(mixed); a >= b {
		t.Errorf("a 4096-token chunk with eight decodes prices %d us with the replay and %d us without", a, b)
	}
}

// decoder_replay_cudagraph_capture_sizes are the sizes the trimmed batch is padded to: a pool with
// one large size launches the 128 kept rows at that size, which costs more than the default
// sizes' 128, and a size above max_num_batched_tokens is refused as vLLM refuses it.
func TestTheStatedReplayGraphSizesPadTheTrimmedBatch(t *testing.T) {
	def := mustReplay(t, "0.32.0", "")
	padded := mustReplay(t, "0.32.0", "      compilation_config: {decoder_replay_cudagraph_capture_sizes: [4096]}\n")
	step := []ReqShape{replayChunk(8192, 0)}
	if a, b := padded.StepTimeOfShapes(step), def.StepTimeOfShapes(step); a <= b {
		t.Errorf("rows padded to a 4096-row graph price %d us, to the default 128 %d us", a, b)
	}
	// Rows beyond the largest size run eager, at their own size: a size no batch fits prices as
	// the default sizes' largest does not.
	small := mustReplay(t, "0.32.0", "      compilation_config: {decoder_replay_cudagraph_capture_sizes: [64]}\n")
	if a, b := small.StepTimeOfShapes(step), def.StepTimeOfShapes(step); a != b {
		t.Errorf("128 kept rows with only a 64-row graph price %d us, in a 128-row graph %d us; both launch 128 rows", a, b)
	}
	_, err := openReplay(t, "0.32.0", "      compilation_config: {decoder_replay_cudagraph_capture_sizes: [16384]}\n")
	if err == nil || !strings.Contains(err.Error(), "decoder_replay_cudagraph_capture_sizes") {
		t.Errorf("a size above max_num_batched_tokens: %v", err)
	}
}

// What stops the replay is disclosed, with the reason, and prices every layer on every token: a
// data-parallel pool, whose replay batch is agreed across ranks, and a pool that turns SWA bounded
// replay off.
func TestAPoolThatCannotReplayDisclosesWhy(t *testing.T) {
	on := mustReplay(t, "0.32.0", "")
	if got := strings.Join(on.EngineAssumptions(), "\n"); !strings.Contains(got, "decoder_replay = on") {
		t.Errorf("a pool that replays does not say so: %s", got)
	}
	no := mustReplay(t, "0.32.0", replayOff)
	if got := strings.Join(no.EngineAssumptions(), "\n"); !strings.Contains(got, "decoder_replay = off: SWA bounded replay is off") {
		t.Errorf("a pool with bounded replay off does not say why: %s", got)
	}
	early := mustReplay(t, "0.31.0", "")
	if got := strings.Join(early.EngineAssumptions(), "\n"); strings.Contains(got, "decoder_replay") {
		t.Errorf("a 0.31.0 pool has no decoder replay to disclose: %s", got)
	}
	dp, err := openReplayAt(t, "0.32.0", 4, 2, "")
	if err != nil {
		t.Skipf("a data-parallel V4.1 pool does not open here: %v", err)
	}
	if got := strings.Join(dp.EngineAssumptions(), "\n"); !strings.Contains(got, "decoder_replay = off: data parallelism") {
		t.Errorf("a data-parallel pool does not say its replay is not priced: %s", got)
	}
}

// Pricing a step leaves the caller's requests as they were: the model states the rows on a batch of
// its own.
func TestPricingAStepDoesNotChangeTheCallersRequests(t *testing.T) {
	m := mustReplay(t, "0.32.0", "")
	step := []ReqShape{replayChunk(4096, 0), replayChunk(60, 0)}
	want := append([]ReqShape(nil), step...)
	first := m.StepTimeOfShapes(step)
	for i := range step {
		if step[i] != want[i] {
			t.Fatalf("request %d became %+v", i, step[i])
		}
	}
	if again := m.StepTimeOfShapes(step); again != first {
		t.Errorf("the same step priced %d us, then %d us", first, again)
	}
}

// A step a piecewise graph covers trims when the graph it is padded to has decoder_replay_trim_threshold
// (768) tokens or more, not when the step itself does: with captures up to 2048, a 752-token chunk
// replays the 752 graph and keeps the whole batch, and one more token replays the 768 graph and trims
// (model_state.py:455-460).
func TestAPiecewiseStepTrimsByTheSizeOfTheGraphItIsPaddedTo(t *testing.T) {
	trimmed := mustReplay(t, "0.32.0", "      max_cudagraph_capture_size: 2048\n")
	full := mustReplay(t, "0.32.0", "      max_cudagraph_capture_size: 2048\n"+replayOff)
	for _, c := range []struct {
		tokens int
		trims  bool
	}{{600, false}, {752, false}, {753, true}, {1000, true}, {2048, true}} {
		step := []ReqShape{replayChunk(c.tokens, 0)}
		a, b := trimmed.StepTimeOfShapes(step), full.StepTimeOfShapes(step)
		if c.trims != (a < b) || (!c.trims && a != b) {
			t.Errorf("a %d-token chunk prices %d us with the replay and %d us without; trims %v, want %v", c.tokens, a, b, a < b, c.trims)
		}
	}
	// Past the largest graph the step is eager and trims.
	step := []ReqShape{replayChunk(4096, 0)}
	if a, b := trimmed.StepTimeOfShapes(step), full.StepTimeOfShapes(step); a >= b {
		t.Errorf("a 4096-token chunk past the 2048 graph prices %d us with the replay and %d us without", a, b)
	}
}
