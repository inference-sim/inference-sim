// Original to blis-vllm.

package vllm

import (
	"bytes"
	"encoding/json"
	"math"
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"
)

// snapshotting records the engine's Snapshot after every Complete, at the step's end.
type snapshotting struct {
	engine.Engine
	at    []engine.Time
	state []engine.State
}

func (s *snapshotting) Complete(id engine.BatchID, o engine.Outcome, end engine.Time) (engine.Result, error) {
	r, err := s.Engine.Complete(id, o, end)
	s.at, s.state = append(s.at, end), append(s.state, s.Snapshot())
	return r, err
}

type seriesEngine interface {
	EnableServerSeries(intervalUs int64)
	ServerSeriesSpan() (interval, end int64)
	ServerSeriesBlock(id string, end int64) ([]byte, error)
}

// The server series bins what the engine's own metrics record: each counter's bins sum to its
// Metrics total and to the run's own tokens (prompts, outputs, prefix hits), each gauge's last
// value in a bin is the Snapshot after the bin's last step, and the block round-trips as JSON.
func TestServerSeriesBinsTheEnginesMetrics(t *testing.T) {
	const interval = 250
	var last *snapshotting
	newE := func() (engine.Engine, error) {
		e, err := build(t, "")()
		if err != nil {
			return nil, err
		}
		e.(seriesEngine).EnableServerSeries(interval)
		last = &snapshotting{Engine: e}
		return last, nil
	}
	// a, b and c share a prompt prefix (same-length IDs), so the later ones hit the cache.
	arr := []enginetest.Arrival{{At: 0, Req: req("a", 48, 6)}, {At: 0, Req: req("dd", 30, 3)},
		{At: 900, Req: req("b", 48, 5)}, {At: 1700, Req: req("c", 80, 4)}}
	tr := enginetest.TestEngine(t, newE, arr, enginetest.Options{})
	var prompt, generated, hits, queries int64
	for _, a := range arr {
		prompt += int64(len(a.Req.Prompt))
	}
	for _, st := range tr.Steps {
		for _, n := range st.Result.Emitted {
			generated += int64(n)
		}
		for _, s := range st.Batch.Reqs {
			if s.PrefixLookup {
				hits += int64(s.CachedTokens)
				queries += int64(s.PromptLen)
			}
		}
	}
	se := last.Engine.(seriesEngine)
	iv, end := se.ServerSeriesSpan()
	if iv != interval || end != int64(tr.End) {
		t.Fatalf("span = (%d, %d), want (%d, %d)", iv, end, interval, tr.End)
	}
	raw, err := se.ServerSeriesBlock("instance_0", end)
	if err != nil {
		t.Fatal(err)
	}
	var out ServerSeriesOutput
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	if again, _ := json.Marshal(out); !bytes.Equal(again, raw) {
		t.Errorf("the block does not round-trip:\n%s\n%s", raw, again)
	}
	if len(out.Engines) != 1 || out.IntervalS != 250e-6 || out.Engines[0].ID != "instance_0" {
		t.Fatalf("block = %+v", out)
	}
	e := out.Engines[0]
	if want := int(end/interval) + 1; e.Bins != want || len(e.PromptTokens) != want || len(e.KVCacheUsage) != want {
		t.Fatalf("bins = %d (%d prompt, %d usage), want %d", e.Bins, len(e.PromptTokens), len(e.KVCacheUsage), want)
	}
	sum := func(c []int64) (n int64) {
		for _, v := range c {
			n += v
		}
		return n
	}
	m := last.Metrics(tr.End)
	value := func(family, source string) int64 {
		f, _ := m.Family(family)
		for _, s := range f.Samples {
			if s.Labels["source"] == source {
				return int64(s.Value)
			}
		}
		t.Fatalf("no %s{source=%q}", family, source)
		return 0
	}
	for _, c := range []struct {
		name      string
		got, want int64
		run       int64
	}{
		{"prompt_tokens", sum(e.PromptTokens), value("vllm:prompt_tokens", ""), prompt},
		{"generation_tokens", sum(e.GenerationTokens), value("vllm:generation_tokens", ""), generated},
		{"prefix_cache_hits", sum(e.PrefixCacheHits), value("vllm:prefix_cache_hits", ""), hits},
		{"prefix_cache_queries", sum(e.PrefixCacheQueries), value("vllm:prefix_cache_queries", ""), queries},
		{"prompt_tokens_local_cache_hit", sum(e.PromptTokensLocalCacheHit),
			value("vllm:prompt_tokens_by_source", "local_cache_hit"), hits},
		{"prompt_tokens_local_compute", sum(e.PromptTokensLocalCompute),
			value("vllm:prompt_tokens_by_source", "local_compute"), prompt - hits},
		{"prompt_tokens_external_kv_transfer", sum(e.PromptTokensExternalKVTransfer),
			value("vllm:prompt_tokens_by_source", "external_kv_transfer"), 0},
	} {
		if c.got != c.want || c.got != c.run {
			t.Errorf("%s: bins sum to %d; Metrics %d, the run %d", c.name, c.got, c.want, c.run)
		}
	}
	if hits == 0 || generated == 0 {
		t.Fatalf("the run hit %d prefix tokens and generated %d: nothing was tested", hits, generated)
	}
	// Each bin's last value is the state after the last step ending in it, or before it.
	k := 0
	var cur engine.State
	for i, at := range last.at {
		for ; int64(k) < int64(at)/interval; k++ {
			checkLast(t, e, k, cur)
		}
		cur = last.state[i]
	}
	for ; k < e.Bins; k++ {
		checkLast(t, e, k, cur)
	}
}

func checkLast(t *testing.T, e ServerSeriesEngine, k int, s engine.State) {
	t.Helper()
	usage := float64(s.KVBlocksUsed) / float64(s.KVBlocksTotal)
	if e.NumRequestsRunningLast[k] != float64(s.Running) || e.NumRequestsWaitingLast[k] != float64(s.Waiting) ||
		math.Abs(e.KVCacheUsageLast[k]-usage) > 1e-12 {
		t.Errorf("bin %d last: running %v waiting %v usage %v; Snapshot %d %d %v", k, e.NumRequestsRunningLast[k],
			e.NumRequestsWaitingLast[k], e.KVCacheUsageLast[k], s.Running, s.Waiting, usage)
	}
}

// A data-parallel group reports one engine per rank, as vLLM labels each engine core's series.
func TestDPGroupServerSeriesIsOneEnginePerRank(t *testing.T) {
	src := "use_v2_model_runner: false\nasync_scheduling: false\npipeline_parallel_size: 1\ndata_parallel_size: 2\n"
	var last engine.Engine
	newE := func() (engine.Engine, error) {
		e, err := build(t, src)()
		if err == nil {
			e.(seriesEngine).EnableServerSeries(300)
		}
		last = e
		return e, err
	}
	tr := enginetest.TestEngine(t, newE, arrivals(req("a", 12, 4), req("b", 20, 6), req("c", 8, 3), req("d", 30, 2)),
		enginetest.Options{})
	raw, err := last.(seriesEngine).ServerSeriesBlock("instance_0", int64(tr.End))
	if err != nil {
		t.Fatal(err)
	}
	var out ServerSeriesOutput
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Engines) != 2 || out.Engines[0].ID != "instance_0/engine=0" || out.Engines[1].ID != "instance_0/engine=1" {
		t.Fatalf("engines = %+v", out.Engines)
	}
	var gen int64
	for _, e := range out.Engines {
		if e.Bins != out.Engines[0].Bins {
			t.Errorf("rank %s has %d bins, rank 0 %d", e.ID, e.Bins, out.Engines[0].Bins)
		}
		for _, v := range e.GenerationTokens {
			gen += v
		}
	}
	if gen != 15 {
		t.Errorf("generation tokens over both ranks: %d, want 15", gen)
	}
}
