// Original to blis-vllm.

package vllm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// group builds a data-parallel group of four ranks with the load-balancing keys in extra.
func group(t *testing.T, extra string) *DPGroup {
	t.Helper()
	e, err := build(t, dp4+"enable_prefix_caching: true\n"+extra)()
	if err != nil {
		t.Fatal(err)
	}
	return e.(*DPGroup)
}

// serve adds one request through add, runs the group until it is idle and returns the tokens the
// request's prompt took from its rank's prefix cache, and the rank that served it.
func serve(t *testing.T, g *DPGroup, r engine.Request, add func(engine.Request) error, now engine.Time) (cached, rank int, end engine.Time) {
	t.Helper()
	if err := add(r); err != nil {
		t.Fatal(err)
	}
	rank = g.rankOf[r.ID]
	for range 1000 {
		b, err := g.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		if b.Empty() {
			if w, ok := g.NextWake(now); ok {
				now = max(w, now+1)
				continue
			}
			return cached, rank, now
		}
		for _, s := range b.Reqs {
			if s.ID == r.ID && s.CachedTokens > cached {
				cached = s.CachedTokens
			}
		}
		now += g.StepTime(b)
		if _, err := g.Complete(b.ID, engine.Outcome{}, now); err != nil {
			t.Fatal(err)
		}
	}
	t.Fatal("the group never went idle")
	return 0, 0, now
}

// Under external balancing every rank is an endpoint the router chooses between: the group has as
// many endpoints as ranks, a request goes to the rank its endpoint serves whatever the load, and
// a request that names no endpoint is refused rather than balanced by the engine.
func TestExternalBalancingExposesEveryRankAsAnEndpoint(t *testing.T) {
	g := group(t, "data_parallel_external_lb: true\n")
	if g.Endpoints() != 4 {
		t.Fatalf("%d endpoints, want 4 (one per rank)", g.Endpoints())
	}
	if first, n := g.EndpointRanks(2); first != 2 || n != 1 {
		t.Errorf("endpoint 2 serves ranks from %d, %d of them; want rank 2 alone", first, n)
	}
	if err := g.Add(req("x", 12, 2), 0); err == nil || !strings.Contains(err.Error(), "AddToEndpoint") {
		t.Errorf("a request naming no endpoint gave %v, want a refusal that names AddToEndpoint", err)
	}
	// Three requests to endpoint 2 all run on rank 2, though ranks 0, 1 and 3 are idle: the engine
	// does not rebalance what the router sent.
	for _, id := range []string{"a", "b", "c"} {
		if err := g.AddToEndpoint(2, req(id, 12, 2), 0); err != nil {
			t.Fatal(err)
		}
		if g.rankOf[id] != 2 {
			t.Errorf("request %s on rank %d, want 2", id, g.rankOf[id])
		}
	}
	if err := g.AddToEndpoint(4, req("z", 12, 2), 0); err == nil {
		t.Error("an endpoint the group does not have was accepted")
	}
}

// Under hybrid balancing each node's front end balances only its node's ranks. With dp 4 and
// data_parallel_size_local 2 there are two endpoints; requests sent to endpoint 1 spread over
// ranks 2 and 3 and never reach ranks 0 and 1.
func TestHybridBalancingBalancesWithinTheNodesRanks(t *testing.T) {
	g := group(t, "data_parallel_hybrid_lb: true\ndata_parallel_size_local: 2\n")
	if g.Endpoints() != 2 {
		t.Fatalf("%d endpoints, want 2 (one per node of 2 ranks)", g.Endpoints())
	}
	if first, n := g.EndpointRanks(1); first != 2 || n != 2 {
		t.Errorf("endpoint 1 serves %d ranks from %d, want 2 from 2", n, first)
	}
	got := map[int]int{}
	for _, id := range []string{"a", "b", "c", "d"} {
		if err := g.AddToEndpoint(1, req(id, 12, 2), 0); err != nil {
			t.Fatal(err)
		}
		got[g.rankOf[id]]++
	}
	if got[0]+got[1] != 0 || got[2] != 2 || got[3] != 2 {
		t.Errorf("endpoint 1 placed requests on ranks %v, want two each on ranks 2 and 3 and none on 0 or 1", got)
	}
}

// vLLM resolves the flags before it picks a client (arg_utils.py:2383-2538): hybrid with one rank
// per node runs as external, hybrid with every rank on one node as internal, and the combinations
// it refuses are refused.
func TestBalancingFlagsResolveAsVLLMResolvesThem(t *testing.T) {
	for _, c := range []struct {
		name, extra string
		endpoints   int
	}{
		{"unstated is the internal balancer", "", 1},
		{"hybrid with one rank per node is external", "data_parallel_hybrid_lb: true\ndata_parallel_size_local: 1\n", 4},
		{"hybrid with every rank local is internal", "data_parallel_hybrid_lb: true\ndata_parallel_size_local: 4\n", 1},
		{"the multi-port launcher is external", "data_parallel_multi_port_external_lb: true\n", 4},
		{"external with local size 1 is external", "data_parallel_external_lb: true\ndata_parallel_size_local: 1\n", 4},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := group(t, c.extra).Endpoints(); got != c.endpoints {
				t.Errorf("%d endpoints, want %d", got, c.endpoints)
			}
		})
	}
	for _, c := range []struct{ name, extra, want string }{
		{"hybrid and external together", "data_parallel_hybrid_lb: true\ndata_parallel_external_lb: true\ndata_parallel_size_local: 2\n", "cannot be enabled together"},
		{"hybrid without a local size", "data_parallel_hybrid_lb: true\n", "requires data_parallel_size_local"},
		{"external with a local size above 1", "data_parallel_external_lb: true\ndata_parallel_size_local: 2\n", "requires data_parallel_size_local 1"},
		{"hybrid over uneven nodes", "data_parallel_hybrid_lb: true\ndata_parallel_size_local: 3\n", "whole number of nodes"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := build(t, dp4+c.extra)()
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("gave %v, want a refusal containing %q", err, c.want)
			}
		})
	}
	// The internal balancer is a group's one endpoint, and an endpoint index is refused there.
	if err := group(t, "").AddToEndpoint(0, req("x", 12, 2), 0); err == nil {
		t.Error("AddToEndpoint was accepted by the internal balancer")
	}
}

// The reason a deployment runs a router in front of external ranks: it can send a request to the
// rank that holds its prefix. A second request sharing the first's prompt hits the prefix cache
// when the router sends it to the same rank, and misses when the internal balancer, which spreads
// by load and rotates its scan start, sends it to another.
func TestARouterThatKeepsAPrefixOnOneRankGetsTheCacheHit(t *testing.T) {
	prompt := func(id string) engine.Request {
		r := req("x", 48, 1)
		r.ID = id
		return r
	}
	ext := group(t, "data_parallel_external_lb: true\n")
	toRank0 := func(r engine.Request) error { return ext.AddToEndpoint(0, r, 0) }
	_, r1, end := serve(t, ext, prompt("first"), toRank0, 0)
	cachedExt, r2, _ := serve(t, ext, prompt("second"), toRank0, end)
	if r1 != 0 || r2 != 0 {
		t.Fatalf("external requests ran on ranks %d and %d, want 0 and 0", r1, r2)
	}

	in := group(t, "")
	add := func(r engine.Request) error { return in.Add(r, 0) }
	_, i1, end := serve(t, in, prompt("first"), add, 0)
	cachedIn, i2, _ := serve(t, in, prompt("second"), add, end)
	if i1 == i2 {
		t.Fatalf("the internal balancer sent both requests to rank %d; the comparison needs it to spread them", i1)
	}
	if cachedExt != 44 || cachedIn != 0 {
		t.Errorf("cached prompt tokens of the second request: external same-rank %d, internal spread %d; want 44 (a 48-token prompt less its last 4-token block) and 0", cachedExt, cachedIn)
	}
}

// modeScenario is a scenario of oracle/scenarios-dp-modes as the engine's tests read it.
type modeScenario struct {
	Config   map[string]any `json:"config"`
	Requests []struct {
		ID       string  `json:"id"`
		Prompt   []int32 `json:"prompt"`
		Output   []int32 `json:"output"`
		Endpoint int     `json:"endpoint"`
	} `json:"requests"`
}

func loadModeScenarios(t *testing.T) map[string]modeScenario {
	t.Helper()
	paths, err := filepath.Glob("../../oracle/scenarios-dp-modes/*.json")
	if err != nil || len(paths) == 0 {
		t.Fatalf("no balancing-mode scenarios (%v)", err)
	}
	out := map[string]modeScenario{}
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var sc modeScenario
		if err := json.Unmarshal(raw, &sc); err != nil {
			t.Fatal(err)
		}
		out[strings.TrimSuffix(filepath.Base(p), ".json")] = sc
	}
	return out
}

// modeConfig is the engine block of a scenario, with a max_model_len this engine's startup check
// accepts for the scenario's 30 blocks (the oracle's mock scheduler does not check it); internal drops the balancing flags, so the same
// requests meet vLLM's internal balancer.
func modeConfig(sc modeScenario, internal bool) string {
	c := sc.Config
	var b strings.Builder
	fmt.Fprintf(&b, "block_size: %v\nmax_num_seqs: %v\nmax_num_batched_tokens: %v\nmax_model_len: %v\nnum_gpu_blocks: %v\n"+
		"use_v2_model_runner: false\nasync_scheduling: false\npipeline_parallel_size: 1\ndata_parallel_size: %v\nenable_prefix_caching: %v\n",
		c["block_size"], c["max_num_seqs"], c["max_num_batched_tokens"], 32, c["num_blocks"], c["data_parallel_size"],
		c["enable_prefix_caching"] == true)
	if !internal {
		for _, k := range []string{"data_parallel_hybrid_lb", "data_parallel_external_lb", "data_parallel_multi_port_external_lb"} {
			if c[k] == true {
				fmt.Fprintf(&b, "%s: true\n", k)
			}
		}
		if v, ok := c["data_parallel_size_local"]; ok {
			fmt.Fprintf(&b, "data_parallel_size_local: %v\n", v)
		}
	}
	return b.String()
}

// runAll sends every request of the scenario at once through add, runs the group until it is
// idle, and returns the requests each rank served and the number that finished.
func runAll(t *testing.T, g *DPGroup, sc modeScenario, add func(i int, r engine.Request) error) (perRank map[int]int, finished int) {
	t.Helper()
	perRank = map[int]int{}
	for i, r := range sc.Requests {
		er := engine.Request{ID: r.ID, Prompt: r.Prompt, Output: r.Output}
		if err := add(i, er); err != nil {
			t.Fatal(err)
		}
		perRank[g.rankOf[r.ID]]++
	}
	now := engine.Time(0)
	for range 5000 {
		b, err := g.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		if b.Empty() {
			if w, ok := g.NextWake(now); ok {
				now = max(w, now+1)
				continue
			}
			return perRank, finished
		}
		now += g.StepTime(b)
		res, err := g.Complete(b.ID, engine.Outcome{}, now)
		if err != nil {
			t.Fatal(err)
		}
		finished += len(res.Finished)
	}
	t.Fatal("the group never went idle")
	return nil, 0
}

// A scenario in each mode runs through the engine to completion, and the router's choice shows in
// the ranks' load in the direction vLLM's own run shows: a burst the router sends to one external
// rank stays on it, where the internal balancer spreads the same requests over every rank.
func TestEachBalancingModeRunsItsScenarioAndConcentratesWhatTheRouterSent(t *testing.T) {
	for name, sc := range loadModeScenarios(t) {
		t.Run(name, func(t *testing.T) {
			routed, err := build(t, modeConfig(sc, false))()
			if err != nil {
				t.Fatal(err)
			}
			g := routed.(*DPGroup)
			perRank, finished := runAll(t, g, sc, func(i int, r engine.Request) error {
				return g.AddToEndpoint(sc.Requests[i].Endpoint, r, 0)
			})
			if finished != len(sc.Requests) {
				t.Fatalf("%d of %d requests finished", finished, len(sc.Requests))
			}
			spread, err := build(t, modeConfig(sc, true))()
			if err != nil {
				t.Fatal(err)
			}
			ig := spread.(*DPGroup)
			internal, ifinished := runAll(t, ig, sc, func(_ int, r engine.Request) error { return ig.Add(r, 0) })
			if ifinished != len(sc.Requests) {
				t.Fatalf("internal: %d of %d requests finished", ifinished, len(sc.Requests))
			}
			maxOf := func(m map[int]int) int {
				top := 0
				for _, n := range m {
					top = max(top, n)
				}
				return top
			}
			if name == "dp4-external-rank-affinity" && maxOf(perRank) <= maxOf(internal) {
				t.Errorf("the router's busiest rank served %d requests, the internal balancer's %d: the router's concentration should show (%v against %v)",
					maxOf(perRank), maxOf(internal), perRank, internal)
			}
		})
	}
}
