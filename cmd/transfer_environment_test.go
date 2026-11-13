package cmd

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"pgregory.net/rapid"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/env"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/units"
	"github.com/inference-sim/blis-substrate/load"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// The host's Environment is the only place a move an engine states becomes a demand. These tests
// hold it to the prices of the paths it replaced: the cluster's PDTransferTime/PDTransferDemand
// (a P/D handoff, from the prefill pool's kernel between the instances' placements) and the
// engine's TransferTime (an offload copy, from the kernel's TierTime). Equivalence of whole runs
// is the byte gate and the run/replay cases of the pull request's body.

// openScenarioModel opens the pool of a scenario in dir. A tier's rates are registry entries (the
// set cost-model-offload-tiers), so a scenario that does not list the set is opened from a copy
// that does: these tests price tier copies on scenarios that do not offload.
func openScenarioModel(t *testing.T, dir, scenario string) *kernelmodel.Model {
	t.Helper()
	_, catalog, registry := kernelRepos(t)
	raw, err := os.ReadFile(filepath.Join(dir, scenario))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "cost-model-offload-tiers") {
		re := regexp.MustCompile(`(?m)^(coefficients: \[[^\]]*)\]`)
		if !re.Match(raw) {
			t.Fatalf("%s: no coefficients list to add the offload set to", scenario)
		}
		raw = re.ReplaceAll(raw, []byte("${1}, cost-model-offload-tiers]"))
		tmp := t.TempDir()
		if err := os.WriteFile(filepath.Join(tmp, scenario), raw, 0o644); err != nil {
			t.Fatal(err)
		}
		dir = tmp
	}
	m, err := kernelmodel.Open(scenario, reposOf(dir, catalog, registry))
	if err != nil {
		t.Fatalf("%s: %v", scenario, err)
	}
	return m
}

// tierPlanFor is the pool plan an engine with a cpu_dram tier on a pool of nodes nodes states.
func tierPlanFor(nodes int) plan.PoolPlan {
	return plan.PoolPlan{Hardware: "chip", Layout: plan.Layout{Tensor: 1, RanksPerNode: 1, Nodes: nodes},
		Tiers: []plan.KVTier{{Device: "cpu_dram"}}}
}

func tierMove(tokens int64, toTier bool, node int) plan.Transfer {
	k := plan.TransferFromTier
	if toTier {
		k = plan.TransferToTier
	}
	at := plan.Endpoint{Node: node}
	return plan.Transfer{Tokens: units.Tokens(tokens), From: at, To: at, Kind: k, Tier: "cpu_dram"}
}

// An offload copy's demand, run alone and truncated to whole microseconds as the engine's tier
// copies always were, is the kernel's TierTime of the same blocks (kernelmodel.TierTransferTicks),
// for every size and both directions, on a dense model, a hybrid one (sliding-window layers) and a
// pool of two nodes; and the demand alone costs the kernel's exact price (a Duration), so the
// Environment states no rate it could not vouch for. The rate is the one the kernel's price of
// a payload that binds recovers (tierRate), never the per-copy fallback.
func TestTransferEnvironment_ATierCopyIsTheRetiredTierPrice(t *testing.T) {
	for _, c := range []struct{ dir, scenario string }{
		{"../testdata/kernel/aisimulate", "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml"},
		{"../testdata/bytegate/scenarios", "gpt-oss-120b-h100-fp4-vllm-tp2.yaml"},
		{"../testdata/scenarios/muse-spark", "muse-h100-roce-armD.yaml"},
	} {
		t.Run(c.scenario, func(t *testing.T) {
			m := openScenarioModel(t, c.dir, c.scenario)
			e, err := newTransferEnvironment(m, nil)
			if err != nil {
				t.Fatal(err)
			}
			rapid.Check(t, func(rt *rapid.T) {
				blocks := rapid.Int64Range(1, 1<<16).Draw(rt, "blocks")
				toTier := rapid.Bool().Draw(rt, "toTier")
				// A copy moves whole blocks; the engine states the tokens they hold, and a short last
				// block is still a whole block to the copy.
				tokens := (blocks-1)*e.blockSize + rapid.Int64Range(1, e.blockSize).Draw(rt, "last")
				d, err := e.TransferDemand(tierPlanFor(2), tierMove(tokens, toTier, 0))
				if err != nil {
					rt.Fatal(err)
				}
				if got, want := sim.TierTicks(d), m.TierTransferTicks("cpu_dram", toTier, blocks); got != want {
					rt.Fatalf("%d blocks (toTier %t): the demand costs %d ticks, the retired price was %d", blocks, toTier, got, want)
				}
				dir := kernel.DirectionFromTier
				if toTier {
					dir = kernel.DirectionToTier
				}
				if want := e.rates[tierKey{"cpu_dram", dir}]; math.Abs(d.Links[0].BytesPerSecond-want) > 1e-6*want {
					rt.Fatalf("%d blocks: priced at a per-copy rate %v, not the device's %v", blocks,
						d.Links[0].BytesPerSecond, e.rates[tierKey{"cpu_dram", dir}])
				}
			})
		})
	}
}

// The Environment refuses what it cannot price rather than state a free transfer: a tier the plan
// does not list, a copy of no tokens, a kind outside the list, a pull from a pool it has no
// kernel for, and a node the engine's own plan does not have; every question other than a
// transfer's is refused by name; and a pull of no tokens demands nothing.
func TestTransferEnvironment_RefusesWhatItCannotPrice(t *testing.T) {
	m := openScenarioModel(t, "../testdata/kernel/aisimulate", "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml")
	e, err := newTransferEnvironment(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	p := tierPlanFor(1)
	bad := map[string]plan.Transfer{
		"unlisted tier":      func() plan.Transfer { x := tierMove(64, true, 0); x.Tier = "nvme"; return x }(),
		"empty copy":         tierMove(0, true, 0),
		"unknown kind":       func() plan.Transfer { x := tierMove(64, true, 0); x.Kind = "sideways"; return x }(),
		"node not in pool":   tierMove(64, true, 3),
		"pull, no peer pool": {Tokens: 64, From: plan.Endpoint{Node: 1}, To: plan.Endpoint{}, Kind: plan.TransferPeer},
	}
	for name, mv := range bad {
		if d, err := e.TransferDemand(p, mv); err == nil {
			t.Errorf("%s: priced as %+v", name, d)
		}
	}
	if d, err := e.TransferDemand(p, plan.Transfer{Kind: plan.TransferPeer, From: plan.Endpoint{Node: 1}}); err != nil || d.Bytes != 0 || len(d.Links) != 0 {
		t.Errorf("a pull of no tokens: demand %+v, error %v; want an empty demand", d, err)
	}
	if _, err := e.Price(p, plan.StepShape{}); err == nil {
		t.Error("the Environment priced a step; it answers transfers only")
	}
	if _, err := e.Memory(p); err == nil {
		t.Error("the Environment sized memory; it answers transfers only")
	}
}

// Moves in one pool, stated on two nodes (the muse-spark arm D engine spans both): the copy is
// priced for the one node the engine states, on that node's link, and costs the same on either
// (blis-substrate#48). The engine's workers on every node copy their own shard at once; the
// boundary has one From and one To, and the retired path priced one rate for the whole copy.
func TestTransferEnvironment_ACopyOnAPoolOfTwoNodesIsPricedOnTheNodeTheEngineNames_substrate48(t *testing.T) {
	m := openScenarioModel(t, "../testdata/scenarios/muse-spark", "muse-h100-roce-armD.yaml")
	e, err := newTransferEnvironment(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	a, err := e.TransferDemand(tierPlanFor(2), tierMove(4096, true, 0))
	if err != nil {
		t.Fatal(err)
	}
	b, err := e.TransferDemand(tierPlanFor(2), tierMove(4096, true, 1))
	if err != nil {
		t.Fatal(err)
	}
	if a.Bytes != b.Bytes || a.AloneTime() != b.AloneTime() || a.Links[0].Link == b.Links[0].Link {
		t.Errorf("node 0 %+v, node 1 %+v: want the same price on two different links", a, b)
	}
}

// A hybrid model's KV groups hold different bytes a block, and vLLM copies each group's blocks
// over that group's layers only. plan.Transfer has one token count, so the Environment prices
// the blocks at the pool's page size, the one rule it can express (blis-substrate#49). For the
// shipped hybrid scenario that equals the retired group-sized price (the byte gate's runs and the
// body's evidence compare whole runs); this test names the rule so that a boundary that grows
// per-group sizes changes it on purpose.
func TestTransferEnvironment_AHybridModelsCopyIsPricedAtThePoolsPageSize_substrate49(t *testing.T) {
	m := openScenarioModel(t, "../testdata/bytegate/scenarios", "gpt-oss-120b-h100-fp4-vllm-tp2.yaml")
	e, err := newTransferEnvironment(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	const blocks = 37
	d, err := e.TransferDemand(tierPlanFor(1), tierMove(blocks*e.blockSize, false, 0))
	if err != nil {
		t.Fatal(err)
	}
	if want := blocks * m.Kernel().SequenceVariableBytes(int(e.blockSize)); d.Bytes != want {
		t.Errorf("the copy moves %d bytes, the pool's page size says %d", d.Bytes, want)
	}
}

// The P/D handoffs of the shipped 3-prefill, 1-decode scenario, in whole blocks, for every
// prompt length: the demand the Environment states is the prefill kernel's PDTransferDemand
// between the two instances' placements field for field (so contention arbitrates the same
// links at the same rates), and alone it costs the retired PDTransferTime in ticks. A prompt that
// does not fill its last block moves the whole block.
func TestTransferEnvironment_APullIsTheRetiredHandoffDemand(t *testing.T) {
	pools, restore := openPools(t, pdScenarios, "glm-5-h200-3p1d-ib.yaml", 3, 1)
	defer restore()
	e, err := newTransferEnvironment(pools.decode, pools.prefill)
	if err != nil {
		t.Fatal(err)
	}
	endpointOf := pools.instanceEndpoint()
	decodePlan := plan.PoolPlan{Hardware: "chip", Layout: plan.Layout{Tensor: 1, RanksPerNode: 1, Nodes: 1}}
	block := e.blockSize
	rapid.Check(t, func(rt *rapid.T) {
		tokens := rapid.Int64Range(1, 200_000).Draw(rt, "tokens")
		capacity := (tokens + block - 1) / block * block
		for i := 0; i < 3; i++ {
			from := "instance_" + string(rune('0'+i))
			d, err := env.DemandFor(e, decodePlan, engine.Transfer{ID: 1, Request: "r", Kind: engine.TransferLoad, Tokens: int(tokens),
				Blocks: int(capacity / block), Move: plan.Transfer{Tokens: units.Tokens(tokens), From: endpointOf(cluster.InstanceID(from)),
					To: endpointOf("instance_3"), Kind: plan.TransferPeer}})
			if err != nil {
				rt.Fatal(err)
			}
			old := pools.prefill.Kernel().PDTransferDemand(int(capacity), pools.prefill.PlacementOf(i), pools.decode.PlacementOf(0))
			if cluster.FlowOf(d).Bytes != old.Bytes || d.Floor != old.Floor || len(d.Links) != len(old.Links) {
				rt.Fatalf("%d tokens from %s: demand %+v, the retired %+v", tokens, from, d, old)
			}
			for k := range old.Links {
				if d.Links[k] != old.Links[k] {
					rt.Fatalf("%d tokens from %s: link %+v, the retired %+v", tokens, from, d.Links[k], old.Links[k])
				}
			}
			if got, want := cluster.AloneTicks(cluster.FlowOf(d)), pools.prefill.PDTransferTicks(capacity,
				pools.prefill.PlacementOf(i), pools.decode.PlacementOf(0)); got != want {
				rt.Fatalf("%d tokens from %s: %d ticks, the retired price %d", tokens, from, got, want)
			}
		}
	})
}

// openPools opens a scenario's prefill and decode pools at the given instance counts, as the CLI
// opens them, and returns them with the function that restores the globals it set.
func openPools(t *testing.T, dir, scenario string, prefill, decode int) (*kernelPools, func()) {
	t.Helper()
	_, catalog, registry := kernelRepos(t)
	saved := []any{kernelScenario, kernelScenarioDir, registryPath, kernelOpened,
		prefillInstances, decodeInstances, numInstances, blockSizeTokens, kernelPDOpened}
	restore := func() {
		kernelScenario, kernelScenarioDir, registryPath = saved[0].(string), saved[1].(string), saved[2].(string)
		kernelOpened, _ = saved[3].(*kernelmodel.Model)
		prefillInstances, decodeInstances, numInstances = saved[4].(int), saved[5].(int), saved[6].(int)
		blockSizeTokens = saved[7].(int64)
		kernelPDOpened, _ = saved[8].(*kernelPools)
	}
	m, err := kernelmodel.Open(scenario, reposOf(dir, catalog, registry))
	if err != nil {
		t.Fatal(err)
	}
	st, err := m.Settings()
	if err != nil {
		t.Fatal(err)
	}
	kernelScenario, kernelScenarioDir, registryPath, kernelOpened = scenario, dir, registry, m
	blockSizeTokens = int64(st.BlockSize)
	prefillInstances, decodeInstances, numInstances = prefill, decode, prefill+decode
	pools := openKernelPools()
	if pools == nil {
		restore()
		t.Fatal("no pools opened for a disaggregated run")
	}
	return pools, restore
}

// Two decode instances of one pool, on two nodes, state different nodes for themselves through the
// engine this binary builds (0.32.0): the host places each instance (placeInstance) from the
// endpoint it states to a decode engine as its peer, so a pull's near end is the instance's own
// node and rack and not the pool's first. Without the call both would report node 0. The pools are
// the 3P1D scenario with a decode pool of two nodes, one instance on each.
func TestPlaceInstance_TwoDecodeInstancesStateTheirOwnNodes(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(pdScenarios, "glm-5-h200-3p1d-ib.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	two := strings.Replace(strings.Replace(string(src), "  nodes: 4\n", "  nodes: 5\n", 1),
		"  - role: decode\n    nodes: 1\n", "  - role: decode\n    nodes: 2\n", 1)
	if two == string(src) {
		t.Fatal("the scenario no longer has the shape this test edits")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "glm-5-two-decode-nodes.yaml"), []byte(two), 0o644); err != nil {
		t.Fatal(err)
	}
	pools, restore := openPools(t, dir, "glm-5-two-decode-nodes.yaml", 3, 2)
	defer restore()
	registerContractEngines()
	data, err := load.Open(load.Options{Catalog: testCatalogDir(), Registry: testRegistryDir()})
	if err != nil {
		t.Fatal(err)
	}
	pull := func(index int) engine.Transfer {
		t.Helper()
		spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0",
			Roots: contractRoots(filepath.Join(dir, "glm-5-two-decode-nodes.yaml")), Pool: 1, Seed: 1}
		eng, err := newContractEngine(spec, data)
		if err != nil {
			t.Fatal(err)
		}
		if err := placeInstance(eng, pools, index); err != nil {
			t.Fatal(err)
		}
		peer := plan.Endpoint{Node: 1, Rack: 1}
		prompt := make([]int32, 256)
		if err := eng.Add(engine.Request{ID: "r", Prompt: prompt, Output: make([]int32, 4), MaxOutputTokens: 4,
			RemotePrefill: true, Peer: &peer}, 0); err != nil {
			t.Fatal(err)
		}
		b, err := eng.Schedule(0)
		if err != nil {
			t.Fatal(err)
		}
		if len(b.TransfersStarted) != 1 {
			t.Fatalf("instance_%d started %d transfers, want one pull", index, len(b.TransfersStarted))
		}
		return b.TransfersStarted[0]
	}
	a, b := pull(3), pull(4) // the two decode instances: ranks 0 and 1 of the decode pool
	want := pools.instanceEndpoint()
	if a.Move.To != want("instance_3") || b.Move.To != want("instance_4") || a.Move.To == b.Move.To {
		t.Errorf("instance_3 lands pulls on %+v and instance_4 on %+v, want their endpoints %+v and %+v, which differ",
			a.Move.To, b.Move.To, want("instance_3"), want("instance_4"))
	}
	if a.Move.From != (plan.Endpoint{Node: 1, Rack: 1}) || b.Move.From != a.Move.From {
		t.Errorf("the pull's far end is %+v and %+v, want the peer the request named", a.Move.From, b.Move.From)
	}
}

// noPlacement is an engine that states Plan, Move and Peer (it is a real engine behind a type that
// promotes only the Engine and Planned methods) and implements no engine.Placeable.
type noPlacement struct {
	engine.Engine
	engine.Planned
}

// A disaggregated run whose engine must be placed and cannot be is refused with the contract's
// error, and not run with every instance at the pool's first node: an engine with the transfer
// boundary and no PlaceAt, in a pool of more than one node, is engine.ErrPlacementUnsupported
// naming the instance; an engine that accepts placement is placed; a colocated run needs none.
func TestPlaceInstance_AnEngineThatAcceptsNoPlacementIsRefusedInADisaggregatedRun(t *testing.T) {
	pools, restore := openPools(t, pdScenarios, "glm-5-h200-3p1d-ib.yaml", 3, 1)
	defer restore()
	if pools.prefill.EnginePlacement().Nodes <= 1 {
		t.Fatal("fixture: the prefill pool must span more than one node for placement to be required")
	}
	registerContractEngines()
	data, err := load.Open(load.Options{Catalog: testCatalogDir(), Registry: testRegistryDir()})
	if err != nil {
		t.Fatal(err)
	}
	spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0",
		Roots: contractRoots(filepath.Join(pdScenarios, "glm-5-h200-3p1d-ib.yaml")), Pool: 0, Seed: 1}
	eng, err := newContractEngine(spec, data)
	if err != nil {
		t.Fatal(err)
	}
	if err := placeInstance(eng, pools, 2); err != nil {
		t.Fatalf("an engine that accepts placement was refused: %v", err)
	}
	bad := noPlacement{Engine: eng, Planned: eng.(engine.Planned)}
	err = placeInstance(bad, pools, 2)
	if !errors.Is(err, engine.ErrPlacementUnsupported) || !strings.Contains(err.Error(), "instance_2") {
		t.Errorf("an engine with Plan/Move/Peer and no PlaceAt: err = %v, want engine.ErrPlacementUnsupported naming instance_2", err)
	}
	if err := placeInstance(bad, nil, 0); err != nil {
		t.Errorf("a colocated run (no pools) must leave the engine as built: %v", err)
	}
}

// A prefill pool of 18 GB200 trays fills rack 0 (72 GPUs, four per tray) and a one-tray decode pool
// starts on the next rack, so the two pools' local rack 0 are different racks of the cluster.
// Every endpoint is in the numbering of the pool it is in, node and rack both: the decode instance is
// at its pool's {0, 0}, and the Environment turns each end of a pull back into a cluster placement
// with the origin of the pool the end is in, so the pull is priced between the two pools' real
// places. A host that offsets the node and leaves the rack in the cluster's numbering states the
// decode instance at the cluster's rack, and reads an engine's pool-relative {0, 0} as rack 0, a
// rack the decode pool does not hold.
func TestPullBetweenPoolsThatStartOnDifferentRacksIsPricedBetweenTheirRealPlaces(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(pdScenarios, "minimax-m2.5-gb200-1p1d-dp2.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	edited := strings.NewReplacer("    nodes: 17\n", "    nodes: 18\n", "    nodes: 2\n", "    nodes: 1\n",
		"      dp: 2\n", "      dp: 1\n").Replace(string(src))
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "gb200-rack-boundary.yaml"), []byte(edited), 0o644); err != nil {
		t.Fatal(err)
	}
	pools, restore := openPools(t, dir, "gb200-rack-boundary.yaml", 1, 1)
	defer restore()
	prefillAt, decodeAt := pools.prefill.GroupPlacementOf(0), pools.decode.GroupPlacementOf(0)
	if prefillAt.Rack == decodeAt.Rack || decodeAt.Rack == 0 {
		t.Fatalf("fixture: prefill is at rack %d and decode at rack %d, want the decode pool to start on a rack other than 0", prefillAt.Rack, decodeAt.Rack)
	}
	endpointOf := pools.instanceEndpoint()
	if got := endpointOf("instance_1"); got != (plan.Endpoint{}) {
		t.Errorf("the decode pool's only instance is at %+v in its own numbering, want node 0 rack 0 (the cluster has it at node %d rack %d)",
			got, decodeAt.Node, decodeAt.Rack)
	}
	e, err := newTransferEnvironment(pools.decode, pools.prefill)
	if err != nil {
		t.Fatal(err)
	}
	const tokens = 4096
	decodePlan := plan.PoolPlan{Hardware: "chip", Layout: plan.Layout{Tensor: 4, RanksPerNode: 4, Nodes: 1}}
	got, err := e.TransferDemand(decodePlan, plan.Transfer{Tokens: tokens, From: plan.Endpoint{}, To: plan.Endpoint{},
		Kind: plan.TransferPeer})
	if err != nil {
		t.Fatal(err)
	}
	want := pools.prefill.Kernel().PDTransferDemand(tokens, prefillAt, decodeAt)
	if len(got.Links) != len(want.Links) || got.Bytes != want.Bytes || got.Floor != want.Floor {
		t.Fatalf("the pull from prefill {0,0} to decode {0,0} demands %+v, want the kernel's demand between the pools' places %+v", got, want)
	}
	for i := range want.Links {
		if got.Links[i] != want.Links[i] {
			t.Errorf("link %d of the pull is %+v, want %+v (prefill at %+v, decode at %+v)", i, got.Links[i], want.Links[i], prefillAt, decodeAt)
		}
	}
}
