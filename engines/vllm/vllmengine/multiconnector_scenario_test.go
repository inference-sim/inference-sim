package vllmengine

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"gopkg.in/yaml.v3"
)

// multiScenario is the Muse Spark 1P1D scenario with pd_transfer.connector: MultiConnector and a
// deployment-level 512 GiB CPU tier. shippedScenario is the DeepSeek V4.1 Flash one a simulator ships
// (a copy of its testdata), which also drafts with DSpark.
const (
	multiScenario   = "muse-h100-roce-1p1d-multi.yaml"
	shippedScenario = "deepseek-v4.1-flash-h100-1p1d-dspark5-cpu1t.yaml"
)

// preparedKV is a pool's kv_transfer as the factory completes it.
type preparedKV struct {
	Connector string `yaml:"connector"`
	Role      string `yaml:"kv_role"`
	Extra     struct {
		Connectors []struct {
			Connector string `yaml:"kv_connector"`
			Role      string `yaml:"kv_role"`
			Extra     struct {
				CPUBytes int64  `yaml:"cpu_bytes_to_use"`
				Spec     string `yaml:"spec_name"`
				Eviction string `yaml:"eviction_policy"`
			} `yaml:"kv_connector_extra_config"`
		} `yaml:"connectors"`
	} `yaml:"kv_connector_extra_config"`
}

func prepareKV(t *testing.T, roots map[string]string, pool int) (preparedKV, kvOffload) {
	t.Helper()
	done, err := prepare(engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: roots, Pool: pool}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var b struct {
		KV  preparedKV `yaml:"kv_transfer"`
		Off kvOffload  `yaml:"kv_offload"`
	}
	if err := done.Config.Decode(&b); err != nil {
		t.Fatal(err)
	}
	return b.KV, b.Off
}

// A deployment that states pd_transfer.connector: MultiConnector and an offload tier is the pair the
// startup warning says is assumed: a NIXL child in the pool's role, then the OffloadingConnector over
// the tier. Both pools must build, and the prefill pool must hold a finished request for the decode
// pool's pull and store its prompt to the CPU tier. Before the factory wrote the children, the
// engine refused this scenario because its MultiConnector listed none.
func TestAMultiConnectorScenarioBuildsBothPoolsAndRunsItsPair(t *testing.T) {
	Register()
	r := roots(t, multiScenario)
	for i, role := range []string{"kv_producer", "kv_consumer"} {
		kt, off := prepareKV(t, r, i)
		if kt.Connector != "MultiConnector" || kt.Role != role || len(kt.Extra.Connectors) != 2 ||
			kt.Extra.Connectors[0].Connector != "NixlConnector" || kt.Extra.Connectors[0].Role != role ||
			kt.Extra.Connectors[1].Connector != "OffloadingConnector" || kt.Extra.Connectors[1].Role != "kv_both" ||
			kt.Extra.Connectors[1].Extra.CPUBytes != 512<<30 {
			t.Errorf("pool %d: kv_transfer %+v, want MultiConnector as %s over NixlConnector then OffloadingConnector with the 512 GiB tier", i, kt, role)
		}
		if off.CPUBytes != 0 || off.BytesPerBlock == 0 {
			t.Errorf("pool %d: kv_offload %+v, want only what vLLM derives from the model", i, off)
		}
		if _, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: r, Pool: i}); err != nil {
			t.Fatalf("pool %d: %v", i, err)
		}
	}
	p, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: r, Pool: 0})
	if err != nil {
		t.Fatal(err)
	}
	rq := req("a", 4096, 8)
	rq.RemoteDecode, rq.MaxOutputTokens = true, 1
	if err := p.Add(rq, 0); err != nil {
		t.Fatal(err)
	}
	now := engine.Time(0)
	var held []engine.Handoff
	stores := 0
	for step := 0; step < 64 && len(held) == 0; step++ {
		b, err := p.Schedule(now)
		if err != nil || b.Empty() {
			t.Fatalf("prefill schedule: %v, empty %v", err, b.Empty())
		}
		for _, tr := range b.TransfersStarted {
			if tr.Kind == engine.TransferStore && tr.Tier == "cpu" {
				stores++
			}
		}
		now += p.StepTime(b)
		res, err := p.Complete(b.ID, engine.Outcome{}, now)
		if err != nil {
			t.Fatal(err)
		}
		held = res.Handoffs
	}
	if len(held) != 1 || len(held[0].Blocks) == 0 {
		t.Fatalf("handoffs %+v, want a's blocks held for the decoder", held)
	}
	if stores == 0 {
		t.Error("the prefill pool stored nothing to the CPU tier")
	}
}

// A host that must state the order the schema cannot hands the engine its pool's block with a kv_transfer
// that lists the OffloadingConnector first. The factory keeps it and adds the deployment's tier to the
// offloading child, so the deployment's offload block is still the one place the tier is stated.
func TestAHostStatesTheMultiConnectorOrderInItsEngineBlock(t *testing.T) {
	Register()
	var given yaml.Node
	if err := yaml.Unmarshal([]byte("kv_transfer:\n  connector: MultiConnector\n  kv_role: kv_consumer\n"+
		"  kv_connector_extra_config:\n    connectors:\n"+
		"      - {kv_connector: OffloadingConnector, kv_role: kv_both}\n"+
		"      - {kv_connector: NixlConnector, kv_role: kv_consumer}\n"), &given); err != nil {
		t.Fatal(err)
	}
	given = *given.Content[0]
	engineBlock := poolEngine(t, multiScenario)
	given.Content = append(given.Content, engineBlock.Content...)
	spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: roots(t, multiScenario), Pool: 1, Config: given}
	done, err := prepare(spec, nil)
	if err != nil {
		t.Fatal(err)
	}
	var b struct {
		KV  preparedKV `yaml:"kv_transfer"`
		Off kvOffload  `yaml:"kv_offload"`
	}
	if err := done.Config.Decode(&b); err != nil {
		t.Fatal(err)
	}
	kt, off := b.KV, b.Off
	if len(kt.Extra.Connectors) != 2 || kt.Extra.Connectors[0].Connector != "OffloadingConnector" ||
		kt.Extra.Connectors[1].Connector != "NixlConnector" {
		t.Fatalf("kv_transfer %+v, want the order the engine block states", kt)
	}
	if got := kt.Extra.Connectors[0].Extra; got.CPUBytes != 512<<30 || got.Eviction != "lru" {
		t.Errorf("the offloading child's tier %+v, want the deployment's 512 GiB and lru", got)
	}
	if off.CPUBytes != 0 {
		t.Errorf("kv_offload %+v, want no vLLM keys beside a MultiConnector", off)
	}
	if _, err := engine.New(spec); err != nil {
		t.Fatal(err)
	}
}

// The shipped scenario (DeepSeek V4.1 Flash, 1P1D, a MultiConnector over NIXL and a 1 TiB CPU tier)
// drafts with DSpark and caches in packed groups: eight 32-token window groups, the 64-token
// compressed cache and an 8-token ring. vLLM offloads the prefix-cacheable group alone
// (get_offloading_group_ids), so both pools build, and the prefill pool stores a prompt's compressed
// blocks to the CPU tier while holding the request for the decoder's pull. Before, the engine gave
// the connector all ten groups and refused them against the 64-token hash block size
// (blis-vllm#132, #146).
func TestTheShippedMultiConnectorScenarioBuildsAndOffloadsItsCacheableGroup(t *testing.T) {
	Register()
	r := roots(t, shippedScenario)
	for pool := range 2 {
		if _, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: r, Pool: pool}); err != nil {
			t.Fatalf("pool %d: %v", pool, err)
		}
	}
	p, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: r, Pool: 0})
	if err != nil {
		t.Fatal(err)
	}
	rq := req("a", 4096, 8)
	rq.RemoteDecode, rq.MaxOutputTokens = true, 1
	if err := p.Add(rq, 0); err != nil {
		t.Fatal(err)
	}
	now := engine.Time(0)
	stored := 0
	var held []engine.Handoff
	for step := 0; step < 200 && len(held) == 0; step++ {
		b, err := p.Schedule(now)
		if err != nil || b.Empty() {
			t.Fatalf("prefill schedule: %v, empty %v", err, b.Empty())
		}
		for _, tr := range b.TransfersStarted {
			if tr.Kind == engine.TransferStore && tr.Tier == "cpu" {
				stored += tr.Tokens
			}
		}
		now += p.StepTime(b)
		res, err := p.Complete(b.ID, engine.Outcome{}, now)
		if err != nil {
			t.Fatal(err)
		}
		held = res.Handoffs
	}
	if len(held) != 1 {
		t.Fatalf("handoffs %+v, want a's blocks held for the decoder", held)
	}
	if stored == 0 || stored%64 != 0 {
		t.Errorf("the prefill pool stored %d tokens to the CPU tier, want whole 64-token blocks of the cacheable group", stored)
	}
}
