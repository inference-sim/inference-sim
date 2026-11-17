package vllmengine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// p2pScenario is tieredScenario with the offload's connector options, in place of a device tier.
func p2pScenario(t *testing.T, options string) map[string]string {
	t.Helper()
	r := tieredScenario(t, "")
	raw, err := os.ReadFile(r[RootScenario])
	if err != nil {
		t.Fatal(err)
	}
	s := strings.Replace(string(raw), "  spec: TieringOffloadingSpec\n", "  spec: TieringOffloadingSpec\n  connector_options:\n"+options, 1)
	if err := os.WriteFile(r[RootScenario], []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
	return r
}

// A deployment states vLLM's p2p tier in its offload's connector options, as vLLM states it in
// kv_connector_extra_config.secondary_tiers, since a p2p tier sits on no catalog device. The engine
// it builds has the tier: it has a peer address. Before, any connector option other than
// blocks_per_chunk was refused.
func TestAP2PTierStatedInTheConnectorOptionsBuildsAPeeredEngine(t *testing.T) {
	Register()
	spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0",
		Roots: p2pScenario(t, "    secondary_tiers: [{type: p2p, port: 5710}]\n")}
	done, err := prepare(spec, nil)
	if err != nil {
		t.Fatal(err)
	}
	var b struct {
		O struct {
			Tiers []map[string]any `yaml:"secondary_tiers"`
		} `yaml:"kv_offload"`
	}
	if err := done.Config.Decode(&b); err != nil {
		t.Fatal(err)
	}
	if len(b.O.Tiers) != 1 || b.O.Tiers[0]["type"] != "p2p" || b.O.Tiers[0]["port"] != 5710 {
		t.Fatalf("kv_offload secondary_tiers %v, want the one p2p tier on port 5710", b.O.Tiers)
	}
	e, err := engine.New(spec)
	if err != nil {
		t.Fatal(err)
	}
	p, ok := e.(engine.Peered)
	if !ok {
		t.Fatal("the engine is not peered")
	}
	if !strings.HasSuffix(p.PeerAddr(), ":5710") {
		t.Fatalf("the engine is peered at %q, want port 5710", p.PeerAddr())
	}
}

// What the connector options may not state is refused by name: a tier on a device, which is one of
// the offload's tiers; another connector's option; and a key no connector has.
func TestAConnectorOptionTierThatIsNotP2PIsRefused(t *testing.T) {
	Register()
	for name, c := range map[string]struct{ options, want string }{
		"a device tier":              {"    secondary_tiers: [{type: fs, root_dir: /kv}]\n", "only p2p"},
		"another connector's option": {"    load_async: true\n", "only blocks_per_chunk and secondary_tiers"},
		"an unknown key":             {"    p2p_port: 5710\n", "p2p_port not found"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := prepare(engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: p2pScenario(t, c.options)}, nil)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("refusal %v, want one naming %q", err, c.want)
			}
		})
	}
}

// A P/D deployment over vLLM's p2p tier states the OffloadingConnector as its only KV connector
// (docs/features/kv_offloading_usage.md "P2P (Including P/D)" at vLLM v0.33.0). Each pool builds a
// peered engine whose offload is that connector, with the p2p tier, and no kv_transfer. Before, the
// factory stated the OffloadingConnector as a kv_transfer beside kv_offload, which the engine refuses.
func TestAP2PHandoffDeploymentBuildsBothPoolsPeered(t *testing.T) {
	Register()
	raw, err := os.ReadFile(roots(t, multiScenario)[RootScenario])
	if err != nil {
		t.Fatal(err)
	}
	s := strings.Replace(string(raw), "pd_transfer:\n  connector: MultiConnector\n", "pd_transfer:\n  connector: OffloadingConnector\n", 1)
	s = strings.Replace(s, "  eviction_policy: lru\n", "  eviction_policy: lru\n  spec: TieringOffloadingSpec\n"+
		"  connector_options:\n    secondary_tiers: [{type: p2p, port: 5710}]\n", 1)
	if !strings.Contains(s, "connector: OffloadingConnector\n\n") || !strings.Contains(s, "type: p2p") {
		t.Fatal("fixture edit failed")
	}
	path := filepath.Join(t.TempDir(), "p2p.yaml")
	if err := os.WriteFile(path, []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
	r := map[string]string{RootScenario: path}
	for pool := range 2 {
		kt, off := prepareKV(t, r, pool)
		if kt.Connector != "" || off.CPUBytes != 512<<30 {
			t.Errorf("pool %d: kv_transfer %+v, kv_offload CPU bytes %d; want no kv_transfer and the 512 GiB tier", pool, kt, off.CPUBytes)
		}
		e, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: r, Pool: pool})
		if err != nil {
			t.Fatalf("pool %d: %v", pool, err)
		}
		if p, ok := e.(engine.Peered); !ok {
			t.Errorf("pool %d: the engine has no p2p tier", pool)
		} else if p.PeerAddr() == "" {
			t.Errorf("pool %d: the engine's p2p tier has no address", pool)
		}
	}
}

// A deployment's tier on the catalog's object store (s3) is vLLM's obj tier in a bucket named by the
// device, not an fs tier, and the engine builds it: it takes its bucket through ShareTierBuckets,
// with a host's own types. Before, the factory built it as an fs tier with root_dir /s3.
func TestATierOnAnObjectStoreIsAnObjTier(t *testing.T) {
	Register()
	spec := engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: tieredScenario(t, "    - {tier: s3, bytes: 1099511627776}\n")}
	done, err := prepare(spec, nil)
	if err != nil {
		t.Fatal(err)
	}
	var b struct {
		O struct {
			Tiers []map[string]any `yaml:"secondary_tiers"`
		} `yaml:"kv_offload"`
	}
	if err := done.Config.Decode(&b); err != nil {
		t.Fatal(err)
	}
	if len(b.O.Tiers) != 1 || b.O.Tiers[0]["type"] != "obj" || b.O.Tiers[0]["device_class"] != "s3" {
		t.Fatalf("kv_offload secondary_tiers %v, want one obj tier on s3", b.O.Tiers)
	}
	if store, _ := b.O.Tiers[0]["store_config"].(map[string]any); store["bucket"] != "s3" {
		t.Errorf("store_config %v, want the bucket named by the device", b.O.Tiers[0]["store_config"])
	}
	e, err := engine.New(spec)
	if err != nil {
		t.Fatal(err)
	}
	type hostBucket = interface {
		Exists(key string) bool
		Write(keys []string) bool
		Read(keys []string) bool
	}
	if _, ok := e.(interface {
		ShareTierBuckets(func(bucket string) hostBucket)
	}); !ok {
		t.Error("the engine takes no bucket a host can give it with its own types")
	}
}
