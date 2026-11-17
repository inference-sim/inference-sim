package vllmengine

import (
	"os"
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
	if !ok || !strings.HasSuffix(p.PeerAddr(), ":5710") {
		t.Fatalf("the engine is not peered at port 5710 (Peered %v, %q)", ok, p.PeerAddr())
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
