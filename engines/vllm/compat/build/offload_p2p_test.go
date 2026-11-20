// Original to blis-vllm.

package build

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"

	"github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// An offload block whose connector options state only vLLM's p2p tier resolves to its CPU tier:
// the v0.33.0 engine builds the p2p tier itself (blis-vllm#204), so the offload the simulator
// prices is the CPU tier the block states. Before, any connector option was refused, so a p2p
// deployment could not run. A block stating any other option alongside is still refused, naming it.
func TestAP2PTierOnlyConnectorOptionsResolveToTheCPUTier(t *testing.T) {
	cpu := []deployment.Tier{{Device: OffloadCPUTier, Bytes: 1 << 34}}
	p2p := []map[string]any{{"type": "p2p", "port": 5710}}
	o := &deployment.Offload{Spec: "TieringOffloadingSpec", Tiers: cpu,
		ConnectorOptions: connectorOptions(engineconfig.ConnectorOptions{SecondaryTiers: p2p})}
	cfg, err := ResolveKVOffloadConfig("s.yaml", o, "", 16, nil)
	if err != nil {
		t.Fatalf("a p2p-only connector options block: %v", err)
	}
	if want, _ := ResolveKVOffloadConfig("s.yaml", &deployment.Offload{Spec: "TieringOffloadingSpec", Tiers: cpu}, "", 16, nil); OffloadConfigDiff(cfg, want) != "" {
		t.Errorf("the p2p deployment's offload differs from its CPU tier's: %s", OffloadConfigDiff(cfg, want))
	}
	o.ConnectorOptions = connectorOptions(engineconfig.ConnectorOptions{SecondaryTiers: p2p, BlocksPerChunk: 16})
	if _, err := KVOffloadBlockFromDeployment(o); err == nil || !strings.Contains(err.Error(), "offload.connector_options (blocks_per_chunk)") {
		t.Errorf("p2p with blocks_per_chunk: err = %v, want the blocks_per_chunk refusal", err)
	}
}

// A scenario stating vLLM's p2p tier adopts: the v0.33.0 engine builds the tier.
func TestAdopt_AdoptsTheP2PTier(t *testing.T) {
	repos := vendoredRepos(t)
	raw, err := os.ReadFile(filepath.Join(repos.Scenarios, "glm-5-h200-tp8-mtp3.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const offload = "offload:\n  connector: OffloadingConnector\n  spec: TieringOffloadingSpec\n" +
		"  tiers:\n    - {tier: cpu_dram, bytes: 17179869184}\n" +
		"  connector_options:\n    secondary_tiers: [{type: p2p, port: 5710}]\n"
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "v.yaml"), []byte(string(raw)+offload), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Adopt(AdoptInputs{Scenario: "v.yaml", ScenariosDir: dir, Data: repos.Data}); err != nil {
		t.Errorf("a p2p tier: %v, want it adopted to the v0.33.0 engine", err)
	}
}

// p2pPDVariant is the GLM-5 P/D fixture handing off as vLLM's P/D deployment over its p2p tier
// does: OffloadingConnector as the only KV connector, and a p2p tier in its connector options
// (p2p false leaves the tier out).
func p2pPDVariant(t *testing.T, p2p bool) kernelmodel.Repos {
	t.Helper()
	repos := vendoredRepos(t)
	raw, err := os.ReadFile(filepath.Join(repos.Scenarios, "glm-5-h200-3p1d-ib.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const nixl = "pd_transfer:\n  connector: NixlConnector\n"
	if !strings.Contains(string(raw), nixl) {
		t.Fatal("the fixture's pd_transfer block changed")
	}
	offload := "offload:\n  connector: OffloadingConnector\n  spec: TieringOffloadingSpec\n" +
		"  tiers:\n    - {tier: cpu_dram, bytes: 549755813888}\n"
	if p2p {
		offload += "  connector_options:\n    secondary_tiers: [{type: p2p, port: 5710}]\n"
	}
	body := strings.Replace(string(raw), nixl, "pd_transfer:\n  connector: OffloadingConnector\n"+offload, 1)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	repos.Scenarios = dir
	return repos
}

// A P/D scenario that hands off through vLLM's p2p tier (pd_transfer OffloadingConnector, a p2p
// tier in the offload's connector options) opens both pools: the decode engine loads the prompt
// through its own tier, so NIXL's rules do not apply, the prefill engines hold no KV for a NIXL
// pull, there is no MultiConnector warning, and its CPU tier is priced. The rules every P/D run keeps still hold (the
// instance counts must add up). Before, the connector was refused as not NIXL's. An
// OffloadingConnector handoff with no p2p tier is still refused.
func TestOpenPools_OpensAP2PHandoff(t *testing.T) {
	top := Topology{NumInstances: 4, Prefill: 3, Decode: 1}
	p, warnings, err := OpenPools("variant.yaml", p2pPDVariant(t, true), true, top)
	if err != nil || p == nil {
		t.Fatalf("a p2p handoff: %+v, %v; want both pools open", p, err)
	}
	if !p.P2P || p.HoldsPrefillKV() || len(warnings) != 0 {
		t.Errorf("a p2p handoff: P2P %t, HoldsPrefillKV %t, warnings %q; want a p2p handoff, no NIXL hold and no warning", p.P2P, p.HoldsPrefillKV(), warnings)
	}
	// Its CPU tier is priced per pool, as a MultiConnector's is: the OffloadingConnector is the
	// handoff's connector and the tier's.
	pre, dec, ok, err := p.PriceOffload("variant.yaml", engine.KVOffloadConfig{}, OffloadPricing{BlockSizeTokens: 64, KVCPUBlocks: 10}, top, noFatal(t))
	if err != nil || !ok || pre.TicksPerBlock <= 0 || dec.TicksPerBlock <= 0 {
		t.Errorf("a p2p handoff's CPU tier: %+v %+v %v %v; want it priced in both pools", pre, dec, ok, err)
	}
	if _, _, err := OpenPools("variant.yaml", p2pPDVariant(t, true), true, Topology{NumInstances: 5, Prefill: 3, Decode: 1}); err == nil ||
		!strings.Contains(err.Error(), "must add up") {
		t.Errorf("a p2p handoff whose counts do not add up: %v, want the refusal", err)
	}
	if _, _, err := OpenPools("variant.yaml", p2pPDVariant(t, false), true, top); err == nil ||
		!strings.Contains(err.Error(), "BLIS models NIXL's handoff") {
		t.Errorf("an OffloadingConnector handoff without a p2p tier: %v, want the NIXL refusal", err)
	}
}

// A tier on the catalog's object store (s3) resolves as a secondary tier on its device, which the
// v0.33.0 engine runs as vLLM's obj tier; before, it was refused as "BLIS simulates the fs tier
// only". Adopt takes it as it does any tier a 0.33.0 deployment states.
func TestAnObjectStoreTierResolvesAndAdopts(t *testing.T) {
	o := &deployment.Offload{Spec: "TieringOffloadingSpec", Tiers: []deployment.Tier{
		{Device: OffloadCPUTier, Bytes: 1 << 34}, {Device: "s3", Bytes: 1 << 40}}}
	block, err := KVOffloadBlockFromDeployment(o)
	if err != nil {
		t.Fatalf("an s3 tier: %v", err)
	}
	if len(block.SecondaryTiers) != 1 || block.SecondaryTiers[0].DeviceClass == nil || *block.SecondaryTiers[0].DeviceClass != "s3" {
		t.Errorf("an s3 tier: secondary tiers %+v, want the one on s3", block.SecondaryTiers)
	}
	repos := vendoredRepos(t)
	raw, err := os.ReadFile(filepath.Join(repos.Scenarios, "glm-5-h200-tp8-mtp3.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const offload = "offload:\n  connector: OffloadingConnector\n  spec: TieringOffloadingSpec\n" +
		"  tiers:\n    - {tier: cpu_dram, bytes: 17179869184}\n    - {tier: s3, bytes: 1099511627776}\n"
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "v.yaml"), []byte(string(raw)+offload), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Adopt(AdoptInputs{Scenario: "v.yaml", ScenariosDir: dir, Data: repos.Data}); err != nil {
		t.Errorf("an s3 tier: %v, want it adopted to the v0.33.0 engine", err)
	}
}

// A p2p handoff is the OffloadingConnector alone on both pools: a decode pool that states its own
// NixlConnector beside a prefill pool's p2p tier is refused, naming the connector it runs.
func TestOpenPools_RefusesAP2PHandoffWhoseDecodePoolRunsAnotherConnector(t *testing.T) {
	repos := p2pPDVariant(t, true)
	body := readScenario(t, repos)
	marker := "  - role: decode\n"
	if strings.Count(body, marker) != 1 {
		t.Fatal("fixture has no single decode pool")
	}
	writeScenario(t, repos, strings.Replace(body, marker, marker+"    pd_transfer: {connector: NixlConnector}\n", 1))
	_, _, err := OpenPools("variant.yaml", repos, true, Topology{NumInstances: 4, Prefill: 3, Decode: 1})
	if err == nil || !strings.Contains(err.Error(), "p2p tier") || !strings.Contains(err.Error(), "decode pool must run too") ||
		!strings.Contains(err.Error(), "NixlConnector") {
		t.Errorf("a p2p handoff with a NIXL decode pool: %v, want the refusal naming the decode pool's connector", err)
	}
}
