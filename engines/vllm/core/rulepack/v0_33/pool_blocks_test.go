package v0_33

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-vllm/core/engineconfig"
)

// The spec, connector and eviction-policy rules check the blocks a pool states for itself, as they
// check the deployment's, and report the pool's field path. A registered value is quiet.
func TestPoolOwnBlocksAreCheckedByTheOffloadAndConnectorRules(t *testing.T) {
	tier := []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 40}}
	for _, c := range []struct {
		name, rule, path string
		bad, good        deployment.Pool
	}{
		{"offload spec", "offload-spec-known", "pools[0].offload.spec",
			deployment.Pool{Offload: &deployment.Offload{Spec: "MagicOffloadingSpec", Tiers: tier}},
			deployment.Pool{Offload: &deployment.Offload{Spec: "CPUOffloadingSpec", Tiers: tier}}},
		{"offload connector", "connector-known", "pools[0].offload.connector",
			deployment.Pool{Offload: &deployment.Offload{Connector: "TelepathyConnector", Tiers: tier}},
			deployment.Pool{Offload: &deployment.Offload{Connector: "OffloadingConnector", Tiers: tier}}},
		{"pd_transfer connector", "connector-known", "pools[0].pd_transfer.connector",
			deployment.Pool{PDTransfer: &deployment.PDTransfer{Connector: "CarrierPigeonConnector"}},
			deployment.Pool{PDTransfer: &deployment.PDTransfer{Connector: "NixlConnector"}}},
		{"eviction policy", "eviction-policy-known", "pools[0].offload.eviction_policy",
			deployment.Pool{Offload: &deployment.Offload{EvictionPolicy: "most_recently_used_typo", Tiers: tier}},
			deployment.Pool{Offload: &deployment.Offload{EvictionPolicy: "arc", Tiers: tier}}},
	} {
		t.Run(c.name, func(t *testing.T) {
			with := func(p deployment.Pool) *deployment.Deployment {
				base := colocated(deployment.Parallelism{TP: 8, PP: 1, DP: 1}, engineconfig.Engine{})
				return dep(deployment.Pool{Role: base.Role, Nodes: base.Nodes, Parallel: base.Parallel, Engine: base.Engine,
					Offload: p.Offload, PDTransfer: p.PDTransfer})
			}
			p := run(t, with(c.bad), moe224Graph())
			found := false
			for _, it := range p.All() {
				found = found || it.Rule == c.rule && strings.Contains(it.Message, c.path)
			}
			if !found {
				t.Errorf("no %s problem at %s: %v", c.rule, c.path, p.All())
			}
			if q := run(t, with(c.good), moe224Graph()); fired(q, c.rule) {
				t.Errorf("%s fired on a registered value: %v", c.rule, q.All())
			}
		})
	}
}
