// Original to blis-vllm.

package pack_test

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-vllm/rules/v0_32"
)

// CheckConnectorOptions reports an offload's connector options at the paths and in the words
// blis-schemas used before it held them unopened, and a key the options do not take at its
// line.
func TestCheckConnectorOptions(t *testing.T) {
	block := func(doc string) deployment.EngineBlock {
		var n yaml.Node
		if err := yaml.Unmarshal([]byte(doc), &n); err != nil {
			t.Fatal(err)
		}
		return deployment.NewEngineBlock(n.Content[0])
	}
	p := v0_32.Pack()
	got := p.CheckConnectorOptions("offload", block("blocks_per_chunk: -1\nkv_load_failure_policy: retry\n")).Error()
	for _, want := range []string{
		`error: offload.connector_options.blocks_per_chunk: must not be negative; omit it, or 0, for the connector's default`,
		`error: offload.connector_options.kv_load_failure_policy: "retry" is not recompute or fail`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("findings:\n%s\nwant %q", got, want)
		}
	}
	if e := p.CheckConnectorOptions("offload", block("lazy_offload: true\nload_asynch: true\n")).Error(); !strings.Contains(e, "offload.connector_options") || !strings.Contains(e, "line 2: field load_asynch not found") {
		t.Errorf("an unknown option: %s", e)
	}
	if pr := p.CheckConnectorOptions("offload", block("lazy_offload: true\nblocks_per_chunk: 4\n")); !pr.OK() || len(pr.All()) != 0 {
		t.Errorf("valid options were refused:\n%s", pr.Error())
	}
}
