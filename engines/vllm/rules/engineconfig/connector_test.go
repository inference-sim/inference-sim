// Moved from inference-sim/blis-schemas/spec/deployment/speculative_offload_test.go at 2c38342bcf3489b567befceebab05a7515babb40
// (TestOffloadConnectorOptions).

package engineconfig

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"gopkg.in/yaml.v3"
)

// The offload connector's options, as kv_transfer_config.kv_connector_extra_config gives
// them, typed rather than a free map so a misspelled option fails to decode.
func TestOffloadConnectorOptions(t *testing.T) {
	for _, c := range []struct {
		name string
		o    *ConnectorOptions
		bad  string
	}{
		{"none stated", nil, ""},
		{"mooncake async", &ConnectorOptions{LoadAsync: ptrBool(true), LookupAsync: ptrBool(true)}, ""},
		{"cross-layer blocks", &ConnectorOptions{EnableCrossLayersBlocks: ptrBool(true)}, ""},
		{"lazy offload", &ConnectorOptions{LazyOffload: ptrBool(true)}, ""},
		{"blocks per chunk", &ConnectorOptions{BlocksPerChunk: 4}, ""},
		{"blocks per chunk negative", &ConnectorOptions{BlocksPerChunk: -1}, "offload.connector_options.blocks_per_chunk"},
		{"recompute on failure", &ConnectorOptions{KVLoadFailurePolicy: "recompute"}, ""},
		{"an unknown failure policy", &ConnectorOptions{KVLoadFailurePolicy: "retry"}, "offload.connector_options.kv_load_failure_policy"},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := &rules.Problems{}
			ValidateConnectorOptions(p, "offload", c.o)
			if c.bad == "" {
				if !p.OK() {
					t.Fatalf("refused:\n%s", p.Error())
				}
				return
			}
			if p.OK() || !strings.Contains(p.Error(), c.bad) {
				t.Fatalf("want a finding on %s, got:\n%s", c.bad, p.Error())
			}
		})
	}
	// Strict decoding: a misspelled option is a decode error, not a silently dropped setting.
	var n yaml.Node
	if err := yaml.Unmarshal([]byte("load_asynch: true\n"), &n); err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeConnectorOptions(deployment.NewEngineBlock(n.Content[0])); err == nil ||
		!strings.Contains(err.Error(), "load_asynch") {
		t.Errorf("a misspelled connector option decoded: %v", err)
	}
	// Absent options are nil, and Encode is Decode's inverse.
	if c, err := DecodeConnectorOptions(deployment.EngineBlock{}); c != nil || err != nil {
		t.Errorf("absent options = %+v, %v", c, err)
	}
	want := &ConnectorOptions{LazyOffload: ptrBool(true), BlocksPerChunk: 4}
	b, err := EncodeConnectorOptions(want)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := DecodeConnectorOptions(b); err != nil || got.BlocksPerChunk != 4 || got.LazyOffload == nil || !*got.LazyOffload {
		t.Errorf("round trip = %+v, %v", got, err)
	}
}
