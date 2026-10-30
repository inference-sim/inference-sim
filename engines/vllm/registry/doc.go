// Original to blis-vllm.

// Package registry holds the vLLM entries of the KV-connector registry (blis-schemas
// spec/connector, inference-sim/inference-sim#2021 B10): which connector names a deployment
// may use, and which engine model in this module reproduces each. An entry earns its
// behaves_like only by passing this module's differential tests against vLLM v0.32.0;
// TestEveryEntryIsEarnedByADiffTest enforces that, so a connector without a model is not
// registered, and a model without an entry fails the test.
package registry

import (
	"bytes"
	_ "embed"
	"fmt"

	"github.com/inference-sim/blis-schemas/spec/connector"
	"gopkg.in/yaml.v3"
)

// raw is the registry file.
//
//go:embed connectors.yaml
var raw []byte

// model is the evidence behind one behaves_like name: the package whose differential test
// proves it, the committed scenarios that test runs (a glob, relative to the module root), the
// connector names those scenarios run, and the oracle tier of their golden traces.
type model struct {
	pkg, scenarios string
	connectors     []string
	tier           int
}

// models maps each behaves_like name to its evidence.
var models = map[string]model{
	"nixl-pull": {pkg: "core/connector/nixl", scenarios: "oracle/scenarios/connector/nixl-*.json",
		connectors: []string{"NixlConnector", "NixlPullConnector"}, tier: 4},
	"offloading-cpu": {pkg: "core/connector/offloading", scenarios: "oracle/scenarios/offloading/offloading-*.json",
		connectors: []string{"OffloadingConnector"}, tier: 4},
}

// Load returns the registry, validated against blis-schemas' connector-registry schema.
func Load() (*connector.Registry, error) { return decode(raw) }

// decode reads a registry as blis-schemas' loaders do, refusing fields the schema does not
// declare, and validates it.
func decode(data []byte) (*connector.Registry, error) {
	var r connector.Registry
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(&r); err != nil {
		return nil, fmt.Errorf("connector registry: %w", err)
	}
	if p := r.Validate(); !p.OK() {
		return nil, fmt.Errorf("connector registry: %s", p.Error())
	}
	for _, name := range r.Names() {
		if u := r.Connectors[name].Upstream; !upstreams[u] {
			return nil, fmt.Errorf("connector registry: connectors.%s.upstream: %q is not a known connector interface (%s)",
				name, u, UpstreamKVConnectorBaseV1)
		}
	}
	return &r, nil
}

// UpstreamKVConnectorBaseV1 is vLLM's connector interface, KVConnectorBase_V1
// (vllm/distributed/kv_transfer/kv_connector/v1/base.py:180@5df63a72), which every built-in
// connector and every connector loaded through kv_connector_module_path implements. It moved
// from blis-schemas' spec/connector at 2c38342, which now checks only that an entry names one.
const UpstreamKVConnectorBaseV1 connector.Upstream = "KVConnectorBase_V1"

// upstreams are the connector interfaces vLLM v0.32.0 defines.
var upstreams = map[connector.Upstream]bool{UpstreamKVConnectorBaseV1: true}
