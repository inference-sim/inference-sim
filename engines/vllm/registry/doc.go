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

// models maps each behaves_like name to the package whose differential test proves it.
var models = map[string]string{
	"nixl-pull": "core/connector/nixl", // relative to the module root
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
	return &r, nil
}
