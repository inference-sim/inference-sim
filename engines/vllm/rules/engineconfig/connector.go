// Moved from inference-sim/blis-schemas/spec/deployment/deployment.go and validate.go at 2c38342:
// vLLM's KV connector options and their checks, which blis-schemas now holds unopened
// (deployment.Offload.ConnectorOptions, a deployment.EngineBlock).

package engineconfig

import (
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// ConnectorOptions are the KV connector options a deployment states, from vLLM's
// kv_transfer_config (vllm/config/kv_transfer.py@5df63a72) and its kv_connector_extra_config,
// each read by the connector named. Typed rather than a free map, so a misspelled option fails
// to load instead of being dropped. Every one is a tri-state request: nil is the connector's
// default.
type ConnectorOptions struct {
	// LoadAsync loads a hit asynchronously (MooncakeStoreConnector, default true:
	// vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:1397@5df63a72).
	LoadAsync *bool `yaml:"load_async,omitempty"`
	// LookupAsync looks a prefix up without blocking the scheduler (MooncakeStoreConnector,
	// default false: .../mooncake/store/scheduler.py:147@5df63a72).
	LookupAsync *bool `yaml:"lookup_async,omitempty"`
	// EnableOffload writes KV to the store as well as reading it (Mooncake's store config,
	// default false: vllm/distributed/mooncake_store.py:37,66@5df63a72).
	EnableOffload *bool `yaml:"enable_offload,omitempty"`
	// EnableCrossLayersBlocks stores one block across all layers rather than per layer
	// (MooncakeStoreConnector's layout check, .../mooncake/store/worker.py:1553@5df63a72).
	EnableCrossLayersBlocks *bool `yaml:"enable_cross_layers_blocks,omitempty"`
	// LazyOffload defers a CPU write until a block is evicted (SimpleCPUOffloadConnector,
	// default false: .../simple_cpu_offload_connector.py:104@5df63a72).
	LazyOffload *bool `yaml:"lazy_offload,omitempty"`
	// BlocksPerChunk groups GPU blocks into one offload chunk (OffloadingConnector,
	// .../offloading/config.py:99-112@5df63a72). Zero means unset; a stated one is positive.
	BlocksPerChunk int `yaml:"blocks_per_chunk,omitempty"`
	// KVLoadFailurePolicy is what a failed KV load does: "recompute" the failed blocks, or
	// "fail" the request, the default (vllm/config/kv_transfer.py:88@5df63a72).
	KVLoadFailurePolicy string `yaml:"kv_load_failure_policy,omitempty"`
}

// kvLoadFailurePolicies is vLLM v0.32.0's closed literal (vllm/config/kv_transfer.py:88@5df63a72).
var kvLoadFailurePolicies = map[string]bool{"recompute": true, "fail": true}

// DecodeConnectorOptions opens an offload's connector options strictly, as Decode opens an
// engine block. Absent options are nil.
func DecodeConnectorOptions(b deployment.EngineBlock) (*ConnectorOptions, error) {
	if b.IsZero() {
		return nil, nil
	}
	var c ConnectorOptions
	if err := b.Decode(&c); err != nil {
		return nil, err
	}
	return &c, nil
}

// EncodeConnectorOptions is DecodeConnectorOptions' inverse; nil is absent options.
func EncodeConnectorOptions(c *ConnectorOptions) (deployment.EngineBlock, error) {
	if c == nil {
		return deployment.EngineBlock{}, nil
	}
	return encodeBlock(*c)
}

// ValidateConnectorOptions is the checks blis-schemas ran on an offload's connector options,
// with the same messages and paths. at is the offload block's path, "offload".
func ValidateConnectorOptions(p *rules.Problems, at string, c *ConnectorOptions) {
	if c == nil {
		return
	}
	if c.BlocksPerChunk < 0 {
		p.Field(at+".connector_options.blocks_per_chunk", "must not be negative; omit it, or 0, for the connector's default")
	}
	if f := c.KVLoadFailurePolicy; f != "" && !kvLoadFailurePolicies[f] {
		p.Field(at+".connector_options.kv_load_failure_policy", "%q is not recompute or fail", f)
	}
}
