// Original to blis-vllm.

package kernelmodel

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// decodeEngine is a pool's engine block as vLLM's engine configuration (engineconfig.Decode),
// strictly. This package decodes each pool's block once, at open, and every reading of the block
// it makes is a reading of that decoding.
//
// TEMPORARY: blis-schemas#102 at ddfd45b still types Pool.Engine as its own struct, so the block
// is re-encoded to a node and held as a deployment.EngineBlock first; when #102's switch makes
// Pool.Engine the EngineBlock, this decodes it directly.
func decodeEngine(pool deployment.Pool) (engineconfig.Engine, error) {
	var n yaml.Node
	if err := n.Encode(pool.Engine); err != nil {
		return engineconfig.Engine{}, fmt.Errorf("kernelmodel: encoding the %s pool's engine block: %w", pool.Role, err)
	}
	e, err := engineconfig.Decode(deployment.NewEngineBlock(&n))
	if err != nil {
		return engineconfig.Engine{}, fmt.Errorf("kernelmodel: the %s pool's engine block: %w", pool.Role, err)
	}
	return e, nil
}

// DecodeEngine is a pool's engine block as vLLM's engine configuration, the decoding this
// package reads every block with (engineconfig.Decode, strictly), for a caller that edits a
// scenario's engine settings.
func DecodeEngine(pool deployment.Pool) (engineconfig.Engine, error) { return decodeEngine(pool) }

// SetEngine writes e as the pool's engine block (engineconfig.Encode).
//
// TEMPORARY: until blis-schemas#102's switch makes Pool.Engine the opaque EngineBlock, the encoded
// block is decoded back into the pool's struct.
func SetEngine(pool *deployment.Pool, e engineconfig.Engine) error {
	b, err := engineconfig.Encode(e)
	if err != nil {
		return err
	}
	var stated deployment.Engine
	if err := b.Decode(&stated); err != nil {
		return fmt.Errorf("kernelmodel: the %s pool's engine block: %w", pool.Role, err)
	}
	pool.Engine = stated
	return nil
}
