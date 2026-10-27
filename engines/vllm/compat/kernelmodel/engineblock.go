// Original to blis-vllm.

package kernelmodel

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// decodeEngine is a pool's engine block as vLLM's engine configuration (engineconfig.Decode),
// strictly. This package decodes each pool's block once, at open, and every reading of the block
// it makes is a reading of that decoding.
func decodeEngine(pool deployment.Pool) (engineconfig.Engine, error) {
	e, err := engineconfig.Decode(pool.Engine)
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
func SetEngine(pool *deployment.Pool, e engineconfig.Engine) error {
	b, err := engineconfig.Encode(e)
	if err != nil {
		return err
	}
	pool.Engine = b
	return nil
}
