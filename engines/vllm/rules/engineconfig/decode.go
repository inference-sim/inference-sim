// Original to blis-vllm.

package engineconfig

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"gopkg.in/yaml.v3"
)

// Decode opens a pool's engine block as vLLM's engine configuration: strictly, so a key the
// release does not take is an error rather than dropped, reported at its line as
// blis-schemas reported it before it held the block unopened. An absent block is the zero
// Engine, every setting unstated.
func Decode(b deployment.EngineBlock) (Engine, error) {
	var e Engine
	if err := b.Decode(&e); err != nil {
		return Engine{}, err
	}
	return e, nil
}

// Encode is Decode's inverse: the engine block an Engine is written as, in the struct's field
// order with its omitempty tags, as blis-schemas encoded the block before it held it
// unopened. The zero Engine encodes as an empty mapping.
func Encode(e Engine) (deployment.EngineBlock, error) {
	var n yaml.Node
	if err := n.Encode(e); err != nil {
		return deployment.EngineBlock{}, fmt.Errorf("engineconfig: %w", err)
	}
	return deployment.NewEngineBlock(&n), nil
}
