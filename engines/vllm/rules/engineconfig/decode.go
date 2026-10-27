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

// Pool is one deployment pool with its engine block opened.
type Pool struct {
	deployment.Pool
	Engine Engine
}

// Pools opens every pool's engine block of a deployment, in order. The first block that does
// not decode is the error, naming its pool.
func Pools(d *deployment.Deployment) ([]Pool, error) {
	if d == nil {
		return nil, nil
	}
	out := make([]Pool, len(d.Pools))
	for i, p := range d.Pools {
		e, err := Decode(p.Engine)
		if err != nil {
			return nil, fmt.Errorf("pools[%d].engine: %w", i, err)
		}
		out[i] = Pool{Pool: p, Engine: e}
	}
	return out, nil
}
