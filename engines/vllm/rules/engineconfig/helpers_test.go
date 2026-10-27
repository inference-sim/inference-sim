// Original to blis-vllm.

package engineconfig

import "github.com/inference-sim/blis-schemas/rules"

// engineOnly is what the moved tests validated: one pool's engine block, at pools[0], as
// blis-schemas' deployment validation reported it for a one-pool deployment.
func engineOnly(e Engine) validated { return validated{e: e} }

type validated struct{ e Engine }

// Validate returns the block's field findings, as Deployment.Validate did for its engine.
func (v validated) Validate() *rules.Problems {
	p := &rules.Problems{}
	Validate(p, "pools[0]", v.e)
	return p
}

func ptrInt(v int) *int           { return &v }
func ptrInt64(v int64) *int64     { return &v }
func ptrFloat(v float64) *float64 { return &v }
func ptrBool(v bool) *bool        { return &v }
