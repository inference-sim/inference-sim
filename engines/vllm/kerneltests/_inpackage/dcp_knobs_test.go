// Moved from blis-latency-kernel/dcp_knobs_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/coefficient"
)

// The three decode-context-parallel engine knobs blis-schemas v0.2.2 made expressible --
// dcp_comm_backend, dcp_q_replicate, cp_kv_cache_interleave_size -- priced the way vLLM
// v0.31.0 runs them. resolve.ResolveDecodeContext and dcpDecodeCollectives carry the
// citations; these pin the behaviour.

// dcpVariant is a committed fixture with one pool's DCP, PCP, TP and node count set, and
// any engine edits applied, laid out as the engine needs it (pcpInputs adds the nodes and
// fabric a PCP layout requires).
func dcpVariant(t *testing.T, fixture string, tp, pcp, dcp int,
	edit func(*Inputs)) (*Kernel, error) {
	t.Helper()
	in := pcpInputs(t, fixture, 1)
	pool := &in.Deployment.Pools[0]
	pool.Parallel.TP, pool.Parallel.PCP, pool.Parallel.DCP = tp, pcp, dcp
	nodes := max((tp*max(pcp, 1)+7)/8, 1)
	pool.Nodes, in.Scenario.Cluster.Nodes = nodes, nodes
	if edit != nil {
		edit(&in)
	}
	return newKernel(in)
}

// scaleFloors returns a copy of the coefficient sets with every collective floor of one
// primitive multiplied by factor. It is how a test asks which collectives a layout
// launches without reading the kernel's internals: inflate one primitive's floors and see
// which step times move.
func scaleFloors(sets []*coefficient.Set, primitive string, factor float64) []*coefficient.Set {
	prefix := "collective_floor_" + primitive + "_"
	out := make([]*coefficient.Set, len(sets))
	for i, s := range sets {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for j := range c.Coefficients {
			if strings.HasPrefix(c.Coefficients[j].Name, prefix) {
				c.Coefficients[j].Value *= factor
			}
		}
		out[i] = &c
	}
	return out
}
