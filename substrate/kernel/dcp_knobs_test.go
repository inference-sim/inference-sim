package latencykernel

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts"
)

// The three decode-context-parallel engine knobs blis-schemas v0.2.2 made expressible --
// dcp_comm_backend, dcp_q_replicate, cp_kv_cache_interleave_size -- priced the way vLLM
// v0.31.0 runs them. resolve.ResolveDecodeContext and dcpDecodeCollectives carry the
// citations; these pin the behaviour.

// The fabric pcpInputs adds must be loadable from the same catalog the fixtures read; this
// keeps dcpVariant's multi-node layouts honest if the catalog moves.
func TestTheFabricDCPLayoutsDeclareIsInTheCatalog(t *testing.T) {
	path := filepath.Join(catalogRoot, "networks", "ib-400g.yaml")
	if _, err := contracts.LoadFabric(path); err != nil {
		t.Fatal(err)
	}
}
