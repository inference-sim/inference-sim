package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// Layouts no catalog fixture reaches, asserted on the selection directly. A 128-head DSA
// model (DeepSeek-V3.2) at tp=8 holds 16 heads a rank: at dcp=8 they gather to 128, which pads
// to a different fp8 decode size from the local 16, and FlashMLA-sparse refuses it; at dcp=4
// they gather to 64 and it starts (flashmla_sparse.py:436-460, :684-687). With PCP the gathered
// count is heads x tp when dcp spans tp x pcp: at tp4/pcp2/dcp8, 32 local heads gather to
// 128 and are refused, where pcp2/dcp2 leaves them at 32 (vLLM admits dcp only as 1, pcp or
// tp x pcp under PCP, vllm/config/parallel.py:571-577). The a2a combine is refused on
// FlashMLA-sparse (:416-422), and a dtype no backend lists has none.
func TestSparseMLABackendRefusesWhatNoBackendServes(t *testing.T) {
	starts := func(r sparseMLABackendRequest) bool {
		if r.dcpComm == "" {
			r.dcpComm = resolve.DCPAllGatherReduceScatter
		}
		_, _, err := sparseMLABackend(r)
		return err == nil
	}
	for _, c := range []struct {
		name string
		r    sparseMLABackendRequest
		want bool
	}{
		{"128 heads, tp8 dcp8", sparseMLABackendRequest{cache: "fp8", headsPerRank: 16,
			tp: 8, dcp: 8, pcp: 1}, false},
		{"128 heads, tp8 dcp4", sparseMLABackendRequest{cache: "fp8", headsPerRank: 16,
			tp: 8, dcp: 4, pcp: 1}, true},
		{"128 heads, tp4 pcp2 dcp8", sparseMLABackendRequest{cache: "fp8", headsPerRank: 32,
			tp: 4, dcp: 8, pcp: 2}, false},
		{"128 heads, tp8 pcp2 dcp2", sparseMLABackendRequest{cache: "fp8", headsPerRank: 16,
			tp: 8, dcp: 2, pcp: 2}, true},
		{"a2a combine", sparseMLABackendRequest{cache: "fp8", headsPerRank: 8, tp: 8, dcp: 2,
			pcp: 1, dcpComm: resolve.DCPAllToAll}, false},
		{"fp8_e5m2 on hopper", sparseMLABackendRequest{cache: "fp8_e5m2", headsPerRank: 8,
			tp: 8, dcp: 1, pcp: 1}, false},
	} {
		if got := starts(c.r); got != c.want {
			t.Errorf("%s: starts %v, want %v", c.name, got, c.want)
		}
	}
}
