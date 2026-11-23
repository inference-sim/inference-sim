// Original to inference-sim.

package build

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
)

func mtp(k string) string {
	return "      speculative_config: {method: mtp, num_speculative_tokens: " + k + "}\n"
}

// Two pools draft alike when neither drafts (no block, or a block of no tokens) or when they state
// the same block; any other pair drafts differently, and each pool's own draft length is read.
func TestPoolsDraftDifferently_ComparesTheStatedBlocks(t *testing.T) {
	for _, c := range []struct {
		name            string
		prefill, decode string
		different       bool
		prefillK        int
		decodeK         int
	}{
		{"neither drafts", "", "", false, 0, 0},
		{"the same block", mtp("2"), mtp("2"), false, 2, 2},
		{"different lengths", mtp("1"), mtp("3"), true, 1, 3},
		{"one drafts", "", mtp("3"), true, 0, 3},
	} {
		repos := pdVariant(t, c.prefill, c.decode)
		pre, dec := openRole(t, repos, deployment.RolePrefill), openRole(t, repos, deployment.RoleDecode)
		got, err := PoolsDraftDifferently(pre, dec)
		if err != nil || got != c.different {
			t.Errorf("%s: PoolsDraftDifferently = %v, %v; want %v", c.name, got, err, c.different)
		}
		for role, want := range map[string]int{"prefill": c.prefillK, "decode": c.decodeK} {
			m := pre
			if role == "decode" {
				m = dec
			}
			d, err := PoolDrafting(m)
			if err != nil || d.Tokens != want || d.Drafts() != (want != 0) {
				t.Errorf("%s: %s pool drafts %+v, %v; want %d tokens", c.name, role, d, err, want)
			}
		}
	}
}
