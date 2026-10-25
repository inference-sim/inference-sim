// Original to blis-vllm.

package engine

import "testing"

// A replica renders on the stated API servers, else on vLLM's default for an internal or hybrid
// replica endpoint: one per rank.
func TestReplicaAPIServers(t *testing.T) {
	for _, c := range [][3]int{{0, 4, 4}, {0, 2, 2}, {3, 4, 3}, {1, 8, 1}} {
		if got := ReplicaAPIServers(c[0], c[1]); got != c[2] {
			t.Errorf("ReplicaAPIServers(stated %d, dp %d) = %d, want %d", c[0], c[1], got, c[2])
		}
	}
}
