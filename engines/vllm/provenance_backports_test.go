// Original to blis-vllm.

package blisvllm

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/engines/vllm/internal/upstream"
)

// A build with no backport states none: the field is omitted from the provenance, so a record that
// predates the field reads the same; one with backports lists their commits in the record's order.
func TestBackportCommitsAreStatedInTheProvenance(t *testing.T) {
	tests := []struct {
		name string
		bs   []upstream.Backport
		want []string
	}{
		{"none", nil, nil},
		{"two, in order", []upstream.Backport{{Commit: "b"}, {Commit: "a"}}, []string{"b", "a"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := provenanceFor("x", tc.bs)
			if !reflect.DeepEqual(p.Backports, tc.want) {
				t.Fatalf("Backports = %v, want %v", p.Backports, tc.want)
			}
			raw, err := json.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			var m map[string]any
			if err := json.Unmarshal(raw, &m); err != nil {
				t.Fatal(err)
			}
			if _, has := m["backports"]; has != (len(tc.want) > 0) {
				t.Errorf("backports present in the JSON = %v with %d commits", has, len(tc.want))
			}
		})
	}
}

// The build's provenance carries the record's backports.
func TestProvenanceForCarriesTheRecordedBackports(t *testing.T) {
	if got, want := ProvenanceFor("x").Backports, backportCommits(upstream.Backports()); !reflect.DeepEqual(got, want) {
		t.Errorf("ProvenanceFor(...).Backports = %v, want %v", got, want)
	}
}
