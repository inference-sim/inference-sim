// Original to blis-vllm.

package blisvllm

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
)

// fields/stand_in_vectors.json is the truth table of the stand-ins, whose rows come from vLLM's own
// functions (fields/test_stand_in.py regenerates it from the pinned source and compares). The Go
// evaluator that consumers call, fields.Exact, must give each row's answer from the committed field list.
func TestStandInVectorsAgreeWithTheGoEvaluator(t *testing.T) {
	raw, err := os.ReadFile("fields/stand_in_vectors.json")
	if err != nil {
		t.Fatal(err)
	}
	var rows []struct {
		Field             string `json:"field"`
		Value             string `json:"value"`
		TPGroupNodes      *int   `json:"tp_group_nodes"`
		ComputeCapability string `json:"compute_capability"`
		Exact             bool   `json:"exact"`
	}
	if err := json.Unmarshal(raw, &rows); err != nil {
		t.Fatal(err)
	}
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]fields.Field{}
	for _, f := range l.Fields {
		byName[f.Name] = f
	}
	if len(rows) == 0 {
		t.Fatal("no vectors")
	}
	for _, r := range rows {
		f, ok := byName[r.Field]
		if !ok {
			t.Fatalf("%s: not in the field list", r.Field)
		}
		got := fields.Exact(f.Estimation, fields.Facts{Value: r.Value, TPGroupNodes: r.TPGroupNodes, ComputeCapability: r.ComputeCapability})
		if got != r.Exact {
			nodes := -1
			if r.TPGroupNodes != nil {
				nodes = *r.TPGroupNodes
			}
			t.Errorf("%s=%q on %d node(s), cc %q: Exact = %v, the upstream's resolution says %v", r.Field, r.Value, nodes, r.ComputeCapability, got, r.Exact)
		}
	}
}
