package deployment

import (
	"reflect"
	"strings"
	"testing"
)

// TestParallelismFieldNames does the same for the layout, and asserts that no field
// named ep exists: the width is derived, and a field for it could disagree.
func TestParallelismFieldNames(t *testing.T) {
	got := yamlNames(reflect.TypeOf(Parallelism{}))
	want := []string{"tp", "pp", "dp", "dp_local", "data_parallel_load_balancing", "enable_expert_parallel",
		"pcp", "dcp"}
	if len(got) != len(want) {
		t.Errorf("Parallelism has %d yaml fields, want %d: %v", len(got), len(want), got)
	}
	for _, w := range want {
		if !contains(got, w) {
			t.Errorf("Parallelism is missing %q", w)
		}
	}
	for _, forbidden := range []string{"ep", "expert_parallel_size"} {
		if contains(got, forbidden) {
			t.Errorf("Parallelism declares %q; the expert-parallel width is derived so a field could disagree with it", forbidden)
		}
	}
}

func yamlNames(t reflect.Type) []string {
	var out []string
	for i := 0; i < t.NumField(); i++ {
		tag := t.Field(i).Tag.Get("yaml")
		if tag == "" || tag == "-" {
			continue
		}
		out = append(out, strings.Split(tag, ",")[0])
	}
	return out
}

func contains(list []string, want string) bool {
	for _, s := range list {
		if s == want {
			return true
		}
	}
	return false
}
