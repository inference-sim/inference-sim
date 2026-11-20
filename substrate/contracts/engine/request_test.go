package engine

import (
	"strings"
	"testing"
)

// vLLM keys the prefix-cache blocks of a LoRA request by its adapter's name and path
// (kv_cache_utils.py _gen_lora_extra_hash_keys, v0.32.0). Two requests that differ only in
// the path must therefore be distinguishable: the field exists and is a plain string.
func TestRequestCarriesTheAdapterPathSeparatelyFromTheName(t *testing.T) {
	a := Request{ID: "a", Adapter: "sql", AdapterPath: "/loras/sql-v1"}
	b := Request{ID: "b", Adapter: "sql", AdapterPath: "/loras/sql-v2"}
	if a.Adapter != b.Adapter || a.AdapterPath == b.AdapterPath {
		t.Fatalf("name and path must be independent: %+v %+v", a, b)
	}
	for _, r := range []Request{a, b, {ID: "base"}, {ID: "named", Adapter: "sql"}} {
		if err := r.Validate(); err != nil {
			t.Errorf("request %s: %v", r.ID, err)
		}
	}
}

func TestRequestValidateRefusesAPathWithNoAdapter(t *testing.T) {
	err := Request{ID: "x", AdapterPath: "/loras/sql-v1"}.Validate()
	if err == nil || !strings.Contains(err.Error(), "no adapter name") {
		t.Fatalf("a path without a name returned %v, want a refusal that says so", err)
	}
}
