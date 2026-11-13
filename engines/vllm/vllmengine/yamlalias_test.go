// Original to blis-vllm.

package vllmengine

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

const aliasDeployment = `kind: Deployment
name: x
pools:
  - role: prefill
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine: &eng
      max_model_len: 131072
      kv_cache_dtype: fp8
  - role: decode
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine: *eng
  - role: decode
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine:
      <<: *eng
      max_num_seqs: 128
`

// A pool written `engine: *eng` states what the anchor states: the block poolBlock hands the engine
// is the anchor's mapping, not the alias node a yaml.Node field keeps. Before, the alias was taken
// for "no block", so the run failed with "max_model_len must be stated" for a file that states it.
func TestPoolBlockOfAnAliasedOrMergedEngineBlockStatesTheAnchorsKeys(t *testing.T) {
	path := filepath.Join(t.TempDir(), "d.yaml")
	if err := os.WriteFile(path, []byte(aliasDeployment), 0o644); err != nil {
		t.Fatal(err)
	}
	for i, wantSeqs := range []string{"", "", "128"} {
		node, _, err := poolBlock(path, i)
		if err != nil {
			t.Fatal(err)
		}
		if node.Kind != yaml.MappingNode {
			t.Fatalf("pool %d: block kind %v, want a mapping", i, node.Kind)
		}
		if got := stated(node, "max_model_len"); got != "131072" {
			t.Errorf("pool %d: max_model_len %q, want the anchor's 131072", i, got)
		}
		if got := stated(node, "kv_cache_dtype"); got != "fp8" {
			t.Errorf("pool %d: kv_cache_dtype %q, want the anchor's fp8", i, got)
		}
		if got := stated(node, "max_num_seqs"); got != wantSeqs {
			t.Errorf("pool %d: max_num_seqs %q, want %q", i, got, wantSeqs)
		}
	}
}

// A block a caller hands Prepare as an alias node is completed as the mapping it names.
func TestCompleteReadsAGivenAliasNodeAsTheMappingItNames(t *testing.T) {
	var doc yaml.Node
	if err := yaml.Unmarshal([]byte("a: &e {max_model_len: 4096}\nb: *e\n"), &doc); err != nil {
		t.Fatal(err)
	}
	alias := doc.Content[0].Content[3]
	if alias.Kind != yaml.AliasNode {
		t.Fatalf("setup: kind %v, want an alias", alias.Kind)
	}
	d001 := openPool(t, "d001-032.yaml")
	if got := stated(d001.complete(*alias), "max_model_len"); got != "4096" {
		t.Errorf("max_model_len %q, want the alias's 4096", got)
	}
}
