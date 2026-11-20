// Original to blis-vllm.

package engineconfig

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"gopkg.in/yaml.v3"
)

// decodePools reads a Deployment document the way a scenario file is read and decodes each pool's
// engine block, so an anchor or merge key in the file reaches FromVLLM as the loader leaves it.
func decodePools(t *testing.T, doc string) []Engine {
	t.Helper()
	var d deployment.Deployment
	if err := yaml.Unmarshal([]byte(doc), &d); err != nil {
		t.Fatal(err)
	}
	var out []Engine
	for i, p := range d.Pools {
		e, err := Decode(p.Engine)
		if err != nil {
			t.Fatalf("pool %d: %v", i, err)
		}
		out = append(out, e)
	}
	return out
}

func TestAnAliasedEngineBlockStatesWhatItsAnchorStates(t *testing.T) {
	es := decodePools(t, `
kind: Deployment
name: x
pools:
  - role: prefill
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine: &eng
      max_model_len: 131072
      kv_cache_dtype: fp8
      compilation_config: {cudagraph_mode: FULL}
  - role: decode
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine: *eng
`)
	for i, e := range es {
		if e.MaxModelLen != 131072 || e.CacheDType != "fp8" || e.CUDAGraphMode != "FULL" {
			t.Errorf("pool %d: max_model_len %v, cache_dtype %q, cudagraph_mode %q; want the anchor's 131072, fp8, FULL",
				i, e.MaxModelLen, e.CacheDType, e.CUDAGraphMode)
		}
	}
}

func TestAMergeKeyInAnEngineBlockCarriesTheMergedKeysAndTheExplicitOneWins(t *testing.T) {
	es := decodePools(t, `
kind: Deployment
name: x
pools:
  - role: prefill
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine: &base
      max_model_len: 4096
      kv_cache_dtype: fp8
      compilation_config: {cudagraph_mode: FULL}
  - role: decode
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine:
      <<: *base
      max_model_len: 8192
  - role: decode
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine:
      <<: [{kv_cache_dtype: fp8, max_num_seqs: 4}, {max_num_seqs: 9, block_size: 32}]
`)
	if es[1].CacheDType != "fp8" || es[1].CUDAGraphMode != "FULL" {
		t.Errorf("merged keys lost: cache_dtype %q, cudagraph_mode %q", es[1].CacheDType, es[1].CUDAGraphMode)
	}
	if got := es[1].MaxModelLen; got != 8192 {
		t.Errorf("the explicit max_model_len must win over the merged 4096, got %v", got)
	}
	if es[2].MaxNumSeqs != 4 || es[2].BlockSize != 32 {
		t.Errorf("in a list of merged maps the earlier wins: max_num_seqs %d (want 4), block_size %d (want 32)", es[2].MaxNumSeqs, es[2].BlockSize)
	}
}

func TestAnAliasNestedInAGroupIsResolved(t *testing.T) {
	es := decodePools(t, `
kind: Deployment
name: x
pools:
  - role: prefill
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine:
      compilation_config: &cc {cudagraph_mode: PIECEWISE}
  - role: decode
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine:
      compilation_config: *cc
`)
	if es[1].CUDAGraphMode != "PIECEWISE" {
		t.Errorf("cudagraph_mode %q, want PIECEWISE through the alias", es[1].CUDAGraphMode)
	}
}

func TestAPlainEngineBlockDecodesAsBefore(t *testing.T) {
	es := decodePools(t, `
kind: Deployment
name: x
pools:
  - role: prefill
    nodes: 1
    parallel: {tp: 8, pp: 1, dp: 1}
    engine: {max_num_seqs: 7, kv_cache_dtype: fp8, compilation_config: {cudagraph_mode: NONE}}
`)
	if es[0].MaxNumSeqs != 7 || es[0].CacheDType != "fp8" || es[0].CUDAGraphMode != "NONE" {
		t.Errorf("plain block changed: %+v", es[0])
	}
}

// A block that reaches FromVLLM as the alias node the parser left (a caller that walks the document
// tree rather than decoding it) is translated as the mapping it names, not returned as it was.
func TestFromVLLMTranslatesAnAliasNodeAsTheMappingItNames(t *testing.T) {
	var doc yaml.Node
	if err := yaml.Unmarshal([]byte("a: &eng {kv_cache_dtype: fp8, compilation_config: {cudagraph_mode: FULL}}\nb: *eng\n"), &doc); err != nil {
		t.Fatal(err)
	}
	alias := doc.Content[0].Content[3]
	if alias.Kind != yaml.AliasNode {
		t.Fatalf("setup: node kind %v, want an alias", alias.Kind)
	}
	out, err := FromVLLM(alias)
	if err != nil {
		t.Fatal(err)
	}
	var e Engine
	if err := out.Decode(&e); err != nil {
		t.Fatal(err)
	}
	if e.CacheDType != "fp8" || e.CUDAGraphMode != "FULL" {
		t.Errorf("alias translated as %+v, want the anchor's cache_dtype fp8 and cudagraph_mode FULL", e)
	}
}
