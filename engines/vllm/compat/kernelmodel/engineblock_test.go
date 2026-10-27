// Original to blis-vllm.

package kernelmodel

import (
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// editPoolEngine changes a pool's engine block the way a caller changes a scenario: decoded as
// vLLM's engine configuration, edited, and written back (DecodeEngine, SetEngine), so the test
// edits the block blis-vllm reads rather than a copy of it.
func editPoolEngine(t testing.TB, pool *deployment.Pool, edit func(*engineconfig.Engine)) {
	t.Helper()
	e, err := decodeEngine(*pool)
	if err != nil {
		t.Fatal(err)
	}
	edit(&e)
	if err := SetEngine(pool, e); err != nil {
		t.Fatal(err)
	}
}

// An edit reaches the block blis-vllm decodes, and nothing else in it changes.
func TestEditPoolEngineReachesTheDecodedBlock(t *testing.T) {
	pool := deployment.Pool{Engine: deployment.Engine{BlockSize: 64, MaxNumSeqs: 8, CacheDType: "fp8"}}
	editPoolEngine(t, &pool, func(e *engineconfig.Engine) {
		e.Speculative = &engineconfig.Speculative{Method: "mtp", NumSpecTokens: 2}
	})
	e, err := decodeEngine(pool)
	if err != nil {
		t.Fatal(err)
	}
	if e.Speculative == nil || e.Speculative.Method != "mtp" || e.Speculative.NumSpecTokens != 2 ||
		e.BlockSize != 64 || e.MaxNumSeqs != 8 || e.CacheDType != "fp8" {
		t.Errorf("after the edit the block decodes as %+v", e)
	}
}
