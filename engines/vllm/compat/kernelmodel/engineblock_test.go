// Original to blis-vllm.

package kernelmodel

import (
	"testing"

	latencykernel "github.com/inference-sim/blis-latency-kernel"

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
	pool := poolWith(t, engineconfig.Engine{BlockSize: 64, MaxNumSeqs: 8, CacheDType: "fp8"})
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

// DraftForTest writes the drafter into the priced pool's block and changes nothing else in it.
func TestDraftForTestWritesThePricedPoolsDrafter(t *testing.T) {
	in := latencykernel.Inputs{PoolIndex: 1, Deployment: &deployment.Deployment{Pools: []deployment.Pool{
		{Engine: blockOf(t, engineconfig.Engine{BlockSize: 16})}, {Engine: blockOf(t, engineconfig.Engine{BlockSize: 64, CacheDType: "fp8"})}}}}
	if err := DraftForTest(&in, "dspark", 5); err != nil {
		t.Fatal(err)
	}
	e := mustDecode(t, in.Deployment.Pools[1])
	if e.Speculative == nil || e.Speculative.Method != "dspark" || e.Speculative.NumSpecTokens != 5 ||
		e.BlockSize != 64 || e.CacheDType != "fp8" {
		t.Errorf("the priced pool decodes as %+v", e)
	}
	if other := mustDecode(t, in.Deployment.Pools[0]); other.Speculative != nil || other.BlockSize != 16 {
		t.Errorf("the other pool changed: %+v", other)
	}
}

// poolWith is a colocated pool whose engine block states e.
func poolWith(t testing.TB, e engineconfig.Engine) deployment.Pool {
	t.Helper()
	return deployment.Pool{Role: deployment.RoleColocated, Engine: blockOf(t, e)}
}

// blockOf is e encoded as an engine block.
func blockOf(t testing.TB, e engineconfig.Engine) deployment.EngineBlock {
	t.Helper()
	b, err := engineconfig.Encode(e)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
