package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
)

// resultMemoryOf runs a scenario once and returns each pool's memory block, keyed by the role the
// Result states it under. The block is fixed when the engine is constructed, so one request is
// enough to make the run emit it; the run is the only surface that reports it.
func resultMemoryOf(t *testing.T, args ...string) map[string]poolMemory {
	t.Helper()
	base := []string{"run", "--rate", "1", "--num-requests", "1", "--seed", "1"}
	stdout, stderr, err := runKernelCLI(t, append(base, args...)...)
	if err != nil {
		t.Fatalf("blis run: %v\n%s", err, lastLines(stderr, 5))
	}
	reports, ok := resultDoc(t, stdout)["reports"].(map[string]any)
	if !ok {
		t.Fatal("the Result states no reports object")
	}
	engine, ok := reports["engine"].(map[string]any)
	if !ok {
		t.Fatal("the Result states no reports.engine")
	}
	out := map[string]poolMemory{}
	for role, section := range engine {
		block, ok := section.(map[string]any)
		if !ok {
			continue
		}
		memory, ok := block["memory"]
		if !ok {
			continue // the cluster section holds no memory block
		}
		raw, err := json.Marshal(memory)
		if err != nil {
			t.Fatal(err)
		}
		var p poolMemory
		if err := json.Unmarshal(raw, &p); err != nil {
			t.Fatalf("reports.engine.%s.memory is not a pool memory block: %v", role, err)
		}
		if string(p.Role) != role {
			t.Errorf("reports.engine.%s.memory carries role %q", role, p.Role)
		}
		out[role] = p
	}
	if len(out) == 0 {
		t.Fatalf("the Result states a memory block for no pool; stderr:\n%s", lastLines(stderr, 10))
	}
	return out
}

// checkPoolMemoryConsistent requires a pool's figures to describe one pool: the KV triple derives
// from its block, the pool is not larger than the memory the engine left for KV, and the fixed
// breakdown sums to its total.
func checkPoolMemoryConsistent(t *testing.T, p poolMemory) {
	t.Helper()
	if p.KV.BlockTokens <= 0 || p.KV.BlockBytes <= 0 {
		t.Fatalf("%s: block is %d tokens of %d bytes", p.Role, p.KV.BlockTokens, p.KV.BlockBytes)
	}
	wantTokens := p.KV.BlocksPerRank * int64(p.KV.BlockTokens)
	if p.KV.TokensPerRank != wantTokens {
		t.Errorf("%s: tokens_per_rank %d != blocks_per_rank %d x block_tokens %d",
			p.Role, p.KV.TokensPerRank, p.KV.BlocksPerRank, p.KV.BlockTokens)
	}
	wantBytes := p.KV.BlocksPerRank * p.KV.BlockBytes
	if p.KV.BytesPerRank != wantBytes {
		t.Errorf("%s: bytes_per_rank %d != blocks_per_rank %d x block_bytes %d",
			p.Role, p.KV.BytesPerRank, p.KV.BlocksPerRank, p.KV.BlockBytes)
	}
	wantPerToken, err := perTokenBytes(p.KV.BlockBytes, p.KV.BlockTokens)
	if err != nil {
		t.Fatal(err)
	}
	if p.KV.PerTokenBytes != wantPerToken {
		t.Errorf("%s: per_token_bytes %d != ceil(%d / %d) = %d",
			p.Role, p.KV.PerTokenBytes, p.KV.BlockBytes, p.KV.BlockTokens, wantPerToken)
	}
	// The blocks are cut from the allocatable bytes, so the pool takes at most one block less than
	// all of it.
	if p.KV.BytesPerRank > p.AllocatableBytes {
		t.Errorf("%s: the KV pool (%d B) is larger than the memory left for it (%d B)",
			p.Role, p.KV.BytesPerRank, p.AllocatableBytes)
	}
	if p.AllocatableBytes-p.KV.BytesPerRank >= p.KV.BlockBytes {
		t.Errorf("%s: the KV pool leaves %d B unused, a whole block (%d B) or more",
			p.Role, p.AllocatableBytes-p.KV.BytesPerRank, p.KV.BlockBytes)
	}
	f := p.Fixed
	if sum := f.Weights + f.ActivationPeak + f.CUDAGraph + f.EPLBRedundant + f.CommBuffers + f.NonTorch + f.EngineBuffers; f.Total != sum {
		t.Errorf("%s: fixed.total %d != the sum of its terms %d", p.Role, f.Total, sum)
	}
	if p.DeviceBytes <= 0 || p.BudgetBytes <= 0 || p.BudgetBytes > p.DeviceBytes {
		t.Errorf("%s: device %d B, budget %d B", p.Role, p.DeviceBytes, p.BudgetBytes)
	}
}

// A colocated scenario reports its one pool: the engine that sizes it, and a KV pool whose block,
// tokens and bytes agree.
func TestResultMemoryReportsTheColocatedPool(t *testing.T) {
	pools := resultMemoryOf(t, "--scenario", "gpt-oss-120b-h100-fp4-vllm-tp2.yaml")
	if len(pools) != 1 {
		t.Fatalf("%d pools, want the colocated scenario's one: %v", len(pools), pools)
	}
	p := pools[string(deployment.RoleColocated)]
	if p.Engine != engineKindVLLM {
		t.Errorf("pool engine %q, want vllm", p.Engine)
	}
	checkPoolMemoryConsistent(t, p)

	// The fixed occupancy and the KV pool partition the budget: vLLM sizes the pool from exactly
	// the budget less its profiler's occupancy and any static adapter reservation (none here).
	if got, want := p.AllocatableBytes, p.BudgetBytes-p.Fixed.Total-p.ReservedBytes; got != want {
		t.Errorf("allocatable %d != budget %d - fixed %d - reserved %d = %d",
			got, p.BudgetBytes, p.Fixed.Total, p.ReservedBytes, want)
	}
}

// A run with a static LoRA adapter reservation reports it: the pool is sized with the reservation
// set aside, so the record states the reservation rather than leaving a gap between the budget, the
// fixed occupancy and the pool.
func TestResultMemoryStatesTheStaticLoRAReservation(t *testing.T) {
	catalogDir := setupKernelTestFixtures(t)
	orig := captureCmdLevelVars()
	origReserved := loraReservedBytesForKV
	t.Cleanup(func() { orig.restore(); loraReservedBytesForKV = origReserved })
	catalogPath = catalogDir

	adoptKernelDeployment(&cobra.Command{})
	loraReservedBytesForKV = 0
	plain, err := vllmPoolMemory(kernelOpened, deployment.RoleColocated)
	if err != nil {
		t.Fatalf("the adopted pool: %v", err)
	}
	if plain.ReservedBytes != 0 {
		t.Errorf("no reservation reports %d bytes", plain.ReservedBytes)
	}

	const reserved = 8 << 30
	loraReservedBytesForKV = reserved
	with, err := vllmPoolMemory(kernelOpened, deployment.RoleColocated)
	if err != nil {
		t.Fatalf("the pool with a reservation: %v", err)
	}
	tp := int64(kernelOpened.Deployment().TP)
	if want := reserved / tp; with.ReservedBytes != want {
		t.Errorf("reserved_bytes %d, want the %d-byte reservation sharded over tp %d = %d",
			with.ReservedBytes, reserved, tp, want)
	}
	if got, want := plain.AllocatableBytes-with.AllocatableBytes, with.ReservedBytes; got != want {
		t.Errorf("the reservation moved allocatable_bytes by %d, want its %d", got, want)
	}
	if with.KV.BlocksPerRank > plain.KV.BlocksPerRank {
		t.Errorf("the reservation grew the pool from %d to %d blocks", plain.KV.BlocksPerRank, with.KV.BlocksPerRank)
	}
	checkPoolMemoryConsistent(t, with)
	if got, want := with.AllocatableBytes, with.BudgetBytes-with.Fixed.Total-with.ReservedBytes; got != want {
		t.Errorf("allocatable %d != budget %d - fixed %d - reserved %d = %d",
			got, with.BudgetBytes, with.Fixed.Total, with.ReservedBytes, want)
	}
}

// A run resolves the LoRA flags onto its pool: a reservation stated on the command line is set
// aside before the pool is sized, so the Result states the pool that run admits against rather
// than one it could not use.
func TestResultMemoryAppliesTheLoRAReservation(t *testing.T) {
	const scenario = "gpt-oss-120b-h100-fp4-vllm-tp2.yaml"
	dir := t.TempDir()
	loraPath := filepath.Join(dir, "lora.yaml")
	if err := os.WriteFile(loraPath, []byte(
		"lora:\n  adapter_capacity: 4\n  adapters:\n    - id: adapter_0\n      rank: 16\n"), 0o644); err != nil {
		t.Fatalf("write the LoRA config: %v", err)
	}

	plain := resultMemoryOf(t, "--scenario", scenario)[string(deployment.RoleColocated)]
	with := resultMemoryOf(t, "--scenario", scenario, "--lora-config", loraPath)[string(deployment.RoleColocated)]
	if plain.ReservedBytes != 0 {
		t.Errorf("no LoRA flags report a %d-byte reservation", plain.ReservedBytes)
	}
	// defaults.yaml's footprint (2.0e6 B a rank) over the file's capacity and widest adapter rank
	// (4 x 16), sharded as the weights are over the scenario's tp.
	const want = 4 * 16 * 2_000_000 / 2
	if with.ReservedBytes != want {
		t.Errorf("reserved_bytes %d, want the 4 x 16-adapter reservation sharded over tp 2 = %d",
			with.ReservedBytes, want)
	}
	if got := plain.AllocatableBytes - with.AllocatableBytes; got != with.ReservedBytes {
		t.Errorf("the reservation moved allocatable_bytes by %d, want its %d", got, with.ReservedBytes)
	}
	if with.KV.BlocksPerRank >= plain.KV.BlocksPerRank {
		t.Errorf("the reservation left the pool at %d blocks, not below the plain %d",
			with.KV.BlocksPerRank, plain.KV.BlocksPerRank)
	}
	checkPoolMemoryConsistent(t, with)
	if got, want := with.AllocatableBytes, with.BudgetBytes-with.Fixed.Total-with.ReservedBytes; got != want {
		t.Errorf("allocatable %d != budget %d - fixed %d - reserved %d = %d",
			got, with.BudgetBytes, with.Fixed.Total, with.ReservedBytes, want)
	}
}

// A disaggregated deployment reports each pool it prices, with the role the Result states it
// under; each pool's figures are its own engine block's.
func TestResultMemoryReportsEveryPoolOfADisaggregatedDeployment(t *testing.T) {
	pools := resultMemoryOf(t, "--scenario", "glm-5-h200-3p1d-ib.yaml", "--scenarios", "../examplescenarios")
	if len(pools) != 2 {
		t.Fatalf("%d pools, want the deployment's prefill and decode pools: %v", len(pools), pools)
	}
	for _, role := range []deployment.Role{deployment.RolePrefill, deployment.RoleDecode} {
		p, ok := pools[string(role)]
		if !ok {
			t.Errorf("no %s pool: %v", role, pools)
			continue
		}
		if p.Engine != engineKindVLLM {
			t.Errorf("%s: engine %q, want vllm", role, p.Engine)
		}
		checkPoolMemoryConsistent(t, p)
	}
}

// An SGLang scenario reports its colocated pool from SGLang's own sizing.
func TestResultMemoryReportsAnSGLangPool(t *testing.T) {
	pools := resultMemoryOf(t, "--scenarios", "../testdata/kernel/sglang", "--scenario",
		"glm-5-h200-fp8-sglang-tp8.yaml")
	if len(pools) != 1 {
		t.Fatalf("%d pools, want the colocated scenario's one: %v", len(pools), pools)
	}
	p := pools[string(deployment.RoleColocated)]
	if p.Engine != engineKindSGLang {
		t.Errorf("pool engine %q, want sglang", p.Engine)
	}
	checkPoolMemoryConsistent(t, p)
	if p.BudgetBytes >= p.DeviceBytes {
		// SGLang sizes within the device's memory before the weights load, which is below the
		// device's total: CUDA holds the difference.
		t.Errorf("budget %d B is not below the device's %d B", p.BudgetBytes, p.DeviceBytes)
	}
}

// A block of zero tokens is refused rather than divided by: upstream sizing requires a positive
// block, and a per-token figure derived from zero would be a crash or a lie.
func TestPerTokenBytesRefusesAZeroTokenBlock(t *testing.T) {
	if _, err := perTokenBytes(4096, 0); err == nil {
		t.Fatal("perTokenBytes divided by a zero-token block")
	}
	if got, err := perTokenBytes(4096, 16); err != nil || got != 256 {
		t.Errorf("perTokenBytes(4096, 16) = %d, %v; want 256", got, err)
	}
}

// A DP>1 MoE deployment sizes its DP cores into one pool, so the instance's aggregate is
// data_parallel x the per-rank figure. A dense model's data parallelism is replicas, each an
// instance of its own, so its aggregate is the per-rank figure.
func TestResultMemoryAggregateFollowsDataParallelism(t *testing.T) {
	t.Run("moe", func(t *testing.T) {
		pools := resultMemoryOf(t, "--scenario", "minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml")
		p := pools[string(deployment.RoleColocated)]
		checkPoolMemoryConsistent(t, p)
		if p.KV.DataParallel != 2 {
			t.Fatalf("data_parallel = %d, want the scenario's 2", p.KV.DataParallel)
		}
		if p.KV.AggregateBytes == p.KV.BytesPerRank {
			t.Fatalf("aggregate_bytes %d equals bytes_per_rank: the DP>1 pool is not scaled",
				p.KV.AggregateBytes)
		}
		if want := int64(p.KV.DataParallel) * p.KV.BytesPerRank; p.KV.AggregateBytes != want {
			t.Errorf("aggregate_bytes %d != data_parallel %d x bytes_per_rank %d",
				p.KV.AggregateBytes, p.KV.DataParallel, p.KV.BytesPerRank)
		}
		if want := int64(p.KV.DataParallel) * p.KV.TokensPerRank; p.KV.AggregateTokens != want {
			t.Errorf("aggregate_tokens %d != data_parallel %d x tokens_per_rank %d",
				p.KV.AggregateTokens, p.KV.DataParallel, p.KV.TokensPerRank)
		}
		if want := int64(p.KV.DataParallel) * p.KV.BlocksPerRank; p.KV.AggregateBlocks != want {
			t.Errorf("aggregate_blocks %d != data_parallel %d x blocks_per_rank %d",
				p.KV.AggregateBlocks, p.KV.DataParallel, p.KV.BlocksPerRank)
		}
	})

	t.Run("dense", func(t *testing.T) {
		// A dense model refuses dp > 1 (independent dense engines are --num-instances), so its
		// data parallelism is always replicas of one rank's pool: the aggregate is that same
		// per-rank figure, never a scaled one.
		pools := resultMemoryOf(t, "--scenario", "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml")
		p := pools[string(deployment.RoleColocated)]
		checkPoolMemoryConsistent(t, p)
		if p.KV.DataParallel != 1 {
			t.Fatalf("data_parallel = %d, want 1 for a dense deployment", p.KV.DataParallel)
		}
		if p.KV.AggregateBytes != p.KV.BytesPerRank {
			t.Errorf("aggregate_bytes %d != bytes_per_rank %d for a dense model",
				p.KV.AggregateBytes, p.KV.BytesPerRank)
		}
		if p.KV.AggregateTokens != p.KV.TokensPerRank {
			t.Errorf("aggregate_tokens %d != tokens_per_rank %d for a dense model",
				p.KV.AggregateTokens, p.KV.TokensPerRank)
		}
	})
}
