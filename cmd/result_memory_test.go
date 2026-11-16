package cmd

// Behavioral contracts of the Result's memory breakdown (reports.kernel.<role>.memory):
//
//	BC-M1 the parts of a GPU's memory (each term of the kernel's fixed occupancy, the LoRA
//	      reservation and the KV pool) add up to the budget, which is the device total times
//	      gpu_memory_utilization rounded up; the CUDA-graph term counts only when the engine
//	      took it from the budget.
//	BC-M2 a replica's figures are its GPUs' figures summed (absent under pipeline parallelism).
//	BC-M3 the KV pool's bytes divide into num_gpu_blocks blocks, of which the instances can use
//	      all but the null block (kv_blocks); for a model with one KV cache group
//	      num_gpu_blocks * block_size_tokens is kv_capacity_tokens_per_replica.
//	BC-M4 each opened pool of a disaggregated run has its own breakdown.

import (
	"encoding/json"
	"fmt"
	"strconv"
	"testing"
)

// memoryOf reads the breakdown of each pool a Result reports.
func memoryOf(t *testing.T, stdout string) map[string]poolMemory {
	t.Helper()
	out := map[string]poolMemory{}
	for role, section := range parseResult(t, stdout).Reports.Kernel {
		raw, err := json.Marshal(section)
		if err != nil {
			t.Fatal(err)
		}
		var s struct {
			Memory *poolMemory `json:"memory"`
		}
		if err := json.Unmarshal(raw, &s); err != nil {
			t.Fatalf("reports.kernel.%s: %v", role, err)
		}
		if s.Memory != nil {
			out[role] = *s.Memory
		}
	}
	return out
}

// partsSum is the sum of the parts of b that the budget is spent on; graphs says whether the
// CUDA-graph estimate is one of them.
func (b memoryBytes) partsSum(graphs bool) int64 {
	sum := b.Weights + b.ActivationPeak + b.EPLBRedundant + b.CommBuffers + b.NonTorchInBudget +
		b.EngineBuffers + b.LoRAReserved + b.KVPool
	if graphs {
		sum += b.CUDAGraph
	}
	return sum
}

func TestResult_MemoryPartsSumToTheBudget(t *testing.T) {
	for name, c := range map[string]struct {
		args  []string
		roles []string
	}{
		// four-way tensor parallel on one replica, MoE
		"tensor parallel": {resultRunArgs(), []string{"colocated"}},
		// two pools, each its own model and parallelism
		"prefill and decode pools": {append([]string{"run"}, pdBurst...), []string{"prefill", "decode"}},
	} {
		args := c.args
		t.Run(name, func(t *testing.T) {
			// GIVEN a run on the kernel backend
			stdout, stderr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
			}
			// WHEN the Result is read
			mem := memoryOf(t, stdout)
			for _, role := range c.roles {
				if _, ok := mem[role]; !ok {
					t.Fatalf("the Result carries no memory breakdown for the %s pool: %v", role, mem)
				}
			}
			// THEN every pool's parts add up to its budget, per GPU and per replica
			for role, pm := range mem {
				if pm.PerGPU.KVPool <= 0 || pm.PerGPU.Weights <= 0 {
					t.Errorf("%s: weights %d, KV pool %d; both must be positive", role, pm.PerGPU.Weights, pm.PerGPU.KVPool)
				}
				if got := pm.PerGPU.partsSum(pm.CUDAGraphInBudget); got != pm.PerGPU.Budget {
					t.Errorf("%s per GPU: the parts add up to %d, the budget is %d", role, got, pm.PerGPU.Budget)
				}
				if pm.PerGPU.Budget > pm.PerGPU.DeviceTotal {
					t.Errorf("%s: budget %d exceeds the device's %d", role, pm.PerGPU.Budget, pm.PerGPU.DeviceTotal)
				}
				if pm.PerReplica == nil {
					continue
				}
				if want := pm.PerGPU.scaled(int64(pm.GPUsPerReplica)); *pm.PerReplica != want {
					t.Errorf("%s per replica: %+v, want the per-GPU figures times %d GPUs, %+v", role, *pm.PerReplica, pm.GPUsPerReplica, want)
				}
				if got := pm.PerReplica.partsSum(pm.CUDAGraphInBudget); got != pm.PerReplica.Budget {
					t.Errorf("%s per replica: the parts add up to %d, the budget is %d", role, got, pm.PerReplica.Budget)
				}
			}
		})
	}
}

// The tensor-parallel fixture keeps one GPU's shard of the weights in each of its four GPUs,
// and the replica holds four of them.
func TestResult_MemoryIsPerGPUAndPerReplicaUnderTensorParallelism(t *testing.T) {
	stdout, stderr, err := runKernelCLI(t, resultRunArgs()...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
	}
	pm, ok := memoryOf(t, stdout)["colocated"]
	if !ok {
		t.Fatal("no breakdown for the colocated pool")
	}
	if pm.GPUsPerReplica != 4 || pm.PerReplica == nil {
		t.Fatalf("gpus_per_replica %d, per_replica %v; want 4 GPUs and a replica total", pm.GPUsPerReplica, pm.PerReplica)
	}
	if pm.PerReplica.Weights != 4*pm.PerGPU.Weights {
		t.Errorf("replica weights %d, want 4 x %d", pm.PerReplica.Weights, pm.PerGPU.Weights)
	}
}

// denseScenario is a model with one KV cache group (Llama 3.1 70B, full attention only), whose
// reported KV capacity is its blocks times the block size. kernelTestScenario (gpt-oss-120b,
// sliding window and full attention) has two groups, and its reported capacity is the tokens one
// max_model_len request can hold across both (inference-sim#1969), not blocks times block size.
const denseScenario = "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml"

func TestResult_MemoryKVPoolHoldsTheInstancesBlocks(t *testing.T) {
	for name, c := range map[string]struct {
		scenario string
		// oneGroup: num_gpu_blocks x block_size_tokens is kv_capacity_tokens_per_replica.
		oneGroup bool
	}{
		"one KV cache group":  {denseScenario, true},
		"two KV cache groups": {kernelTestScenario, false},
	} {
		t.Run(name, func(t *testing.T) {
			// GIVEN a run of the model
			stdout, stderr, err := runKernelCLI(t, "run", "--scenario", c.scenario, "--seed", "7",
				"--num-requests", "10", "--rate", "10", "--horizon", "600000000")
			if err != nil {
				t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
			}
			// WHEN its breakdown and its reported capacity are read
			r := parseResult(t, stdout)
			pm, ok := memoryOf(t, stdout)["colocated"]
			if !ok {
				t.Fatal("no breakdown for the colocated pool")
			}
			if pm.NumGPUBlocks <= 0 || pm.KVBlocks <= 0 || pm.BlockSizeTokens <= 0 || pm.KVBytesPerBlock <= 0 {
				t.Fatalf("num_gpu_blocks %d, kv_blocks %d, block_size_tokens %d, kv_bytes_per_block %d; all must be positive",
					pm.NumGPUBlocks, pm.KVBlocks, pm.BlockSizeTokens, pm.KVBytesPerBlock)
			}
			cluster, _ := r.Reports.Engine["cluster"].(map[string]any)
			capacity, err := strconv.ParseInt(fmt.Sprint(cluster["kv_capacity_tokens_per_replica"]), 10, 64)
			if err != nil || capacity <= 0 {
				t.Fatalf("reports.engine.cluster.kv_capacity_tokens_per_replica = %v", cluster["kv_capacity_tokens_per_replica"])
			}
			// THEN a one-group pool's blocks are its reported capacity, and a two-group pool's are not
			tokens := pm.NumGPUBlocks * pm.BlockSizeTokens
			if c.oneGroup && tokens != capacity {
				t.Errorf("num_gpu_blocks %d x block_size_tokens %d = %d tokens, kv_capacity_tokens_per_replica is %d",
					pm.NumGPUBlocks, pm.BlockSizeTokens, tokens, capacity)
			}
			if !c.oneGroup && tokens == capacity {
				t.Errorf("num_gpu_blocks x block_size_tokens = %d equals the reported capacity; a two-group model's should differ", tokens)
			}
			// AND the pool's bytes divide into num_gpu_blocks blocks, all but the null block usable
			if blocks := pm.PerGPU.KVPool / pm.KVBytesPerBlock; blocks != pm.NumGPUBlocks {
				t.Errorf("the pool's %d bytes hold %d blocks of %d bytes, num_gpu_blocks is %d", pm.PerGPU.KVPool, blocks, pm.KVBytesPerBlock, pm.NumGPUBlocks)
			}
			if pm.KVBlocks != pm.NumGPUBlocks-1 {
				t.Errorf("kv_blocks %d, want num_gpu_blocks %d less the null block", pm.KVBlocks, pm.NumGPUBlocks)
			}
		})
	}
}
