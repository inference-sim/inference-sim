// Moved from blis-latency-kernel/examples/example_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

// The Go examples on the documentation site are sections of this file, included between the
// --8<-- markers. Go runs each Example and compares what it prints with its // Output:
// comment, so a page cannot show a call that no longer compiles or a number this kernel no
// longer computes. They read the fetched catalog and registry (scripts/fetch-testdata.sh).

import (
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	vllmrules "github.com/inference-sim/blis-vllm/rules"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
)

// --8<-- [start:repos]
// The roots a scenario's names resolve against (its model, chip, fabric and coefficient
// sets), and the engine rules its engine_version selects: vLLM's, from the engine model.
var repos = latencykernel.Repos{
	Scenarios:      "../testdata",
	Catalog:        "../testdata/catalog",
	Registry:       "../testdata/registry",
	Rules:          latencykernel.RulesFrom(vllmrules.Lookup),
	RulesVersions:  vllmrules.Versions,
	EngineSettings: vllmsettings.OfPool,
}

func ms(d time.Duration) float64 { return d.Seconds() * 1e3 }

// Pricing a decode step: open a scenario, describe the batch, read the estimate.
func ExampleOpen() {
	// --8<-- [start:open]
	// DeepSeek-V3, served in fp8 on eight H200s at tensor-parallel width 8.
	const scenario = "aisimulate/deepseek-v3-h200-fp8-sglang-tp8.yaml"
	k, err := latencykernel.Open(scenario, repos)
	if err != nil {
		panic(err)
	}

	// 32 requests, each decoding one token against a context of 4,096 tokens.
	batch := kernel.Batch{DecodeThreshold: 8}
	for range 32 {
		batch.Reqs = append(batch.Reqs, kernel.ReqShape{
			Scheduled: 1, Computed: 4095, PromptLen: 4096,
		})
	}

	e := k.StepTime(batch)
	fmt.Printf("between %.1f and %.1f ms, bottleneck %s\n",
		ms(e.Overlap), ms(e.NoOverlap), e.Bottleneck)
	for _, r := range []kernel.Resource{kernel.ResourceSM, kernel.ResourceHBM,
		kernel.ResourceNVLink, kernel.ResourceHost} {
		fmt.Printf("%-7s %5.1f ms\n", r, ms(e.PerResource[r]))
	}
	// --8<-- [end:open]

	// Output:
	// between 24.1 and 25.7 ms, bottleneck hbm
	// sm        4.5 ms
	// hbm      16.5 ms
	// nvlink    1.2 ms
	// host      3.5 ms
}

// Memory: what one rank holds before any request arrives, and what each request adds.
func ExampleKernel_FixedBytes() {
	k, err := latencykernel.Open("aisimulate/deepseek-v3-h200-fp8-sglang-tp8.yaml", repos)
	if err != nil {
		panic(err)
	}
	// --8<-- [start:memory]
	const gib = 1 << 30
	m := k.FixedBytes()
	fmt.Printf("weights      %5.1f GiB\n", float64(m.Weights)/gib)
	fmt.Printf("activations  %5.1f GiB\n", float64(m.ActivationPeak)/gib)
	fmt.Printf("CUDA graphs  %5.1f GiB\n", float64(m.CUDAGraph)/gib)
	fmt.Printf("comm buffers %5.1f GiB\n", float64(m.CommBuffers)/gib)
	fmt.Printf("EPLB         %5.1f GiB\n", float64(m.EPLBRedundant)/gib)
	fmt.Printf("total        %5.1f GiB per rank\n", float64(m.Total())/gib)
	// The communicator and engine workspace, held before the budget is measured.
	fmt.Printf("before the budget %4.1f GiB\n", float64(k.PreSnapshotBytes())/gib)

	// KV cache for one request, rounded up to whole pages.
	fmt.Printf("a 4,096-token request: %.0f MiB of KV cache per rank\n",
		float64(k.SequenceVariableBytes(4096))/(1<<20))
	// --8<-- [end:memory]

	// Output:
	// weights       78.2 GiB
	// activations    1.1 GiB
	// CUDA graphs    0.8 GiB
	// comm buffers   0.0 GiB
	// EPLB           0.0 GiB
	// total         80.1 GiB per rank
	// before the budget  3.9 GiB
	// a 4,096-token request: 137 MiB of KV cache per rank
}

// Evidence: how much of what a kernel used was measured, and what it assumed.
func ExampleKernel_Evidence() {
	k, err := latencykernel.Open("aisimulate/deepseek-v3-h200-fp8-sglang-tp8.yaml", repos)
	if err != nil {
		panic(err)
	}
	// --8<-- [start:evidence]
	measured, total, assumed := k.Evidence()
	fmt.Printf("%d of %d values rest on evidence; %d were assumed\n",
		measured, total, len(assumed))

	// The same entries, counted by the method that produced each.
	byMethod := map[string]int{}
	for _, o := range k.Provenance() {
		byMethod[o.Method]++
	}
	for _, m := range slices.Sorted(maps.Keys(byMethod)) {
		fmt.Printf("  %-12s %3d\n", m, byMethod[m])
	}

	// The kernel's own assumptions, as distinct from registry values whose method is
	// "assumed", carry the set name KernelAssumptionSet.
	for _, o := range k.Provenance() {
		if o.Set == latencykernel.KernelAssumptionSet {
			fmt.Println("the kernel assumed", o.Name)
		}
	}
	// --8<-- [end:evidence]

	// Output:
	// 113 of 155 values rest on evidence; 41 were assumed
	//   assumed       41
	//   measured      99
	//   not_charged    1
	//   vendor_spec   14
	// the kernel assumed max_cudagraph_capture_size
	// the kernel assumed engine_version
}

// A deployment variant that was never written to disk: load the documents once, change one
// setting, and build a kernel from the result.
func ExampleNew() {
	// --8<-- [start:new]
	in, err := latencykernel.OpenInputs("aisimulate/deepseek-v3-h200-fp8-sglang-tp8.yaml",
		repos, 0)
	if err != nil {
		panic(err)
	}
	base, err := latencykernel.New(in)
	if err != nil {
		panic(err)
	}

	// The same deployment with a 16-bit KV cache instead of fp8. The engine settings are the
	// caller's decoding of the pool's engine block (Inputs.Engine), and New copied what it
	// needed, so changing them now leaves base as it was.
	in.Engine.CacheDType = "bfloat16"
	wide, err := latencykernel.New(in)
	if err != nil {
		panic(err)
	}
	fmt.Printf("fp8:  %d bytes of KV per token per rank\n", base.SequenceVariableBytes(64)/64)
	fmt.Printf("bf16: %d bytes of KV per token per rank\n", wide.SequenceVariableBytes(64)/64)
	// --8<-- [end:new]

	// Output:
	// fp8:  35136 bytes of KV per token per rank
	// bf16: 70272 bytes of KV per token per rank
}
