# Price a batch from Go

<p class="lede">Point the kernel at its inputs, open a kernel for a deployment, and price a
batch. The later sections build variants of a deployment in memory, read its memory and its
evidence, and say what to do when a deployment is refused or a step must be priced often.</p>

```sh
go get github.com/inference-sim/blis-latency-kernel@latest
```

## Point it at its inputs

A scenario names its model, its chip, its fabric and its coefficient sets, but not where they
live. `Repos` says where: a directory of scenario files, a blis-catalog checkout, and a
blis-registry checkout.

```go
// The repos example lives in blis-vllm's kerneltests/example_test.go
// (https://github.com/inference-sim/blis-vllm/blob/main/kerneltests/example_test.go),
// which decodes the vLLM engine block it opens the kernel from.
```

These are the paths in this repository after `scripts/fetch-testdata.sh` has fetched the
catalog and registry at the commits the kernel is tested against. Elsewhere, point `Catalog`
and `Registry` at checkouts of the two repositories, at the versions on
[Sources of truth](../reference/sources.md). [Scenarios and their inputs](inputs.md) describes
the scenario files.

## Open a kernel and price a step

`Open` reads a scenario file and everything it names, and builds a kernel for the
deployment's first pool. `OpenPool` does the same for any pool of a disaggregated deployment,
which states a prefill pool and a decode pool, each with its own kernel.

A batch lists, for each request, the tokens it schedules this step (`Scheduled`), the tokens
it has already computed (`Computed`) and its prompt length (`PromptLen`). A steady-state
decode schedules one token against its whole context:

<!-- example_test.go:open: blis-vllm kerneltests/example_test.go -->
```go
// The open example lives in blis-vllm's kerneltests/example_test.go
// (https://github.com/inference-sim/blis-vllm/blob/main/kerneltests/example_test.go),
// which decodes the vLLM engine block it opens the kernel from.
```

```text
between 24.1 and 25.4 ms, bottleneck hbm
sm        4.5 ms
hbm      16.5 ms
nvlink    0.9 ms
host      3.5 ms
```

`DecodeThreshold` is the engine's: a request scheduling at most that many tokens is priced as
a decode. In vLLM v0.31.0 the attention backend sets it: 1 by default, 128 on FlashMLA and 512
on FlashAttention-MLA. Under speculative decoding, a backend that runs speculative tokens as
decodes raises it to one more than their number, and with decode-context parallelism most
backends lower it to 1. A simulator should pass its engine's value. The examples here use 8, which
every one of those values classifies the same way for a one-token decode.

`Overlap` assumes the resources within a layer work at once and `NoOverlap` that they take
turns; when one number is wanted, take `NoOverlap`.
[Pricing a step](../concepts/step.md#which-estimate-to-use) explains both, and the five
resources.

## Build a variant without writing a file

`OpenInputs` loads the documents without building a kernel. Change them, and call `New`. A
configuration search does this to try many deployments without writing any of them to disk.

<!-- example_test.go:new: blis-vllm kerneltests/example_test.go -->
```go
// The new example lives in blis-vllm's kerneltests/example_test.go
// (https://github.com/inference-sim/blis-vllm/blob/main/kerneltests/example_test.go),
// which decodes the vLLM engine block it opens the kernel from.
```

```text
fp8:  35136 bytes of KV per token per rank
bf16: 70272 bytes of KV per token per rank
```

`New` copies what it needs, so changing the documents after it returns does not change the
kernel it returned.

## Read the memory

<!-- example_test.go:memory: blis-vllm kerneltests/example_test.go -->
```go
// The memory example lives in blis-vllm's kerneltests/example_test.go
// (https://github.com/inference-sim/blis-vllm/blob/main/kerneltests/example_test.go),
// which decodes the vLLM engine block it opens the kernel from.
```

```text
weights       78.2 GiB
activations    1.1 GiB
CUDA graphs    0.8 GiB
comm buffers   0.0 GiB
EPLB           0.0 GiB
total         80.1 GiB per rank
before the budget  3.9 GiB
a 4,096-token request: 137 MiB of KV cache per rank
```

[Memory](../concepts/memory.md) explains each figure.

## Read the evidence

<!-- example_test.go:evidence: blis-vllm kerneltests/example_test.go -->
```go
// The evidence example lives in blis-vllm's kerneltests/example_test.go
// (https://github.com/inference-sim/blis-vllm/blob/main/kerneltests/example_test.go),
// which decodes the vLLM engine block it opens the kernel from.
```

```text
115 of 157 values rest on evidence; 41 were assumed
  assumed       41
  measured     101
  not_charged    1
  vendor_spec   14
the kernel assumed max_cudagraph_capture_size
the kernel assumed engine_version
```

[Refuse, assume, override](../concepts/resolution.md) explains what these entries mean.

## When `New` refuses

`Open`, `OpenPool` and `New` return an error rather than a kernel for a deployment the engine
would not start, one the kernel cannot price truthfully, or one missing a coefficient it
needs. The message names the setting at fault. [What New refuses](../reference/refusals.md)
lists every case.

## Calling it often

A kernel holds no mutable state: it can be shared between goroutines, and the same batch
always prices the same. A step is cheap because everything that does not depend on the batch
is computed once, in `New`. A step then walks the model's distinct *kinds* of layer rather
than its layers, and multiplies by how many of each there are: three kinds instead of 108
layers for the deepest model in the catalog. On a laptop, `bench_test.go` measures a step at a
few hundred nanoseconds for one request and a few microseconds for a 256-request batch.

A caller pricing millions of steps can use `StepTimeInto`, which writes the per-resource
breakdown into a map the caller reuses instead of allocating one; the step then allocates
only a few small scratch slices.
