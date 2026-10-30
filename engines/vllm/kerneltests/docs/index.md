# blis-latency-kernel

<p class="lede">blis-latency-kernel is the cost model of BLIS. Given a model, a GPU, a set of
coefficients for that GPU and a deployment, it answers the two questions a serving simulator asks:
how long one forward pass over a batch of requests takes, and how much GPU memory the
deployment holds.</p>

It is a Go library. A simulator builds one kernel per *pool*, a set of GPUs running one engine
configuration (a disaggregated deployment has a prefill pool and a decode pool), then calls it
once per simulated step. The kernel prices the step from first principles: it counts the
arithmetic, the bytes read from memory and the bytes moved between GPUs, and turns each count
into time with coefficients for the hardware, most of them measured. It holds no data of its own; the model
and the hardware come from [blis-catalog](https://github.com/inference-sim/blis-catalog), and
the coefficients from [blis-registry](https://github.com/inference-sim/blis-registry).

--8<-- "docs/figures/lifecycle.html"

## A first example

The deployment is DeepSeek-V3 in fp8 on eight H200 GPUs at tensor-parallel width 8.[^sglang]
The batch is 32 requests, each decoding one token against a 4,096-token context.

[^sglang]: The scenario's file name includes `sglang` because it comes from NVIDIA's
    AISimulate measurements, which ran that engine. The kernel prices vLLM v0.31.0's behavior
    for whatever deployment it is given.

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

The first number assumes the GPU's resources (arithmetic, memory, interconnect and CPU) work
at the same time within each layer; the second assumes they take turns. For one number, use the
second, `NoOverlap`; [Pricing a step](concepts/step.md#which-estimate-to-use) says why. NIC,
the network between nodes, is absent because no collective here leaves the node. The output above
is checked by `go test`, so it is what this version computes.

## Where to go next

**To use the kernel**, start with [Price a batch from Go](guides/go.md), then
[Use the kernel from a simulator](guides/simulator.md).

**To understand what it computes**, read the [Concepts](concepts/index.md) in order. They
explain how a step is priced, how memory is counted, and what each kind of parallelism
changes, without assuming you have read the code.

**To judge how far to trust it**, read [Accuracy](research/accuracy.md) and the
[known divergences](reference/divergences.md): the places where it knowingly prices
something differently from the engine it models.

**To change it**, read [Contributing](contributing/index.md).

## What it is not

It is not a simulator. It does not queue requests, form batches or decide when a request is
admitted; those depend on time, and the kernel is a pure function of its inputs. It prices
whatever batch it is given.

It is not a lookup table. It evaluates closed-form laws, so it can price a model or a shape
that was never measured. The same property bounds its accuracy against a table on shapes
that were.

It does not know model names. It reads a model as a graph of a few kinds of operation, and
dispatches on those. A new model reaches it as data, with no change to its code.

## Versions

This kernel prices the engine behavior of vLLM v0.31.0: which kernels run, how work is split
across GPUs, what an unstated setting defaults to. The exact pins of vLLM and of every other
input are on [Sources of truth](reference/sources.md). These pages are published per release
of the kernel; the selector at the top of the page switches between them.
