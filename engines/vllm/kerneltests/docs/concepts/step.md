# Pricing a step

<p class="lede">A step is one forward pass of the model over a batch of requests. The kernel
prices it one layer at a time: for each layer it works out how long each resource of the GPU
is busy, takes the largest of those times as the layer's time, and adds the layers up. It
reports the result as two estimates, which differ in how much they let the resources
overlap.</p>

## What a batch is

A batch is a list of requests, and for each request three token counts: how many tokens it
schedules this step, how many it has already computed, and how long its prompt is. Each
request keeps the attention keys and values of every token it has processed, so that later
tokens need not recompute them; this store is the *KV cache*.

The engine classes each request by a *decode threshold*, a small number of tokens: 1 by
default in vLLM, larger under speculative decoding and on some attention backends. A request
scheduling more tokens than the threshold is a *prefill*: it is still reading its prompt, and
its attention covers the new tokens and everything before them. A request scheduling at most
the threshold is a *decode*: it is generating, usually one token per step, and its attention
reads the whole of its cached context.

The two cost differently. A prefill does a great deal of arithmetic per token. A decode does
little, but reads its context from memory every step. Most of what follows is about counting
those two things.

## The resources

The kernel charges every piece of work to one of five resources, and keeps a running total
for each.

| Resource | What is charged to it |
|---|---|
| SM | Arithmetic on the GPU's streaming multiprocessors: matrix multiplies, prefill attention, recurrent mixers |
| HBM | Reading the GPU's memory: weights, the KV cache during decode, normalizations |
| NVLink | Collectives among GPUs on the same node |
| NIC | Collectives that cross nodes, over the network fabric |
| host | CPU time: dispatching GPU kernels, and launching or replaying CUDA graphs |

A *collective* is an operation in which a group of GPUs exchange data, such as an all-reduce
that sums each GPU's partial result. A *recurrent mixer* is the state-space or linear-attention
layer of a hybrid model, which carries a fixed-size state from token to token instead of a KV
cache. A *CUDA graph* is a recorded sequence of GPU kernels that the CPU can replay with one
call instead of launching each.

## One layer

A model graph lists each kind of layer once, with how many times it repeats. For one layer
the kernel computes:

**Arithmetic.** The floating-point operations of each matrix multiply, divided by the chip's
peak rate times an efficiency. The efficiency is the fraction of peak a matrix multiply
reaches, and it rises with the number of tokens \(m\) in the step:

\[
\varepsilon(m) = \varepsilon_{\max}\,\frac{m}{m + m_{1/2}}
\]

At \(m = m_{1/2}\) it is exactly half of \(\varepsilon_{\max}\). Both are coefficients
measured per chip and data type. Using the peak rate instead would price the matrix multiplies
of a 512-token step between 1.2 and 2.8 times too fast, depending on the chip and data type.
Where the registry also measures how efficiency depends on a matrix multiply's width and
depth, the kernel multiplies in a factor of the same form for each, so that every matrix
multiply in a layer runs at its own efficiency.

**Memory.** The bytes the layer reads from HBM, divided by the chip's memory bandwidth times a
measured fraction (the *derate*), because no GPU kernel reaches the datasheet bandwidth. The weights are read once per step whatever the batch size,
so one read is shared by every request in the batch.

**Attention.** Decode attention is priced as a read of the cached context; prefill attention
as arithmetic, which grows with the number of query-key pairs: with the square of the prompt,
for a prompt read in one step. Each has a measured fixed cost per layer, and decode a measured
rate for each kind of attention: full, sliding-window, latent (MLA) or sparse latent. A layer
that states a window reads at most the window, whatever its kind; a sparse layer reads its
window plus the positions its indexer selects, at most its top-k, or, where it selects
nothing, its whole compressed stream, in decode and in prefill alike; and a sparse layer's
indexer scan is priced in both phases.

Where the registry carries a model's own measured prefill form for a kind of attention or for
an indexer (DeepSeek-V4.1's sparse latent attention and its block-index scorer), prefill is
priced with it instead: a fixed cost, a cost per prefill request and per prefill token, and the
bytes the kernel reads in its own blocks at a measured rate. Each such form is used only when
every one of its terms is present.

**Hyper-connections.** DeepSeek-V4.1's mHC kernels are priced, where the registry has their
measured form, as a fixed cost plus the residual streams' bytes at a measured rate for each of
their two halves, in place of the bytes the graph states and of the small mixing GEMM the
measured kernel already includes. Without it they are charged as memory traffic, as before.

**Arithmetic rates.** Each matrix multiply runs at the rate of the format it computes in:
BF16 weights at the BF16 rate, FP32 weights at the TF32 rate, MXFP8 on Hopper and four-bit
weights on any part without a four-bit peak at the BF16 rate of the weight-only kernels vLLM
uses for them, the grouped four-bit experts on their own measured efficiency. A node the graph marks replicated is not divided by the tensor-parallel width.

**Recurrent mixers.** Decode rows are priced at the measured decode rate; prefill tokens at a
measured prefill rate, or, where the registry has none, as the chunked scan vLLM runs for them.

**Speculative decoding.** When the deployment speculates and the graph states its drafter,
each step also pays the drafter's passes, and the drafter's weights are resident.

**Experts.** In a mixture-of-experts layer each token is routed to a few experts. The kernel
estimates how many distinct experts a batch touches, reads their weights, and prices their
arithmetic. Unlike every other term, the arithmetic and the reads of the experts are *added*
rather than overlapped, because the sum fits measured expert timings better than the larger
of the two.[^experts]

[^experts]: Against NVIDIA's expert-GEMM measurements on four chips, the sum fits better on
    every one. The comparison is recorded beside the composition in `step.go`.

**Collectives.** Each collective the layer runs, an all-reduce after a tensor-parallel matrix
multiply for instance, is priced from three coefficients measured for that operation, that
number of GPUs and that chip: a fixed cost \(t_0\), a rate \(r_t\) over the range a step's
messages fall in, and the highest rate reached on large messages, \(r_\text{peak}\):

\[
t(b) = \max\!\left(t_0 + \frac{b}{r_t},\; \frac{b}{r_\text{peak}}\right)
\]

for a message of \(b\) bytes. Every launch pays its own fixed cost. A collective whose group
spans nodes is charged to NIC; one within a node, to NVLink. On a rack-scale NVLink system
such as GB200 NVL72, a group within one rack stays on NVLink.

## Composing the step

Write \(T^\ell_r\) for the time layer \(\ell\) needs on resource \(r\), not counting its
experts, and \(E^\ell\) for its expert time, arithmetic plus reads. The first estimate
assumes the resources of a layer work at the same time, so the layer takes as long as its
busiest one. The next layer needs this one's output, so layers do not overlap one another:

\[
\text{Overlap} = \sum_{\ell} \Big(\max\big(T^\ell_\text{SM}, T^\ell_\text{HBM},
T^\ell_\text{NVLink}, T^\ell_\text{NIC}\big) + E^\ell\Big) + H + T_\text{host}
\]

where \(H\) is the output head, which runs once per step and takes the larger of its
arithmetic and its reads, and host time is added last: a GPU kernel cannot start before the
CPU has dispatched it.

The second estimate assumes the resources take turns. It is the sum of the five resources'
totals over the whole step, where each total includes the experts' and the head's share:

\[
\text{NoOverlap} = T_\text{SM} + T_\text{HBM} + T_\text{NVLink} + T_\text{NIC} +
T_\text{host}
\]

Overlap can never exceed NoOverlap. The two differ most when the resources are evenly
loaded, and least when one dominates.

The *bottleneck* is the resource with the largest total: the one to relieve first. It
changes with the batch size within one deployment, as the figure shows.

--8<-- "docs/generated/step-anatomy.html"

## Which estimate to use

Neither estimate is a bound on the measured step. They are two models of how the work
overlaps, and on measured data both come out too fast, NoOverlap by much less. When a caller
wants one number, it should take **NoOverlap**. [Accuracy](../research/accuracy.md#whole-step-time)
gives the measurements. The likely reason the resources overlap so little is how vLLM
captures CUDA graphs: in piecewise mode, attention runs outside the captured graph, between
its pieces.

inference-sim takes NoOverlap as its step time.

## Host time

The CPU's work per step depends on how much of the step was captured as a CUDA graph, and
the engine decides that per batch. The kernel follows vLLM v0.31.0's choice:

- With the whole step captured, the host pays one graph replay.
- With *piecewise* capture, the graph is split at each attention or recurrent mixer (and, on a
  sparse-attention layer, at its indexer too), and the host pays one replay per piece.
- With no capture, or for a batch larger than the largest captured size, the host launches
  every kernel eagerly, at the eager launch cost per kernel. A batch past the largest
  captured size is never charged less than the same batch captured, so adding a request
  never lowers the price.
- A model whose own code breaks the capture, as DeepSeek-V4.1 does around every layer's
  attention, replays one piece per captured region under piecewise capture. Full capture is
  unaffected: inside it the breaks are not taken, and the step is one replay.

- A parallel-block drafter (DSpark) has CUDA graphs of its own, captured only when the
  target's decode mode is full, and its own ceiling: its rows are `num_spec_tokens` wide, not
  `1 + num_spec_tokens`, so under a 64-token capture with five draft tokens it replays a graph
  up to 12 requests while the target does up to 10. Past its ceiling, or with no graph, its
  kernels launch eagerly; a step with a prefill row also stores the drafter's extra context
  eagerly before the replay.

On top of that, every kernel the step runs costs a fixed dispatch time. A step with no
prefill row does not run the nodes the model graph marks `phase: prefill`, so it neither
launches nor dispatches them. For a single request
the host can be the largest resource, as the figure above shows.

A step costs microseconds to price, because everything that does not depend on the batch is
computed once, when the kernel is built; [Price a batch from Go](../guides/go.md#calling-it-often)
says more.

## In the source

The step is composed in `stepTime` in [`step.go`](https://github.com/inference-sim/blis-latency-kernel/blob/main/step.go),
which prices each layer in `priceLayers`,
and the laws it uses are pure functions in
[`internal/price`](https://github.com/inference-sim/blis-latency-kernel/tree/main/internal/price).
Each term's comment records the measurement that justified its form.
