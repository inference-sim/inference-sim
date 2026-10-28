# Parallelism

<p class="lede">A deployment spreads a model over several GPUs in up to five ways at once.
Each way divides some of the work or memory and adds communication to pay for it. The kernel
prices one rank, so for each way it needs two answers: what this rank no longer has to do,
and what it now has to exchange with its group.</p>

A *rank* is one GPU's process in the deployment, numbered from 0. A *group* is the set of
ranks that share one kind of parallelism, and its *width* is how many ranks it holds; the
widths are written tp, dp, ep, pcp and dcp below.

| Parallelism | Divides | Costs |
|---|---|---|
| Tensor (tp) | Each layer's weights and attention heads | An all-reduce after the layers that split their output |
| Data (dp) | The requests: each replica serves its own | Nothing for dense layers; with experts, the replicas share them |
| Expert (ep) | A mixture-of-experts layer's experts, whole experts per rank | Sending each token to its experts and back |
| Prefill-context (pcp) | A prompt's tokens during prefill | Gathering the KV cache back onto every rank |
| Decode-context (dcp) | The KV cache, by token | Combining each rank's partial attention output |

## Tensor parallelism

Each rank holds a slice of every weight matrix and a share of the attention heads, and does
that share of the arithmetic. The one exception is in the kernel, not the engine: it charges
each rank the whole of a routed expert's arithmetic rather than its slice, a
[known over-charge](../reference/divergences.md) kept because it offsets a cost not yet
identified. The layers that split their output must add the partial
results back together, which is an all-reduce over the group, in every layer. The model graph
in the catalog states where these collectives fall, so the kernel prices exactly those.

It also divides the KV cache, but only down to one head per rank: a head is never split.
[Memory](memory.md) gives the consequence for latent attention, which has one head to begin
with.

## Data parallelism

Each replica schedules its own requests, so a batch the kernel prices is one replica's
batch. For a dense model the replicas share nothing at run time.

A mixture-of-experts model is different: vLLM shares the expert layers across the replicas
even when expert parallelism is off. Each expert is then sliced over every replica's ranks,
and every replica's tokens are gathered to the experts and the results scattered back. A rank
therefore holds a dp-th of the expert weights it would hold alone and receives dp times the
tokens for them, so its expert arithmetic is unchanged while its weights shrink and the gather
and scatter are added. The kernel charges it that way.

## Expert parallelism

Each rank holds whole experts rather than a slice of every one. The group spans the tensor,
prefill-context and data-parallel ranks together, tp × pcp × dp of them. Tokens must travel
to the ranks holding their experts and back, and those exchanges are priced as collectives
over the group. When the experts do not divide evenly, the ranks holding one extra are the
ones the step waits for, and the kernel prices those.

## Context parallelism

Two kinds, which divide different things.

**Prefill-context parallelism** splits a long prompt's tokens among ranks during prefill, so
each computes attention for its share. vLLM splits a prompt into twice as many chunks as there
are ranks and gives each rank one chunk from each end, which balances the attention work.
Where the tokens do not divide evenly the chunks differ in size, and the kernel charges the
busiest rank's pair, since the step waits for it. The KV cache is not
divided: every rank needs the whole cache afterwards, so the ranks all-gather what each
computed. vLLM v0.31.0 runs it only on models whose attention is latent (MLA), and the
kernel refuses it for any other.

**Decode-context parallelism** divides the KV cache itself, by token: each rank holds every
*n*th stretch of a request's context, so a longer context fits. During decode each rank
attends over its share and produces a partial result, and the group combines them. The
combine is the price of the capacity, and it is paid in every layer, every step. Which
collectives it runs depends on a setting, `dcp_comm_backend`, and the kernel prices exactly
the ones vLLM launches for it. The slowest rank, the one holding the most tokens, sets the
time.

## Which ranks a group contains

The cost of a collective depends on where its members are: within a node it runs over
NVLink, across nodes over the network fabric, which is far slower. vLLM numbers ranks in a
fixed order, with tensor parallelism innermost, so a group's members are evenly spaced, and
the spacing decides whether the group leaves its node.

--8<-- "docs/figures/ranks.html"

The kernel reproduces this numbering for every group it prices. A test builds every group the
way vLLM's own group construction does, rank by rank, and checks two things against the
kernel: whether the group leaves its node, and how many of its members share the busiest
node. It covers 4- and 8-GPU nodes, tensor widths 1, 2, 4 and 8, prefill-context widths 1, 2
and 4, data-parallel widths 1 and 2, and every decode-context width each of those admits.

## In the source

What each kind divides is resolved in
[`internal/resolve/layout.go`](https://github.com/inference-sim/blis-latency-kernel/blob/main/internal/resolve/layout.go).
Group placement is in `groupStride` and `crossesNodes` in
[`kernel.go`](https://github.com/inference-sim/blis-latency-kernel/blob/main/kernel.go), and
the context-parallel arithmetic is in the functions there whose names begin with `pcp` and
`dcp`.
