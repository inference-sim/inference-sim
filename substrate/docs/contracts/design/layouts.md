# Parallel layouts

<p class="lede">A pool's <code>parallel</code> block says how one engine spreads over GPUs.
Some widths are stated, one is derived, and a few checks decide whether the result could be
launched at all. Each check follows from how the engine builds its process groups, and each
refuses only layouts that no launch could run.</p>

## The widths

A pool states five widths: tensor (`tp`), pipeline (`pp`), data (`dp`), prefill-context
(`pcp`) and decode-context (`dcp`). An engine has one rank per GPU, and its rank count is

<div style="text-align:center" markdown>
**ranks = pp × tp × pcp × dp**
</div>

Neither decode-context nor expert parallelism adds ranks. Decode-context parallelism reuses
the ranks that tensor and prefill-context parallelism create, and an expert group is built
over ranks that already exist.

The ranks must fit in the GPUs of the pool's nodes. This is an upper bound and not an
equality, because a pool may hold several engines. llm-d, a Kubernetes stack for serving
with vLLM, runs each engine as one replica of a group of pods. A 36-node prefill pool of
8-GPU nodes at `tp 1`, `pp 1`, `pcp 1`, `dp 8` is 36 engines of one node each, every one
with the pool's layout.

## Expert-parallel width is derived

There is no `ep` key. With `enable_expert_parallel` set, the expert-parallel width is

<div style="text-align:center" markdown>
**ep = tp × pcp × dp**
</div>

because the engine builds its expert group over the data, prefill-context and tensor axes
together. If a document could state the width separately, it could state one the engine
would never build. Without expert parallelism the width is 1.[^history]

[^history]: An earlier version of the schema used `tp × max(dp, pcp)`. The two agree
    whenever `dp` or `pcp` is at most 1, which was every layout the schema then admitted.
    They differ once both exceed 1: at `tp 1`, `dp 4`, `pcp 8`, which llm-d deploys, the
    expert group has 32 ranks, not 8.

`ExpertParallelWidth()` saturates at the largest integer rather than wrapping, so an absurd
layout reads as an absurdly wide group and not a small or negative one.

## Local replicas on a node

`dp_local` states how many data-parallel replicas live on one node. It must divide
`gpus_per_node`, and the engine places replica *i* on a run of devices starting at
*i* × world, where world = pp × tp × pcp.

--8<-- "docs/contracts/figures/placement.html"

Take a pool of two 8-GPU nodes at `tp 4`, `dp 4`, `dp_local 4`. Its 16 ranks fit in the
pool's 16 GPUs. The engine's placement would still put replicas 2 and 3 past the end of the
node, so the layout is refused:

```text
error: deployment.pools[0].parallel.dp_local: 4 local replicas of 4 GPUs (pp 1 x tp 4 x pcp 1) are placed one after another from device 0, so the last would start at device 12, but a node has 8
```

With `dp_local: 2` the same pool validates.

The full check is more general than the figure. When an engine spans several nodes, it
splits each replica over a number of nodes that must divide the world size, and the
replica's share on one node shrinks accordingly. The validator looks for any split the
pool's nodes allow under which every local replica's share ends on the node, and refuses
the layout only if there is none. With `dp_local` unset the engine infers it from the
launch, and no per-node verdict is given.

## Decode-context parallelism

`dcp` has no ranks of its own. It must partition the group it reuses:

- With `pcp` at most 1, it reuses the tensor-parallel ranks, and `tp` must be divisible by
  `dcp`.
- With `pcp` above 1, it may be 1, span the prefill-context axis (`pcp`), or span the whole
  `tp × pcp` block. Any other width would straddle the two axes. The message names the
  widths that would work:

```text
error: deployment.pools[0].parallel.dcp: with prefill-context parallelism enabled, decode-context parallelism must be disabled, span the pcp axis, or span the full tp x pcp axis; got tp 8, pcp 2, dcp 4, so the admissible widths are [1 2 16]
```

These are field checks, not rules, because they follow from how the group is built and
hold in every release.

## What depends on the release

Some layout questions have different answers in different releases, and belong to a rules
pack:

- vLLM 0.29.0 refuses prefill-context parallelism combined with data parallelism; 0.31.0
  supports it. The `pcp-excludes-data-parallelism` rule reports it for 0.29.0, and the
  0.31.0 pack has no such rule.
- With expert-parallel load balancing on, the expert-parallel width must divide the experts
  plus the redundant experts, or the engine does not start. Without it, an uneven split runs
  with a static imbalance, which the rules report as a warning with its ratio, and a width
  larger than the number of experts is an error.
- Whether the custom all-reduce GPU kernel is used depends on the release's supported
  world sizes.

See [Rules packs](../reference/rules.md).

## Exact arithmetic

The rank-fit, placement and decode-context checks compute their counts with
arbitrary-precision integers. An absurd width must still compare correctly with an absurdly
large pool, and a message must never print a clamped number as if it were the real one. The placement search is bounded so that
no input can make validation hang. A layout that reaches the bound is reported as
undecided, with a warning, rather than accepted or refused without proof; no layout a real
cluster could hold comes near it.

Source: [`spec/deployment/validate.go`](https://github.com/inference-sim/blis-substrate/blob/main/contracts/spec/deployment/validate.go).
