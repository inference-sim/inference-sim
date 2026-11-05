# The registry in BLIS

A latency estimate depends on three kinds of number, and BLIS keeps them apart:

Declared facts
:   What a vendor states about a model or a chip: layer counts, head dimensions, peak
    FLOP rates, memory bandwidth. Someone wrote them down; nothing about them is
    learned. They live in the [catalog](../../catalog/index.md).

Learned numbers
:   What a cost model needs but no datasheet states: the latency floor of a decode
    attention kernel on an H200, the bandwidth an 8-rank all-reduce actually reaches,
    the token count at which a GEMM gets halfway to peak efficiency. They are obtained by
    fitting measurements, or assumed where no measurement exists. They live here.

Deployment choices
:   What a particular run selects: the GPU, the parallelism, the engine settings. They are
    stated per run, in a scenario, and committed nowhere.

The test for which side of the first line a number falls on is simple: **could a
measurement revise it?** A chip's SM count cannot be revised by a benchmark; its
achievable bandwidth can. The first belongs in the catalog, the second in the registry.

## Where the registry sits

The registry is one directory of blis-substrate, beside the parts it works with:

| Directory | Holds | Relationship to the registry |
|---|---|---|
| `contracts/` | Go types and validators; no data | Defines what a coefficient set may contain. `validate-registry` rejects any file that breaks the rules, on every pull request. |
| `catalog/` | Models, chips, fabrics, workloads: declared facts | The registry's `scope.hardware` values are the catalog's chip names (`h200`, `gb200-nvl72`, …). |
| `registry/` | Coefficient sets: learned numbers with provenance | — |
| `kernel/` | The cost model: prices one forward pass | Reads coefficient sets by name and evaluates its cost laws with them. Holds no numbers of its own. |

The catalog and the registry are data, validated against the contracts. The cost model
reads both through one loader, from the copy embedded in each release or from a directory
given by path (design §3.3). The simulator, inference-sim, is the only repository that
imports the cost model and the data.

## What happens at run time

A scenario names everything a run needs, including which coefficient sets to use:

```yaml
kind: Scenario
name: kimi-k3-h100-nospec
engine_version: "0.29.0"
model: kimi-k3
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent]
cluster:
  hardware: h100
  ...
```

(Abridged from `testdata/kernel/kimi-k3-h100-nospec.yaml` in inference-sim.)

The kernel then does three things with the registry, once, before the first step:

1. **Loads** each named set from `<registry>/coefficients/<name>.yaml` with the
   strict loader in `contracts/`, then validates it with the same rules the
   repository's CI runs. A file that fails either is refused before anything is priced.
2. **Filters** by scope. An entry applies only if its scope admits the deployment: an
   entry scoped to `hardware: [h200]` is skipped on an H100. A dimension the scope does
   not mention is unconstrained.
3. **Resolves** names. Sets apply in the order the scenario lists them; if two admitted
   entries share a name, the later one wins.

The result is a fixed table of numbers the kernel's cost laws read for the rest of the
run.

What happens when a name is absent is the kernel's decision, made term by term in its
[`new.go`](https://github.com/inference-sim/blis-substrate/blob/main/kernel/new.go). Some
lookups are required and fail with an error naming the missing coefficient. Many others
fall back, usually to zero, so a part with no entry for a family has that term priced at
nothing, and the run still completes. This is why coverage matters: the
[recurrent set](../reference/sets/cost-model-recurrent.md) fits every part its sweeps
measure for exactly this reason.

## Why the learned numbers are kept apart from the code

Because a learned number is only as good as the evidence behind it, and the evidence has
to travel with the number. Each coefficient carries its method, its sources and its scope;
can be re-derived and checked in CI on its own; and changes by a pull request whose diff
shows exactly which numbers moved. Because the simulator can read the registry from a
directory instead of its embedded copy, a candidate registry is scored with the released
simulator and no rebuild.

Next: [what a coefficient records](coefficients.md).
