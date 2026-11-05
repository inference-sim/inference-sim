# The registry in BLIS

A latency estimate depends on three kinds of number, and BLIS keeps them apart:

Declared facts
:   What a vendor states about a model or a chip: layer counts, head dimensions, peak
    FLOP rates, memory bandwidth. Someone wrote them down; nothing about them is
    learned. They live in [blis-catalog](https://github.com/inference-sim/blis-catalog).

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

## Five repositories

| Repository | Holds | Relationship to the registry |
|---|---|---|
| [blis-schemas](https://github.com/inference-sim/blis-schemas) | Go types and validators; no data | Defines what a coefficient set may contain and rejects any file that breaks the rules. The registry's CI runs its validator on every pull request. |
| [blis-catalog](https://github.com/inference-sim/blis-catalog) | Models, chips, fabrics, workloads: declared facts | The registry's `scope.hardware` values are the catalog's chip names (`h200`, `gb200-nvl72`, …). |
| **blis-registry** | Coefficient sets: learned numbers with provenance | — |
| [blis-latency-kernel](https://github.com/inference-sim/blis-latency-kernel) | The cost model: prices one forward pass | Reads coefficient sets by name and evaluates its cost laws with them. Holds no numbers of its own. |
| [inference-sim](https://github.com/inference-sim/inference-sim) | The discrete-event simulator (BLIS) | Calls the kernel once per simulated step. The kernel-backed path is [inference-sim#1851](https://github.com/inference-sim/inference-sim/pull/1851), branch `kernel-exclusive` of the `sriumcp/inference-sim` fork. |

Dependencies run one way. `blis-schemas` depends on no other BLIS repository. The
catalog and the registry are data, validated against the schemas. The kernel imports
the schemas and reads the catalog and registry from disk at run time. The simulator
imports the kernel. Nothing imports the simulator.

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

(Abridged from [`testdata/kimi-k3-h100-nospec.yaml`](https://github.com/inference-sim/blis-substrate/blob/main/kernel/testdata/kimi-k3-h100-nospec.yaml)
in blis-latency-kernel.)

The kernel then does three things with the registry, once, before the first step:

1. **Loads** each named set from `<registry>/coefficients/<name>.yaml` with
   blis-schemas' strict loader, then validates it with the same schema rules the
   registry's CI runs. A file that fails either is refused before anything is priced.
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

## Why the learned numbers have their own repository

Because a learned number is only as good as the evidence behind it, and the evidence has
to travel with the number. Putting coefficients in a repository of their own means each
one can carry its method, its sources and its scope; can be re-derived and checked in CI
independently of the code that uses it; and can be changed by a pull request whose diff
shows exactly which numbers moved. A change here alters every estimate that uses the
affected sets the next time the files are read; no code is rebuilt.

Next: [what a coefficient records](coefficients.md).
