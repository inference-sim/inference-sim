# Contributing

<p class="lede">How the repository is laid out, how to run its checks, and the conventions a
change is reviewed against. The conventions exist because a cost model that is wrong usually
still produces a plausible number.</p>

## Set up

```sh
git clone https://github.com/inference-sim/blis-substrate
cd blis-substrate
go test ./kernel/...
```

The tests read the repository's own `catalog/` and `registry/`, so the kernel is always tested
against the data it is released with. `BLIS_CATALOG` and `BLIS_REGISTRY` name other
directories to test against a candidate. CI runs the same tests on every pull request that
changes `kernel/` or `contracts/`, with `gofmt`, `go vet` and a race test.

## Layout

| Path | Holds |
|---|---|
| `kernel/kernel.go` | The interface methods and the composition of a step |
| `kernel/new.go` | `New`: validation, resolution, coefficient lookup, everything computed once |
| `kernel/sparse_mla.go` | Which sparse-attention backend vLLM chooses, and what that decides |
| `kernel/open.go` | `Open`, `OpenPool`, `OpenInputs`: from a scenario file to the documents |
| `kernel/internal/price` | The cost laws, as pure functions |
| `kernel/internal/resolve` | From a deployment to a layout: widths, groups, which collectives run |
| blis-vllm `kerneltests/` | The commands on [Commands](../reference/commands.md), their harness, the generator for `docs/kernel/generated/`, and the tests that price vLLM engine blocks. They decode vLLM's configuration, so they live with the engine model, and the kernel imports nothing from them |

## Conventions

**Cite the engine.** A statement about what vLLM does names the lines it rests on, as
`vllm/<path>:<lines>`, at v0.31.0. A reviewer can then check it in one command.

**Refuse, or say so.** A deployment the engine would not start, or the kernel cannot price
truthfully, is refused with an error that names the setting. A default filled in on the
deployment's behalf is recorded with `assume`; a coefficient the kernel can do without is read
through `optional`, which records its absence. Nothing is defaulted silently.

**Mark what is knowingly wrong.** A term priced differently from the engine on purpose, or
left out, carries a comment marked `KNOWN DIVERGENCE`, `KNOWN over-charge` or `COVERAGE LIMIT`,
with the reason and the engine lines. Those notes are listed on
[Known divergences](../reference/divergences.md), each under a plain summary kept in
blis-vllm's `kerneltests/internal/docgen/summaries.go`; the generator refuses a note without
one.

**Test behavior, and see the test fail.** A test asserts what the kernel does for an input,
not how its code is arranged. [How the laws are tested](../research/testing.md) describes the
kinds of test used. Break a new law on purpose and confirm a test fails, and say in the commit
what you broke.

**Measure a change's effect.** A change to pricing reports, in its pull request, the scoring
harness's figures on published units before and after it. End-to-end figures come from
accuracy-dashboard's scoring harness and from nothing else.

## Evidence a kernel pull request carries

- the vLLM lines each pricing statement rests on, at the modelled release;
- for a change that moves a published estimate, the held-out and gate evidence that
  [Changes across repositories](../../contributing/changes.md) requires;
- for a coefficient the change reads or adds, the registry's
  [fitting principles](../../registry/fitting-principles.md).

## Documentation

[Documentation and releases](../../contributing/documentation.md) describes the site, how its
generated parts are kept current, and how a release publishes it.

## Changes across repositories

A change to the kernel and to the contracts, the catalog or the registry is one pull request in
this repository. A change that also touches blis-vllm or inference-sim follows
[Changes across repositories](../../contributing/changes.md).
