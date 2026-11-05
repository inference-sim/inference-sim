# Contributing

<p class="lede">How the repository is laid out, how to run its checks, and the conventions a
change is reviewed against. The conventions exist because a cost model that is wrong usually
still produces a plausible number.</p>

## Set up

```sh
git clone https://github.com/inference-sim/blis-latency-kernel
cd blis-latency-kernel
scripts/fetch-testdata.sh
go test ./...
```

The fetch script brings in the catalog and registry at the commits the tests are pinned to;
`testdata/README.md` describes what else is in `testdata/` and why some of it is not
committed. CI runs the same tests on every pull request, with `gofmt` and `go vet`.

## Layout

| Path | Holds |
|---|---|
| `kernel.go` | The interface methods and the composition of a step |
| `new.go` | `New`: validation, resolution, coefficient lookup, everything computed once |
| `sparse_mla.go` | Which sparse-attention backend vLLM chooses, and what that decides |
| `open.go` | `Open`, `OpenPool`, `OpenInputs`: from a scenario file to the documents |
| `internal/price` | The cost laws, as pure functions |
| `internal/resolve` | From a deployment to a layout: widths, groups, which collectives run |
| blis-vllm `kerneltests/` | The commands on [Commands](../reference/commands.md), their harness, the generator for `docs/generated/`, and the tests that price vLLM engine blocks. They decode vLLM's configuration, so they live with the engine model and this module imports nothing from it |

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
`internal/docgen/summaries.go`; the generator refuses a note without one.

**Test behavior, and see the test fail.** A test asserts what the kernel does for an input,
not how its code is arranged. [How the laws are tested](../research/testing.md) describes the
kinds of test used. Break a new law on purpose and confirm a test fails, and say in the commit
what you broke.

**Measure a change's effect.** A change to pricing reports how it moves `cmd/shape` and
`cmd/score` in its commit message, with the figures before and after. Those commands need the
measurement corpora, which are not in the repository.

## Documentation

[Documentation and releases](documentation.md) describes the site, how its generated parts are
kept current, and how a release publishes it.

## Changes across the repositories

A change that also touches blis-schemas, the catalog, the registry or inference-sim follows
[Changes across the repositories](cross-repo-changes.md), which links the canonical process and
lists the kernel's own steps.
