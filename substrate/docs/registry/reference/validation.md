# Validation

Three kinds of check guard the registry. They answer different questions, and passing
one says nothing about the others.

| Check | Question it answers | Where it runs |
|---|---|---|
| Schema | Could a consumer load this file? | CI, every pull request and every push to `main` |
| Properties | Is every value physically sensible, and is its provenance complete? | CI, every pull request and every push to `main` |
| Re-derivation | Does every value still come out of its fitter, from its cited data? | Locally, with an AISimulate clone; skipped in CI |

The documentation build is a fourth, narrower check: it fails if a set has no reference
page, or if a set holds data the reference tables cannot represent faithfully.

## Schema

The `schema-validate` job runs blis-schemas' own validator over every file under
`coefficients/`:

```sh
go run ./cmd/validate-registry /path/to/blis-registry   # from a blis-schemas checkout
```

It applies exactly the rules a consumer applies when it loads a set: strict decoding,
then field validation. The rules are listed on [File format](format.md). The registry
keeps no schema validator of its own, so there is one definition of a valid file, and
it is the one the kernel uses.

blis-schemas is pinned to a release tag in `.github/workflows/validate.yml`. A schema
release changes nothing here until the pin is moved.

## Properties

`validator/test_coefficient_properties.py` reads only the committed files and a vendored
copy of the catalog's chip descriptors, so it always runs. Each test encodes a defect
that this project has shipped at least once:

- no bandwidth rate exceeds its part's datasheet peak, and every rate is a plausible
  fraction of it;
- GEMM asymptotes are fractions; MoE imbalance is near one, and its 90th percentile is
  not below its median;
- collective floors do not fall as the group widens, rates do not rise steeply, and no
  transition rate exceeds its peak rate;
- recurrent floors are positive and bounded, and rates are slower than the convolution
  alone;
- a kind-specific attention floor is not a module measurement in disguise;
- the hardware-independent sets cover every catalog part;
- every `measured` entry cites a source, and every `assumed` entry explains itself;
- the memory set has a capture entry for every vLLM CUDA-graph mode, respects the
  capture-pool overlay, and its activation multiples fall with width; and the capture
  sizes re-derive from the committed log samples.

## Re-derivation

`validator/test_generated_from_aisimulate.py` and `validator/test_writers_are_idempotent.py`
re-run the fitters and compare with the committed files. They need the AISimulate sweeps,
which are not vendored here, and they skip when the data is absent, saying so. CI has no
copy, so in CI they always skip.

Run them before any pull request that changes a fitted value, and before a release:

```sh
export AISIMULATE_DATA=/path/to/aisimulate/python/aisimulate/src/aisimulate_core/systems/data
export BLIS_CATALOG=/path/to/blis-catalog
export COLLECTIVEX_DATA=/path/to/collectivex-run-bodies
scripts/rederive.sh
```

`scripts/rederive.sh` is also what the manual-dispatch `rederive` workflow runs.

They check that:

- each writer reproduces its committed file byte for byte (`--check`), and every fitted
  family has a writer;
- the generated GEMM and collective values survive an independent re-derivation by a
  second code path;
- the all-reduce entries come from the vLLM CUDA-graph lane specifically, since the eager
  lane is slower than NCCL and accepting it would misprice all-reduce by 3.4×;
- in the primitive and collective sets, every source is NVIDIA's, and no `assumed`
  entry carries a foreign measurement in through its rationale.

A test that cannot run says why. Treat an unexpected skip as a test that did not run.

## The end-to-end check

None of the above can tell whether a change makes BLIS more accurate. That needs the
evaluation corpus and the scorer on the `kernel-exclusive` branch of the
`sriumcp/inference-sim` fork (inference-sim#1851), and is
described in [methodology §6.1](../methodology.md#61-reproducing-the-end-to-end-evaluation-tables).
[`evaluation-baseline.json`](../evaluation-baseline.json) records the last recorded
figures, with the scorer and catalog revisions they were scored at.
