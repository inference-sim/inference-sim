# Validate a catalog or registry

<p class="lede">Two commands check every committed file in a data repository. The catalog
and the registry run them in CI, pinned to a release of this module. Run the same release
locally and a change that passes will pass in CI too.</p>

You need Go 1.24 or later. Neither command needs a checkout of blis-schemas. The examples
use `@latest`; to match CI, use the release its workflow pins instead.

## The catalog

Point `validate-catalog` at the root of a blis-catalog checkout:

```sh
go run github.com/inference-sim/blis-schemas/cmd/validate-catalog@latest ~/src/blis-catalog
```

It walks `models/`, `hardware/`, `networks/`, `devices/` and `workloads/`, prints a line for
each file that validates, and ends with a count:

```text title="Output, abbreviated"
# models/*/graph.yaml
codellama-34b-instruct-hf                       48 layers, 1 kind(s), bf16
deepseek-v3                                     61 layers, 2 kind(s), fp8
...
# hardware/*.yaml
h100                                           bf16 989.5, fp8 1979 TFLOP/s, 80 GiB
...
# workloads/*.yaml
chatbot                                        prompt~256, output~256 tokens
...

all 120 artifact(s) validate
```

When something is wrong, each problem goes to standard error, under the name of the file
it came from. Run from the catalog root as `validate-catalog .`, a failing run writes this to
standard error:

```text
slip: error: SMCount: must be positive; every chip must declare its SM count
slip: error: TFlopsFP8: FP8 peak 900.0 does not exceed BF16 peak 989.0; check the transcription
typo: decoding hardware/typo.yaml: unknown field(s) [_mfu]; this type declares [BwPeakTBs GPUsPerNode GPUsPerRack IntraNodeBwGBps IntraRackBwGBps MemoryGiB Provenance SMCount TFlopsFP8 TFlopsNVFP4 TFlopsPeak]

2 of 3 artifact(s) failed
```

The first file has two mistakes, and both are reported. The second has a key the chip
schema does not declare, so it does not load; the error lists the keys it could have used.

What each kind of file must contain is in the reference:
[model graph and identity](../reference/model.md), [chip, fabric and storage
classes](../reference/hardware.md), [workload shape](../reference/workload.md#shape). Each
model's `config.json` is checked only for being a non-empty JSON object.

## The registry

Point `validate-registry` at the root of a blis-registry checkout:

```sh
go run github.com/inference-sim/blis-schemas/cmd/validate-registry@latest ~/src/blis-registry
```

It finds every `.yaml` and `.yml` file under `coefficients/`, at any depth, and validates
each as a coefficient set:

```text title="Output, abbreviated"
cost-model-attention.yaml                       60 coefficient(s)
cost-model-collectives.yaml                    864 coefficient(s)
...

all 6 set(s) validate
```

A set that fails prints `FAILED` on standard output and its problems on standard error.
What a set must contain is in [Coefficient set](../reference/coefficients.md).

Both commands exit with `0` when everything validates, `1` when something fails, and `2`
when the path is not a catalog or registry or the command was misused. The details are in
[Command line](../reference/cli.md#exit-codes).

## In CI

Pin a release, so that a new release of blis-schemas cannot fail an unrelated pull
request. The catalog's workflow does this:

```yaml
- uses: actions/setup-go@v5
  with:
    go-version: '1.24'
- name: Validate every committed catalog entry
  run: go run github.com/inference-sim/blis-schemas/cmd/validate-catalog@vX.Y.Z .
```

Replace `vX.Y.Z` with the release you have tested against. Moving to a newer one is then a
one-line change. The releases are listed on
[GitHub](https://github.com/inference-sim/blis-schemas/releases).
