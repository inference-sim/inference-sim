# Validate the catalog and the registry

<p class="lede">Two commands check every committed file in <code>catalog/</code> and
<code>registry/</code>. CI runs them on every pull request, from the same commit as the
types they check. Run them locally from the root of a blis-substrate checkout, and a change
that passes will pass in CI too.</p>

Both commands take the directory to check. It may be the repository's own `catalog/` or
`registry/`, or any directory with the same layout, such as a candidate registry a fitting
script wrote.

## The catalog

```sh
go run ./validate/cmd/validate-catalog catalog
```

It walks `models/`, `hardware/`, `networks/`, `devices/`, `workloads/` and `measurements/`,
prints a line for each file that validates, and ends with a count:

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

all 151 artifact(s) validate
```

When something is wrong, each problem goes to standard error, under the name of the file
it came from. Run over a directory holding two broken chip files, a failing run writes this
to standard error:

```text
slip: error: SMCount: must be positive; every chip must declare its SM count
slip: error: TFlopsFP8: FP8 peak 900.0 does not exceed BF16 peak 989.5; check the transcription
typo: decoding hardware/typo.yaml: unknown field(s) [_mfu]; this type declares [BwPeakTBs ComputeCapability GPUsPerNode GPUsPerRack IntraNodeBwGBps IntraRackBwGBps MemoryGiB Provenance SMCount TFlopsFP8 TFlopsNVFP4 TFlopsPeak]

2 of 2 artifact(s) failed
```

The first file has two mistakes, and both are reported. The second has a key the chip type
does not declare, so it does not load; the error lists the keys it could have used.

What each kind of file must contain is in the reference:
[model graph and identity](../reference/model.md), [chip, fabric and storage
classes](../reference/hardware.md), [workload shape](../reference/workload.md#shape),
[measured run](../reference/measurement.md). Each model's `config.json` is checked only for
being a non-empty JSON object.

## The registry

```sh
go run ./validate/cmd/validate-registry registry
```

It finds every `.yaml` and `.yml` file under `coefficients/`, at any depth, and validates
each as a coefficient set, then the sets under `acceptance/` and `routing/`:

```text title="Output, abbreviated"
cost-model-attention.yaml                       90 coefficient(s)
cost-model-collectives.yaml                    888 coefficient(s)
...
acceptance/inferencex-golden-al.yaml            51 profile(s)
routing/servingstudio-dsv41-flash.yaml           8 profile(s)

all 10 set(s) validate
```

A set that fails prints `FAILED` on standard output and its problems on standard error.
What a set must contain is in [Coefficient set](../reference/coefficients.md).

Both commands exit with `0` when everything validates, `1` when something fails, and `2`
when the path is not a catalog or registry or the command was misused. The details are in
[Command line](../reference/cli.md#exit-codes).

## In CI

The `catalog` and `registry` jobs of `.github/workflows/ci.yml` run these commands on every
pull request that changes their directory, or the Go module:

```yaml
- run: go run ./validate/cmd/validate-catalog catalog
```

The validators and the data are in one repository, so CI checks the data against the
validators of the same commit. There is no version to pin.
