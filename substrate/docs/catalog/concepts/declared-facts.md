# Declared, learned, chosen

Every number behind a BLIS estimate is one of three kinds.

Declared
:   A figure a person read off a source and wrote down: a model's hidden size from its `config.json`, a chip's peak FP8 rate from its datasheet. It is either correct or a transcription error, and once correct it does not change. **Declared numbers live in the catalog.**

Learned
:   A number fitted to measurements, or assumed, for a cost model: the fraction of nominal bandwidth a fabric achieves, the efficiency of a matrix multiply at a given shape. New measurements can revise it, and it applies only within a stated scope, such as a set of chips or parallelism degrees. **Learned numbers live in [blis-registry](https://github.com/inference-sim/blis-registry)** as *coefficients*, each recording how it was obtained and the scope it applies to.

Chosen
:   A setting picked for one run: which chip, how many GPUs, the tensor-parallel width, which fabric connects the nodes. **Chosen settings live in the run's scenario file**, which the user supplies; no BLIS repository owns scenarios.

The test for the catalog: *would a measurement revise this number?* If it would, the number belongs in the registry.

## One quantity, three kinds of number

A single physical quantity often has a number of each kind. Take NVLink bandwidth on an H100:

| Kind | Number | Where it lives |
| :--- | :--- | :--- |
| declared | 450 GB/s per GPU per direction: the datasheet's 900 GB/s bidirectional figure, halved | `hardware/h100.yaml`, `IntraNodeBwGBps` |
| learned | the fraction of that a collective achieves on a real cluster | a coefficient in blis-registry |
| chosen | eight GPUs in one node, tensor-parallel width 8 | the run's scenario |

## Nominal figures

The catalog's figures are *nominal*: what the source states, not what a benchmark achieves. A cost model that used 450 GB/s as the achieved rate would overestimate transfer speed. The correction from nominal to achieved is a learned number, applied where learned numbers are kept.

## Provenance

Each chip and fabric file states its provenance in one field, `Provenance`. The schema accepts two values:

`vendor_spec`
:   The figures are the vendor's, read directly or with a simple step: a unit conversion, half of a bidirectional figure, or the dense figure rather than the sparse one.

`derived`
:   The figures are computed from vendor figures by other arithmetic. They are still declared and still unmeasured.

The label applies to a whole file. The schema has no label for a figure that is neither of these, such as a capacity or SM count that a device reports, so a file holding one keeps the label that fits the rest of its figures and says which fields are exceptions in its comments. Every chip and fabric file is labeled `vendor_spec` today. Read a file's comments, not only its label, before relying on a figure. The [Chips](../reference/hardware.md#sources) and [Fabrics](../reference/networks.md#sources) references reproduce every comment.

## Where the catalog falls short of its rule

The rule is the aim, and some current entries do not yet meet it. As of this release:

- Every chip's peak rates and HBM bandwidth are vendor datasheet figures, but some memory capacities and SM counts are not. The capacities of `b300` and `gb300` are what `nvidia-smi` reports, and `b200` and `h200` record their datasheets' GB figure as GiB. The SM count of `b300` comes from a device-properties listing NVIDIA publishes. The SM count of `gb200-nvl72` rests on NVIDIA's die-level disclosure for B200. The SM count of `gb300` comes from a third party's record of what the device reports. No datasheet states any of these, and each chip's comments say which source applies.
- The storage tiers are representative device classes chosen by the simulator's authors. They cite no source.
- The workload shapes are presets chosen by the simulator's authors. They cite no trace or dataset. The replay workloads cite their corpus by repository, revision and digest.

Each of these files says so in its own comments. A pull request that replaces one of these with a cited declared figure, or moves a chosen number to blis-registry, is welcome.

## What the catalog leaves out

- Achieved efficiencies and overheads, such as model FLOPs utilization (MFU), effective bandwidth and fabric overhead, are learned numbers.
- Latency floors no vendor states, such as the fixed cost of moving a KV cache from a prefill server to a decode server, are modeling inputs. That one is an inference-sim setting today (`--pd-transfer-base-latency`, 0.05 ms). blis-registry briefly held it: PR #16, for [blis-registry#10](https://github.com/inference-sim/blis-registry/issues/10), added it, and commit 3c9879d removed it because nothing read it there.
- Run choices, such as GPU count, parallelism and engine settings, are stated in the scenario file.
- A cluster that pairs a chip with a fabric is a run choice too. The catalog defines chips and fabrics separately, and the scenario pairs them.
