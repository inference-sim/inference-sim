# Declared, learned, chosen

Every number behind a BLIS estimate is one of three kinds.

Declared
:   A figure a person read off a source and wrote down: a model's hidden size from its `config.json`, a chip's peak FP8 rate from its datasheet. It is either correct or a transcription error, and once correct it does not change. **Declared numbers live in the catalog.**

Learned
:   A number fitted to measurements, or assumed, for a cost model: the fraction of nominal bandwidth a fabric achieves, the efficiency of a matrix multiply at a given shape. New measurements can revise it, and it applies only within a stated scope, such as a set of chips or parallelism degrees. **Learned numbers live in the [registry](../../registry/index.md)** as *coefficients*, each recording how it was obtained and the scope it applies to.

Chosen
:   A setting picked for one run: which chip, how many GPUs, the tensor-parallel width, which fabric connects the nodes. **Chosen settings live in the run's scenario file**, which the user supplies; no BLIS repository owns scenarios.

The test for the catalog: *would a measurement revise this number?* If it would, the number belongs in the registry.

## One quantity, three kinds of number

A single physical quantity often has a number of each kind. Take NVLink bandwidth on an H100:

| Kind | Number | Where it lives |
| :--- | :--- | :--- |
| declared | 450 GB/s per GPU per direction: the datasheet's 900 GB/s bidirectional figure, halved | `hardware/h100.yaml`, `IntraNodeBwGBps` |
| learned | the fraction of that a collective achieves on a real cluster | a coefficient in the registry |
| chosen | eight GPUs in one node, tensor-parallel width 8 | the run's scenario |

## Nominal figures

The catalog's figures are *nominal*: what the source states, not what a benchmark achieves. A cost model that used 450 GB/s as the achieved rate would overestimate transfer speed. The correction from nominal to achieved is a learned number, applied where learned numbers are kept.

## Provenance

Each chip and fabric file states its provenance in one field, `Provenance`. The schema accepts two values:

`vendor_spec`
:   The figures are the vendor's, read directly or with a simple step: a unit conversion, half of a bidirectional figure, or the dense figure rather than the sparse one.

`derived`
:   The figures are computed from vendor figures by other arithmetic. They are still declared and still unmeasured.

The label applies to a whole file: a field the file's `FieldProvenance` does not name has the source the label gives it, and the label has to be a source for a field. `vendor_spec` is: its figures are the datasheet's. `derived` is not, because it says how a figure was obtained and not where it was read, so a file labeled `derived` states a source for every fact it records. A field's source is one of three words, which `FieldProvenance` states by the field's name:

`datasheet`
:   The vendor's datasheet states the figure.

`published`
:   The vendor or another public source publishes the figure outside a datasheet: a blog, a device-properties dump, a conference disclosure.

`measured:<device>`
:   The figure was read off one named device, such as the total a driver reports. The device is named in the file's comments too, and a test fails if they disagree.

A fitted or assumed value is none of these. It is a coefficient and belongs in the registry, and the schema refuses it here with that reason. The set of exceptions each chip records is pinned exactly by a test (`committedExceptions` in `contracts/spec/hardware`), so deleting or changing one fails it. Every chip and fabric file is labeled `vendor_spec` today. The [Chips](../reference/hardware.md#sources) and [Fabrics](../reference/networks.md#sources) references reproduce every comment.

## Where the catalog falls short of its rule

The rule is the aim, and some current entries do not yet meet it. As of this release:

- Every chip's peak rates and HBM bandwidth are vendor datasheet figures, but some memory capacities and SM counts are not, and each chip records which in `FieldProvenance`. The capacities of `b200`, `b300`, `gb200-nvl72`, `gb300`, `h100` and `h200` are what a named device's driver reports. The SM counts of `b300`, `gb200-nvl72` and `gb300` are what the device reports, taken from a published record of one named device. The SM counts of `a100-80`, `a100-sxm` and `b200` are published by NVIDIA outside a datasheet. The `MemoryGiB` of `a100-80`, `a100-sxm` and `l40s` is the datasheet's nominal capacity (80, 80 and 48) read as GiB; the CUDA driver reports less on the devices NVIDIA's published dumps cover (79.1 GiB for an A100-SXM4-80GB, 44.4 GiB for an L40S).
- A storage tier states no number. Its rates and latency were fitted or chosen by the simulator's authors, so they are coefficients, and they are in the registry.
- The workload shapes are presets chosen by the simulator's authors. They cite no trace or dataset. The replay workloads cite their corpus by repository, revision and digest.

Each of these files says so in its own comments. A pull request that replaces one of these with a cited declared figure, or moves a chosen number to the registry, is welcome.

## What the catalog leaves out

- Achieved efficiencies and overheads, such as model FLOPs utilization (MFU), effective bandwidth and fabric overhead, are learned numbers.
- Latency floors no vendor states, such as the fixed cost of moving a KV cache from a prefill server to a decode server, are modeling inputs. That one is an inference-sim setting (`--pd-transfer-base-latency`, 0.05 ms). It is not a registry coefficient: blis-registry PR #16, for [blis-registry#10](https://github.com/inference-sim/blis-registry/issues/10), added one, and commit 3c9879d removed it because nothing read it.
- Run choices, such as GPU count, parallelism and engine settings, are stated in the scenario file.
- A cluster that pairs a chip with a fabric is a run choice too. The catalog defines chips and fabrics separately, and the scenario pairs them.
