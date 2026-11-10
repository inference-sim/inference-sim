# Hardware

<p class="lede">Three schemas describe physical resources: a chip, an inter-node fabric, and
the storage classes KV cache can be offloaded to. Every figure in them is a datasheet
figure, or one computed from datasheet figures; <code>Provenance</code> says which. A
measured correction to any of them is a coefficient and belongs in the registry.</p>

The chip and fabric keys are in PascalCase, as the catalog's files write them. Each key
carries its unit. All three schemas accept the catalog's `_comment` and `_comment_*` keys
inside each chip, fabric and storage class, and discard them on load. Any other undeclared
key, including any other key that begins with an underscore, is an error.

## Chip

A file under `hardware/`, named by its file.

```yaml title="hardware/h100.yaml"
--8<-- "catalog/hardware/h100.yaml"
```

<!-- fields: hardware.Chip -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `Provenance` | string | yes | Where the figures come from. One of the [provenances](vocabularies.md#provenances). |
| `TFlopsPeak` | number | yes | Dense BF16 peak, TFLOP/s. |
| `TFlopsFP8` | number | no | Dense FP8 peak, TFLOP/s. Omit on parts without native FP8. If set, it must exceed `TFlopsPeak`. |
| `TFlopsNVFP4` | number | no | Dense NVFP4 (NVIDIA's 4-bit float) peak, TFLOP/s. Omit where the format is emulated rather than native. |
| `BwPeakTBs` | number | yes | HBM (on-package memory) bandwidth, TB/s. |
| `MemoryGiB` | number | yes | HBM capacity, GiB. |
| `IntraNodeBwGBps` | number | yes | On-node bandwidth per GPU, one direction, GB/s. NVLink where present. |
| `SMCount` | integer | yes | Streaming multiprocessors, positive. |
| `ComputeCapability` | string | no | CUDA compute capability, `major.minor`, such as `9.0` (H100, H200) or `10.0` (B200, GB200). A model graph that distinguishes architectures is resolved by the major version; one cannot be resolved for a chip that states none. |
| `GPUsPerNode` | integer | no | GPUs in one node. |
| `GPUsPerRack` | integer | no | GPUs in one rack, where a rack is a distinct link tier. A multiple of `GPUsPerNode`. |
| `IntraRackBwGBps` | number | no | Bandwidth between nodes inside a rack, GB/s. Needs `GPUsPerRack`. |

## Fabric

A file under `networks/`, named by its file.

```yaml title="networks/ib-400g.yaml, comments omitted"
Provenance: vendor_spec
InterNodeBwGBps: 50
```

<!-- fields: hardware.Fabric -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `Provenance` | string | yes | One of the [provenances](vocabularies.md#provenances). |
| `InterNodeBwGBps` | number | yes | Bandwidth per GPU, one direction, GB/s: the NIC's line rate in Gb/s divided by eight. Nominal, not measured. |
| `RDMA` | boolean | no | Whether transfers bypass the CPU. |

## Storage classes

One file, `devices/storage.yaml`, maps each storage class to its figures. The key is the
class's name, which scenarios and deployments use. A `name` key inside a class is accepted
and replaced by that key. Comment keys are allowed inside a class
but not at the top level of the file, where every key is a class.

```yaml title="devices/storage.yaml"
--8<-- "catalog/devices/storage.yaml"
```

<!-- fields: hardware.StorageDevice -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `read_bandwidth_mb_s` | number | yes | Read bandwidth, MB/s. |
| `write_bandwidth_mb_s` | number | yes | Write bandwidth, MB/s. Separate because flash writes are slower than reads. |
| `base_latency_us` | number | yes | Fixed cost per transfer, µs. It dominates small transfers. |

## Checks

Every number must be finite. Beyond that, a chip is invalid if

- `Provenance` is not recognized;
- `TFlopsPeak`, `BwPeakTBs`, `MemoryGiB` or `IntraNodeBwGBps` is not positive;
- `SMCount` is missing or not positive;
- `TFlopsFP8` or `TFlopsNVFP4` is negative;
- `TFlopsFP8` is positive but not above `TFlopsPeak`; no shipped part has an FP8 peak at or
  below its BF16 peak, so this is almost always a transcription slip;
- a peak is a sparse figure entered as dense: `TFlopsFP8` is more than 3.5 times `TFlopsPeak`,
  `TFlopsNVFP4` is more than 3.5 times `TFlopsFP8`, or `TFlopsNVFP4` does not exceed `TFlopsFP8`
  (design SUB-16: catalog peaks are dense figures, and their ratios between data types are the
  datasheet's; the shipped parts run 2.0 to 3.11);
- `GPUsPerRack` is not a multiple of `GPUsPerNode`, when both are set;
- `IntraRackBwGBps` is set without `GPUsPerRack`;
- `ComputeCapability` is set but is not `major.minor`.

A fabric is invalid if `Provenance` is not recognized or `InterNodeBwGBps` is not
positive. A storage class is invalid if any of its three figures is not positive.
`devices/storage.yaml` fails to load if it declares no class or a class with no value.

The catalog's convention that each `SMCount` cites a source in its `_comment_sm` prose is
not checked, here or anywhere else, because comments are discarded before validation.
[blis-schemas#32](https://github.com/inference-sim/blis-schemas/issues/32) records why it was left
out of the schema.

## Why three schemas

A chip's peak rates belong to the die. A fabric belongs to the cluster: the same chip sits
behind InfiniBand in one cluster and RoCE in another. Storage belongs to whoever provisioned
the node. The cost of a collective that crosses nodes depends on the ratio of on-node to
off-node bandwidth, which is a property of a chip paired with a fabric, not of either one.
The ratio is therefore a function of both, `hardware.IntraToInterRatio(chip, fabric)`, and
not a key in either file.

Source: [`spec/hardware`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/hardware).
