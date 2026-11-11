# Add a chip, fabric or storage tier

Chips, fabrics and storage tiers are separate files because they describe separate things. A chip's peak rates belong to the die and its package. A fabric belongs to the cluster the chip is installed in: the same chip can sit behind InfiniBand in one cluster and Ethernet in another. A storage tier belongs to whoever provisioned the node. A scenario pairs them; no catalog file does. The pairing matters: the cost of a collective that crosses nodes depends on the ratio of the chip's intra-node bandwidth to the fabric's.

Every figure is [nominal](../concepts/declared-facts.md#nominal-figures): what the source states, not what a benchmark achieves.

## A chip

Create `hardware/<name>.yaml`. The file name is the chip's identity; the file has no `name` key.

```yaml
Provenance: vendor_spec
TFlopsPeak: 989.5        # dense BF16, TFLOP/s
TFlopsFP8: 1979.0        # dense FP8, TFLOP/s; 0 on a chip without native FP8
BwPeakTBs: 3.35          # HBM bandwidth, TB/s
MemoryGiB: 80.0          # HBM capacity, GiB
IntraNodeBwGBps: 450     # NVLink, per GPU, per direction, GB/s
SMCount: 132             # streaming multiprocessors enabled on this chip
_comment_sm: "SMCount ... from the NVIDIA datasheet (https://...)."
```

[File formats](../reference/formats.md#chips) lists every key with its unit. These rules catch most transcription errors:

- Use dense rates. Datasheets often print the rate under 2:4 structured sparsity, which is twice the dense rate.
- Halve a bidirectional figure. NVIDIA quotes NVLink as the sum of both directions, and `IntraNodeBwGBps` is one direction: 900 GB/s bidirectional is 450.
- A non-zero `TFlopsFP8` must exceed `TFlopsPeak`. The validator rejects an FP8 peak at or below the BF16 peak as a probable slip.
- State `TFlopsNVFP4` only where the chip runs NVFP4 natively. Leave it out where the format is emulated, so no cost model reads a peak the chip reaches only by converting to a wider format.
- State `GPUsPerNode`, `GPUsPerRack` and `IntraRackBwGBps` only for a rack-scale NVLink domain that spans several nodes, as on GB200 NVL72 and GB300. When both counts are given, `GPUsPerRack` must be a multiple of `GPUsPerNode`, and `IntraRackBwGBps` requires `GPUsPerRack`. No other chip states them.

### Cite the source

Put the source of each figure, and any conversion you applied, in a comment key: `_comment`, or a key that begins `_comment_` such as `_comment_interconnect` or `_comment_sm`. The validator accepts those keys and rejects every other unknown key, including other names that begin with an underscore, so a fitted factor cannot be slipped in as `_mfu`. If a figure is not on a datasheet, for instance an SM count read from a measurement, say so in its comment; if it is assumed, it belongs in the [registry](../../registry/index.md) instead.

Every `SMCount` should cite a source a reviewer can follow, because enabled SM counts differ between products built on the same die. No check enforces this yet; [blis-schemas#32](https://github.com/inference-sim/blis-schemas/issues/32) tracks where it should live, and review enforces it until then. The [Chips](../reference/hardware.md#sources) reference reproduces every file's notes, which show the expected detail.

## A fabric

Create `networks/<name>.yaml`. A fabric is a reusable class, such as `ib-400g`, not one particular cluster.

```yaml
# InfiniBand NDR (400 Gb/s) — reusable inter-node fabric CLASS.
Provenance: vendor_spec
InterNodeBwGBps: 50      # per GPU, per direction, GB/s
_comment_interconnect: "One 400 Gb/s ConnectX-7 NIC per GPU: 400 Gb/s / 8 bits per byte = 50 GB/s ..."
```

The first `#` comment line names the fabric; the [Fabrics](../reference/networks.md) reference shows it as the description. `InterNodeBwGBps` is the line rate of the network cards serving one GPU, converted from gigabits to gigabytes per second. Set the optional `RDMA: true` when the fabric moves data between GPUs without passing it through the CPU (remote direct memory access).

A fabric has no latency field, because vendors state no single figure for it.

## A storage tier

A storage tier is a place a KV cache can be offloaded to, such as CPU memory or an NVMe drive. All tiers share one file, `devices/storage.yaml`, keyed by tier name. A tier is a name and a description, and the catalog states no number for it:

```yaml
nvme_gen4:
  _comment: "A PCIe Gen4 NVMe drive, as a class and not a product. ..."
```

What a transfer to or from a tier costs is a rate in each direction and a fixed cost per transfer. Those are fitted or assumed values, never a datasheet's figure for a named device, so they are coefficients and live in the registry, not here. Each tier has three, for every chip, in the registry set `cost-model-offload-tiers`:

| Coefficient | Unit | Meaning |
| :--- | :--- | :--- |
| `offload_tier_<tier>_read_rate` | bytes per µs (equal to MB/s) | rate from the tier to the GPU |
| `offload_tier_<tier>_write_rate` | bytes per µs | rate from the GPU to the tier; flash writes more slowly than it reads |
| `offload_tier_<tier>_base_latency` | µs per transfer | fixed cost of one transfer, which dominates small transfers |

To add a tier, add its name and description here, then add the three coefficients for every catalog chip by adding the tier to `registry/scripts/emit_offload_tiers.py` and running it (the [reproduction guide](../../registry/reproducing-coefficients.md#offload-tiers) says how, and what method each kind of figure takes). `TestEveryTierHasItsTermsOnEveryChip` fails until all of them are there. A tier that states `read_bandwidth_mb_s`, `write_bandwidth_mb_s` or `base_latency_us` is refused, with that reason. A tier may carry comment keys inside its own mapping, but not at the top level of the file.

## Then

Run the [checks](checks.md) and open a pull request that cites the source of every figure.
