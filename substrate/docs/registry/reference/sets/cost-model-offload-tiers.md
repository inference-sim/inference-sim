---
hide:
  - navigation
  - toc
---

# cost-model-offload-tiers

What a transfer between the GPU and a storage tier costs, for each tier the catalog names in
`devices/storage.yaml`. A tier is a name and a description in the catalog; its three terms are
here, because none is a declared fact of a device (design SUB-18): each was a fit to one host or
a representative value the simulator's authors chose. They are the figures the catalog held,
to the digit, now with a method, a scope and their evidence. The kernel's `TierTime` prices a
transfer as the larger of the tier's `base_latency` and the bytes divided by the `read_rate` or
`write_rate` of the direction, with the rate held to the chip's `host_link_bandwidth` and divided
by the queue depth.

Each tier has three entries on every catalog chip, named by the tier:
`offload_tier_<tier>_read_rate` and `offload_tier_<tier>_write_rate` (bytes per microsecond,
which is MB/s) and `offload_tier_<tier>_base_latency` (microseconds per transfer). A scenario
whose deployment offloads to a tier lists this set; the kernel refuses to build when it does not.

**Where the values come from.** None is a measurement of the path the engine runs.

- `nvme_gen4`, `nvme_gen3`, `sata_ssd` and `s3` are `assumed`: representative classes seeded in
  inference-sim's defaults with no source cited, the same on every chip.
- `nvme_gen4_raid0_8x` is `assumed`: a DGX H100 node's local drives, whose model is not public,
  priced from NVIDIA's `> 50 GB/s` design target and one drive's published figures.
- `cpu_dram`'s two rates are `transferred`: fitted on one H100 host at the engine version that
  times `swap_blocks`, and moved to the engine priced, which batches the same copies, by a stated
  rule. Its floor is `assumed`: the small-copy floor of `swap_blocks` does not carry over to a
  batched call. On every chip but the H100 host the figures are the same, labeled as carried
  and not measured.

How to regenerate the file, what each figure rests on and what it does not show:
[Offload tiers](../../reproducing-coefficients.md#offload-tiers).

<!-- registry:set cost-model-offload-tiers -->
