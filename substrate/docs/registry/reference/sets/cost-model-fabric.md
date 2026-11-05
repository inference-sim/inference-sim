---
hide:
  - navigation
  - toc
---

# cost-model-fabric

The effective / nominal bandwidth of each inter-node fabric class blis-catalog defines under
`networks/`. A fabric's catalog figure, `InterNodeBwGBps`, is its line rate divided by eight,
per GPU, per direction. A transfer over it achieves less. blis-latency-kernel multiplies the
nominal rate by `fabric_bandwidth_efficiency` wherever bytes leave a node: a P/D handoff, a
pipeline-stage boundary, and the cross-node share of a collective. It refuses a value outside
(0, 1].

Each entry is scoped by the fabric's catalog identity (`scope: {fabric: [ib-400g]}`), not by
hardware. A rack-wide NVLink domain (gb200-nvl72) belongs to the chip, so it is not corrected
here.

**Where the values come from.** None is a measurement this project made.

- `ib-400g` is `literature`: published nccl-tests all_reduce results on H100 hosts with eight
  ConnectX-7 400G InfiniBand NICs each ([NVIDIA/nccl-tests#309](https://github.com/NVIDIA/nccl-tests/issues/309)),
  as a fraction of the 400 GB/s ceiling an NVIDIA maintainer states there.
- `roce-200g` and `ethernet-100gbe` are `assumed`. No public multi-node measurement of either
  was found. Each rationale states the range the value could lie in and the one public datum
  that bounds it.

<!-- registry:set cost-model-fabric -->

**Regenerate:** `emit_fabric.py`; `--check` is the drift gate, and
`validator/test_fabric.py` requires exactly one entry in (0, 1] per catalog fabric.
