#!/usr/bin/env python3
"""Write coefficients/cost-model-offload-tiers.yaml: what a transfer to or from a storage tier costs.

A storage tier named in the catalog (catalog/devices/storage.yaml) is an identity, a class a
deployment can offload KV cache to. What a transfer to it costs is a coefficient, not a
declared fact of a device (SUB-18, SUB-3), so each tier has three entries here, one per
term the kernel's TierTime reads:

    offload_tier_<tier>_read_rate     bytes per microsecond, tier to GPU (equal to MB/s)
    offload_tier_<tier>_write_rate    bytes per microsecond, GPU to tier
    offload_tier_<tier>_base_latency  microseconds per transfer, the floor that dominates
                                      small transfers

TierTime is max(floor, bytes / rate) per direction, with the rate capped at the chip's
host_link_bandwidth and divided by the queue depth.

WHERE THE FIGURES CAME FROM. Before this set the catalog carried them as the tiers' facts.
None is a datasheet or published figure of a named device, and the catalog's own header said
so: four are representative classes that inference-sim's defaults seeded without a source;
cpu_dram is a fit to InferenceX CollectiveX swap_blocks rows on one H100 host; and
nvme_gen4_raid0_8x is an assumed class for a DGX H100 node's local drives. Every value below is
the value the catalog held, to the digit, so no price moves; what changes is that each now says
how it was obtained and where it holds.

METHODS. The four classes and the NVMe aggregate are `assumed`: a stated judgement with its
reasoning. cpu_dram on the H100 host is not `measured`, because the engine priced is not the
engine measured (D5, VAL-8): the rows are vLLM v0.27.1's swap_blocks, and v0.31.0's offload
issues the same copies as one cuMemcpyBatchAsync over the same copy engines. The two directions'
large-payload rates carry over, so they are `transferred` on the stated rule; the per-call floor
does not (fit_host_link.py says the same of host_link_bandwidth), so the floor is `assumed`. On
every other chip cpu_dram has no measurement at all: the same figures are `transferred`
(rates) or `assumed` (floor) from the H100 host, as the catalog applied them before without
saying so, and TierTime holds a slower host link to its own measured host_link_bandwidth.

SCOPE. `hardware` lists every catalog chip, because the registry refuses an empty scope and a
chip left out would price its tier at nothing (test_offload_tiers.py checks every tier on every
chip). The tier rides in the entry name, as a dtype or a backend does elsewhere.

Usage:
    python scripts/emit_offload_tiers.py [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-offload-tiers.yaml"
CATALOG = HERE.parent.parent / "catalog"

# Every chip the catalog holds, by file stem. The writer reads them from the catalog so a chip
# added there fails --check here until its tiers are written.
CHIPS = sorted(p.stem for p in (CATALOG / "hardware").glob("*.yaml"))
H100 = ["h100"]
OTHERS = [c for c in CHIPS if c not in H100]

PR1594 = ("inference-sim PR #1594 (commit 3673d3655e), the kv_offload_devices section of defaults.yaml, "
          "split into blis-catalog at b20b7f1")

# The four representative classes. (read MB/s, write MB/s, base latency us) are the values the
# catalog held. The sentence after the name is what the class stands for.
CLASSES = {
    "nvme_gen4": (7.0e3, 5.0e3, 80.0, "a PCIe Gen4 NVMe drive"),
    "nvme_gen3": (3.5e3, 3.0e3, 100.0, "a PCIe Gen3 NVMe drive"),
    "sata_ssd": (5.5e2, 5.0e2, 150.0, "a SATA SSD"),
    "s3": (1.0e3, 1.0e3, 30000.0, "an object store reached over the network (first-byte latency 30 ms)"),
}

CLASS_RATIONALE = (
    "ASSUMED CLASS, not a datasheet figure: {what}. The figure is the one inference-sim's defaults seeded "
    "with the comment 'representative device classes seeded for the multi-tier offload holes; operators "
    "override per run for their hardware'. Neither that comment, that pull request nor blis-catalog's history "
    "cites a datasheet, benchmark or other source for it, and a class is not a product, so no single datasheet "
    "could state it. It is the same on every chip because nothing about the class depends on the GPU. The "
    "caveat is its size: a device's sustained rate under a deep queue, and a store's latency under load, "
    "can sit well below the rate of a drive's datasheet maximum, and TierTime divides the rate by the queue "
    "depth rather than modelling saturation. To replace it, name a product and measure it: sequential "
    "{direction} at queue depths 1 to 32 and 4 KiB to 1 MiB, then fit floor + bytes / rate per direction as "
    "fit_host_link.py does for host memory."
)

TERMS = {
    "read_rate": ("bytes_per_us", "{direction}", "read"),
    "write_rate": ("bytes_per_us", "{direction}", "write"),
    "base_latency": ("us_per_transfer", "latency", None),
}


def scope(chips: list[str]) -> str:
    return "{hardware: [" + ", ".join(chips) + "]}"


def num(v: float) -> str:
    return repr(float(v))


def entry(name, value, units, method, chips, sources, rationale, *, transferred=None, validated=None):
    lines = [
        f"  - {name}:",
        f"      value: {num(value)}",
        f"      units: {units}",
        f"      method: {method}",
        "      fitted: false",
        f"      scope: {scope(chips)}",
    ]
    if transferred:
        frm, rule = transferred
        lines += [f'      transferred_from: "{frm}"', f"      transfer_rule: >", f"        {rule}"]
    if validated:
        lines += ["      validated:"] + [f'        - "{v}"' for v in validated]
    lines += ["      sources:"] + [f"        - {s}" for s in sources]
    lines += ["      rationale: >", f"        {rationale}"]
    return lines


def src(kind, cite, role):
    return '{kind: %s, cite: "%s", role: %s}' % (kind, cite.replace('"', "'"), role)


# --- cpu_dram ------------------------------------------------------------------------------

CX = ("InferenceX CollectiveX swap_blocks, sku h100-dgxc, image vllm/vllm-openai:v0.27.1, runs 35158674963 and "
      "36334394602 (public InferenceX Data API /api/v1/collectivex/runs/{id}); pinned host memory, contiguous "
      "layout")

CPU_FIT = (
    "FIT. Contiguous layout; rate = median payload rate at payloads >= 64 MiB over all block sizes (26 points "
    "per direction); floor = median p50 latency at payloads <= 16 KiB. Host-to-device 27,058 MB/s and "
    "device-to-host 46,089 MB/s; the two directions' floors are 19.7 and 18.9 us. HELD OUT (random layout, "
    "payload >= 1 MiB): the median absolute error of p50 latency is 0.51 host-to-device and 0.14 "
    "device-to-host, against 0.55 and 0.98 for the 2.0e4 MB/s / 1 us placeholder it replaced. The data show a "
    "per-descriptor cost this form cannot state: at 64 KiB blocks the rate stays near 9-10 GB/s at any count, "
    "at 1 MiB blocks it reaches 40-44 GB/s. These are the same rows fit_host_link.py fits for "
    "host_link_bandwidth, whose h100 value is the larger of the two rates, so tier and cap are one fit and not "
    "two. The run bodies the fit read are no longer served by the API, so the fit cannot be re-derived from "
    "public data today (blis-registry#57)."
)

CPU_TRANSFER_RULE = (
    "The rows time vLLM v0.27.1's swap_blocks. vLLM v0.27.1 to v0.31.0 changes only a descriptor-chunking loop "
    "in swap_blocks_batch (csrc/libtorch_stable/cache_kernels.cu) that is a no-op on CUDA, and v0.31.0's "
    "offload issues its copies as one cuMemcpyBatchAsync over the same copy engines, so the large-payload rate "
    "is used unchanged as the rate of the path priced. The small-copy floor is not carried over (see "
    "offload_tier_cpu_dram_base_latency)."
)

CPU_OTHER_RULE = (
    "Used unchanged: the rate measured on the H100 host is the only one measured for this tier, and before "
    "this entry the catalog applied it to every host without saying so. TierTime then holds the rate to this "
    "chip's own measured host_link_bandwidth, so a host whose link is slower than the H100's is priced at its "
    "own link. No H200, Blackwell or Grace-Blackwell host has been timed for this tier."
)


def cpu_dram() -> list[list[str]]:
    out = []
    held_out = ["held-out random-layout p50 latency, payload >= 1 MiB: median absolute error 0.51 host-to-device "
                "and 0.14 device-to-host, against 0.55 and 0.98 for the 2.0e4 MB/s / 1 us placeholder"]
    cx = src("model", CX, "primary")
    cx_supporting = src("model", CX, "supporting")
    for term, value, direction, frm in (("read_rate", 27058.0, "host-to-device", "read"),
                                        ("write_rate", 46089.0, "device-to-host", "write")):
        name = f"offload_tier_cpu_dram_{term}"
        out.append(entry(
            name, value, "bytes_per_us", "transferred", H100, [cx], (
                f"The {direction} payload rate of pinned host memory across an H100 SXM host link, used as the "
                f"{frm} rate of the cpu_dram tier. Formerly the catalog's cpu_dram {frm}_bandwidth_mb_s, which "
                f"was a fit to H100 measurements and not a device fact. " + CPU_FIT),
            transferred=("hardware=h100", CPU_TRANSFER_RULE), validated=held_out))
        out.append(entry(
            name, value, "bytes_per_us", "transferred", OTHERS, [cx], (
                f"The H100 host's {direction} payload rate (see the h100 entry), carried to a host that has not "
                f"been measured. Before this entry the catalog applied the figure to every host as a generic fact. "
                + CPU_FIT),
            transferred=("hardware=h100", CPU_OTHER_RULE)))
    name = "offload_tier_cpu_dram_base_latency"
    floor_why = (
        "19.3 us is the mean of the two directions' small-copy floors (19.7 and 18.9 us, median p50 latency at "
        "payloads <= 16 KiB, contiguous layout) of vLLM v0.27.1's swap_blocks on an H100 host, used as the fixed "
        "cost of one transfer of the path v0.31.0 runs, swap_blocks_batch. The per-call floor does not carry over "
        "to that call (a batch of descriptors submitted as one cuMemcpyBatchAsync is not one copy per call), so "
        "the figure is a judgement that the two floors are alike and not a measurement of the path priced. It "
        "matters for small transfers only: 64 KiB is 1.4 to 2.4 us at the rates and 19.3 us at the floor. To "
        "replace it, time swap_blocks_batch on a pinned host buffer over 1 to 16 KiB payloads with the "
        "CollectiveX harness at the engine version priced. The measurement it rests on: " + CX + "."
    )
    out.append(entry(name, 19.3, "us_per_transfer", "assumed", H100, [cx_supporting],
                     "ASSUMED, from a measurement of another call. Formerly the catalog's cpu_dram "
                     "base_latency_us, a fit and not a device fact. " + floor_why))
    out.append(entry(name, 19.3, "us_per_transfer", "assumed", OTHERS, [cx_supporting],
                     "ASSUMED, from a measurement of another call on another host: the H100 host's figure (see the "
                     "h100 entry), carried to a host that has not been measured. Before this entry the catalog "
                     "applied it to every host as a generic fact. " + floor_why))
    return out


# --- nvme_gen4_raid0_8x ---------------------------------------------------------------------

RAID_SOURCES = [
    src("vendor_doc", "NVIDIA DGX H100 user guide and datasheet: 8 x 3.84 TB NVMe U.2 in RAID 0 at /raid (PCIe "
        "Gen4); the drive model is not public", "supporting"),
    src("vendor_doc", "NVIDIA DGX SuperPOD H100 reference architecture: '> 50 GB/s' from internal NVMe (a design "
        "target, not a measurement)", "supporting"),
    src("publication", "Haas and Leis, VLDB 2023 (high-performance I/O for NVMe storage engines): ~6.5 GB/s "
        "measured per Samsung PM1733 under SPDK", "supporting"),
    src("datasheet", "Samsung PM1733 datasheet: 7.0 GB/s sequential read maximum, 3.8 GB/s sequential write; 128 KiB "
        "QD1 read 220 us and write 80 us", "upper_bound"),
    src("publication", "Samsung PM1743 white paper: ~1.2-1.5 GB/s per Gen4 drive at queue depth 1 on md RAID 0 "
        "(128 KiB splits)", "supporting"),
    src("publication", "FxMark, USENIX ATC 2016: 25-50 us of ext4 create/rename metadata per file", "supporting"),
    src("publication", "XRP, OSDI 2022: ~3 us of O_DIRECT software path", "supporting"),
]

RAID_COMMON = (
    "ASSUMED CLASS for a DGX/HGX H100 node's local NVMe data volume: the drive model is not public. NVIDIA's "
    "DGX H100 user guide and datasheet state 8 x 3.84 TB NVMe U.2 in RAID 0 at /raid (PCIe Gen4); NVIDIA's DGX "
    "SuperPOD H100 reference architecture states '> 50 GB/s' from internal NVMe, which is a design target. "
    "These are DEVICE aggregates; TierTime divides them among the transfers in service, so one of vLLM's "
    "16+16 fs threads alone sees the whole rate, which a single O_DIRECT stream does not reach on md RAID 0 "
    "(128 KiB splits; ~1.2-1.5 GB/s per Gen4 drive at QD1, Samsung PM1743 white paper): the figure is optimistic "
    "at low concurrency. The sources were saved with their provenance under the r2b work's data/external/nvme-* "
    "and fsop-* (not redistributed), and no URL was recorded with them."
)

RAID_TERMS = {
    "read_rate": (5.0e4, "bytes_per_us",
                  "Read 50,000 MB/s: the SuperPOD design target, and 8 x ~6.5 GB/s measured per PM1733 under SPDK "
                  "(Haas and Leis, VLDB 2023), which sits below 8 x the 7.0 GB/s datasheet maximum."),
    "write_rate": (3.0e4, "bytes_per_us",
                   "Write 30,000 MB/s: 8 x 3.8 GB/s PM1733 datasheet sequential write, rounded down."),
    "base_latency": (100.0, "us_per_transfer",
                     "Base latency 100 us per file: a 128 KiB QD1 read is 220 us and a write 80 us on PM1733 "
                     "(datasheet), plus 25-50 us of ext4 create/rename metadata on a store (FxMark, ATC 2016) and "
                     "~3 us of O_DIRECT software path (XRP, OSDI 2022); the range assumed is 50-250 us."),
}

RAID_REPLACE = (
    " To replace it, run fio on the node's /raid volume at the concurrency vLLM's offload connector uses and fit "
    "floor + bytes / rate per direction.")


def raid() -> list[list[str]]:
    out = []
    for term, (value, units, why) in RAID_TERMS.items():
        name = f"offload_tier_nvme_gen4_raid0_8x_{term}"
        out.append(entry(name, value, units, "assumed", H100, RAID_SOURCES, why + " " + RAID_COMMON + RAID_REPLACE))
        out.append(entry(name, value, units, "assumed", OTHERS, RAID_SOURCES,
                         "Carried unchanged from the DGX H100 assumption (see the h100 entry) to a chip whose own "
                         "node-local NVMe volume is not described here; before this entry the catalog applied it to "
                         "every chip as a generic fact. " + why + " " + RAID_COMMON + RAID_REPLACE))
    return out


def classes() -> list[list[str]]:
    out = []
    for tier, (read, write, lat, what) in CLASSES.items():
        for term, value in (("read_rate", read), ("write_rate", write), ("base_latency", lat)):
            units, _, _ = TERMS[term]
            direction = {"read_rate": "reads", "write_rate": "writes", "base_latency": "small transfers"}[term]
            out.append(entry(
                f"offload_tier_{tier}_{term}", value, units, "assumed", CHIPS,
                [src("discussion", PR1594 + ": a representative class seeded with no source cited", "primary")],
                CLASS_RATIONALE.format(what=what, direction=direction)))
    return out


def all_entries() -> list[list[str]]:
    return classes() + cpu_dram() + raid()


HEADER = """\
# Offload tiers: what a transfer to or from a storage tier costs.
#
# GENERATED by scripts/emit_offload_tiers.py. Re-run it and commit the diff rather than editing
# values here; `emit_offload_tiers.py --check` is the drift gate.
#
# A storage tier (catalog/devices/storage.yaml) is a name and a description. Its read rate,
# write rate and fixed cost per transfer are three entries here, offload_tier_<tier>_read_rate,
# _write_rate and _base_latency, which the kernel's TierTime reads (price of one transfer =
# max(base_latency, bytes / rate) per direction, the rate held to the chip's
# host_link_bandwidth and divided by the queue depth). They are coefficients because none is a
# declared fact of a device: every figure was a fit to one host or a representative value the
# simulator's authors chose (design SUB-18, SUB-3). They are the figures the catalog held, to
# the digit, now with a method, a scope and their evidence.
#
# A scenario whose deployment offloads to a tier lists this set; the kernel refuses to build
# when it does not, naming the entries. The set is separate from cost-model-primitives because
# that set is single-sourced to NVIDIA AISimulate and these are not.
#
# Methods: the four classes and the NVMe aggregate are `assumed`. cpu_dram's rates are
# `transferred` from the H100 host they were fitted on, and its floor `assumed`: the rows time
# vLLM v0.27.1's swap_blocks and the engine priced runs swap_blocks_batch (design D5, VAL-8).
# Every other chip carries the H100 figures under the same methods until its own host is timed.
kind: CoefficientSet
name: cost-model-offload-tiers
coefficients:
"""


def render() -> str:
    return HEADER + "\n".join(line for e in all_entries() for line in e) + "\n"


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if the file would change; write nothing")
    args = ap.parse_args(argv[1:])
    after = render()
    before = SET_PATH.read_text(encoding="utf-8") if SET_PATH.exists() else ""
    if args.check:
        if after != before:
            print(f"{SET_PATH.name} would change", file=sys.stderr)
            return 1
        print(f"{SET_PATH.name} already matches")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"{SET_PATH.name} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
