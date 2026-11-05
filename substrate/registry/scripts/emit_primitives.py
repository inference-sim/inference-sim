#!/usr/bin/env python3
"""Generate the per-SKU primitive coefficient set from NVIDIA AISimulate.

Two kinds of number, both from NVIDIA and nothing else:

  * FITTED, from the measurement sweeps — the GEMM efficiency envelope per dtype
    and the MoE routing-imbalance order statistics.
  * DECLARED, from AISimulate's own system descriptors (systems/<sku>.yaml) — the
    HBM derate, the PCIe host-link bandwidth, the NCCL communicator reservation per
    rank count, and the engine workspace allowance.

Two descriptor fields are deliberately NOT emitted: `mem_empirical_constant_latency`
and `p2p_latency`. Nothing in BLIS reads them, and a reader would double-count -- the
kernel's per-kernel attention floors and per-operation collective floors are measured
and already carry the fixed cost these two constants stand in for in NVIDIA's own
simulator (inference-sim/blis-registry#36).

A100-80 IS AN ALIAS OF A100-SXM (methodology section 5), so the `a100_pcie` descriptor
is not emitted and the `a100_sxm` one is scoped to both names. Emitting both put two
entries of every descriptor name in scope for a100-80, and the resolver kept whichever
came later -- equal values today, a silent divergence after a refit that touched one.

The declared group matters as much as the fitted one. Those are numbers NVIDIA
ships with its own simulator, annotated there as corrections based on observation,
and three of them replace figures this model previously carried as assumptions: the
communicator buffer size, the runtime workspace allowance, and the host link rate.

Usage:
    python scripts/emit_primitives.py <aisimulate>/systems <catalog-root>   # render, stdout
    python scripts/emit_primitives.py --check   # the entries it owns, against the committed set

Only the descriptor entries and A100's MoE median are committed from this script; the
rendered GEMM path is superseded by relane_gemm_envelope.py, so the whole-file output is a
reference, never a file to commit. `--check` reads $AISIMULATE_DATA's parent `systems/`.

The first argument is the `systems` directory, which holds both the per-SKU
descriptors (`<sku>.yaml`) and the measurement collections (`data/<sku>/...`).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import yaml

_spec = importlib.util.spec_from_file_location(
    "fit_gemm_envelope", Path(__file__).parent / "fit_gemm_envelope.py"
)
fg = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(fg)

HEADER = '''# Cost-model primitives — every entry from NVIDIA AISimulate.
#
# WRITTEN by per-family writers, each with --check: relane_gemm_envelope.py (the dense
# gemm_*_{bf16,fp8,fp8_block,nvfp4} ramps), relane_moe_imbalance.py
# (moe_routing_imbalance_median), fit_marlin_grouped_ramp.py (gemm_*_w4a16_mxfp4_grouped),
# fit_deepseek_v41_h200.py (the assumed gemm_*_w8a16_mxfp8 and host_gather_rate_uva), and
# emit_primitives.py (the descriptor entries and A100's MoE median).
# docs/reproducing-coefficients.md maps every family to its owner.
# Re-run the owner and commit the diff rather than editing values here.
#
# Two kinds of number, both NVIDIA-sourced:
#
#   FITTED, from the measurement sweeps. The GEMM efficiency envelope
#   eff(M) = eps_max * M / (M + M_half), fitted per dtype against the per-M
#   envelope of achieved fraction of that dtype's dense peak; and the MoE
#   routing-imbalance multiplier, as order statistics of paired measurements that
#   differ in nothing but the routing distribution.
#
#   DECLARED, from AISimulate's own system descriptors (systems/<sku>.yaml). These
#   are numbers NVIDIA ships with its simulator, annotated there as corrections
#   based on observation. They are recorded as `vendor_spec` rather than
#   `measured`: NVIDIA observed them, this project did not, and the sweep behind
#   them is not published.
#
# The envelope is per SKU as well as per dtype, and the H100/H200 pair shows why
# it has to be. Those two share the GH100 die and therefore an identical compute
# peak, and their bf16 asymptotes agree to within 0.013 as that predicts. Their knees
# do not: bf16 M_half is 170 on H100 and 110 on H200, a factor of 1.55. The knee is set
# by how fast operands reach the tensor cores, so it tracks memory bandwidth
# (3.35 against 4.8 TB/s) rather than compute. A model that carries one part's
# ramp to another on shared-die reasoning gets the asymptote right and the knee
# wrong, and the knee is what governs every decode-sized batch.
kind: CoefficientSet
name: cost-model-primitives

coefficients:
'''

DTYPE_SUFFIX = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "fp8": "fp8",
    "fp8_block": "fp8_block",
    "nvfp4": "nvfp4",
}

# Which measurement collection each SKU's coefficients come from, pinned by name.
#
# Pinned rather than selected by a rule, for two reasons the data forces. First,
# "newest version" is wrong: h200's moe has trtllm 1.3.0rc10, rc20 and rc23 with
# 24 786, 42 174 and 3 564 paired shapes respectively, and their medians read
# 1.103, 1.056 and 1.002. The newest is the smallest and least representative.
# Second, "largest collection" is also wrong: it picks vLLM for GEMM and TRT-LLM
# for MoE on the same part, so a cross-SKU comparison would confound hardware with
# runtime.
#
# TRT-LLM 1.3.0rc20 carries both families for h100, h200, l40s and gb200, which
# makes those four directly comparable under one runtime. A100 predates it and has
# no equivalent, so it is pinned to what exists and the difference is stated in
# every A100 entry's rationale rather than hidden. a100_pcie has no measurement
# collection at all and therefore gets no fitted coefficients.
COLLECTIONS = {
    "h100_sxm": {"gemm": "trtllm/1.3.0rc20", "moe": "trtllm/1.3.0rc20"},
    "h200_sxm": {"gemm": "trtllm/1.3.0rc20", "moe": "trtllm/1.3.0rc20"},
    # Blackwell pins the same trtllm collection as Hopper, so an envelope difference
    # between the two is the silicon rather than a framework version.
    "b200_sxm": {"gemm": "trtllm/1.3.0rc20", "moe": "trtllm/1.3.0rc20"},
    "b300_sxm": {"gemm": "trtllm/1.3.0rc20", "moe": "trtllm/1.3.0rc20"},
    "l40s": {"gemm": "trtllm/1.3.0rc20", "moe": "trtllm/1.3.0rc20"},
    "gb200": {"gemm": "trtllm/1.3.0rc20", "moe": "trtllm/1.3.0rc20"},
    # A100 predates the rc20 sweeps. sglang/0.5.10 is the most complete collection
    # that exists for this part, and it is a different runtime from the other four,
    # so its numbers are not strictly comparable with theirs.
    "a100_sxm": {"gemm": "sglang/0.5.10", "moe": "sglang/0.5.10"},
}

# The runtime every cross-SKU comparison in this set is made under. An entry from
# a different collection says so in its rationale.
REFERENCE_COLLECTION = "trtllm/1.3.0rc20"

# AISimulate SKU directory to its system descriptor and catalog chip.
SKUS = fg.SKU_TO_CHIP

# a100-80 is served by the a100_sxm descriptor (see the module docstring), so the
# a100_pcie descriptor is skipped and the SXM entries are scoped to both names.
SKIPPED_SKUS = {"a100_pcie"}
SCOPE = {"a100-sxm": "a100-80, a100-sxm"}


def entry(
    name: str,
    value,
    units: str,
    method: str,
    chip: str,
    cite: str,
    rationale: str,
    fitted: bool = False,
    ci95: tuple[float, float] | None = None,
) -> str:
    out = (
        f"  - {name}:\n"
        f"      value: {value}\n"
        f"      units: {units}\n"
        f"      method: {method}\n"
        f"      fitted: {'true' if fitted else 'false'}\n"
        f"      scope: {{hardware: [{SCOPE.get(chip, chip)}]}}\n"
    )
    if ci95:
        out += f"      ci95: [{ci95[0]}, {ci95[1]}]\n"
    out += (
        f"      sources:\n"
        f"        - {{kind: model, cite: \"{cite}\", role: primary}}\n"
        f"      rationale: >\n"
        f"        {rationale}\n"
    )
    return out


def gemm_entries(root: Path, catalog: Path, sku: str, chip: str) -> list[str]:
    pinned = COLLECTIONS.get(sku, {}).get("gemm")
    if pinned is None:
        return []
    version = root / "data" / sku / "gemm" / pinned
    if not version.is_dir():
        raise FileNotFoundError(
            f"{sku}: pinned GEMM collection {pinned} is absent; update COLLECTIONS "
            f"rather than falling back to another one silently"
        )
    rows = fg.read_rows(version, "gemm_perf")
    if rows is None:
        return []
    peaks = fg.peaks_from_catalog(catalog, chip)
    device = rows[0]["device"]
    measured = {r["gemm_dtype"] for r in rows}
    out = []
    for dtype, suffix in DTYPE_SUFFIX.items():
        if dtype not in measured or dtype not in peaks:
            continue
        env = fg.envelope(rows, dtype, peaks[dtype])
        if not env:
            continue
        eps, m_half, rms = fg.fit_ramp(env)
        n = sum(1 for r in rows if r["gemm_dtype"] == dtype)
        cite = (
            f"NVIDIA AISimulate systems/data/{sku}/gemm/{pinned}/gemm_perf.parquet "
            f"({device})"
        )
        offref = (
            ""
            if pinned == REFERENCE_COLLECTION
            else f" Measured under {pinned} rather than the "
            f"{REFERENCE_COLLECTION} the other parts in this set use, because no "
            f"{REFERENCE_COLLECTION} collection exists for it; the value is "
            f"therefore not strictly comparable with theirs."
        )
        ceiling = ""
        if eps >= 1.0:
            ceiling = (
                " The asymptote is pinned at peak: the two-parameter ramp "
                "extrapolates past the highest observed fraction "
                f"({max(env.values()):.3f}), so this part's envelope is better "
                "read as approaching peak than as having a fitted ceiling."
            )
        out.append(
            entry(
                f"gemm_eps_max_{suffix}",
                round(eps, 3),
                "dimensionless",
                "measured",
                chip,
                cite,
                f"Least-squares fit of eff = eps_max * M / (M + M_half) to the "
                f"per-M efficiency envelope of {n} {dtype} rows against this "
                f"part's dense {dtype} peak, over {len(env)} distinct M. RMS "
                f"residual {rms:.3f}, which is the interval. Highest single "
                f"observation {max(env.values()):.3f}.{ceiling}{offref}",
                fitted=True,
                ci95=(round(max(0.0, eps - rms), 3), round(min(1.0, eps + rms), 3)),
            )
        )
        out.append(
            entry(
                f"gemm_m_half_{suffix}",
                m_half,
                "tokens",
                "measured",
                chip,
                cite,
                f"The half-max row count from the same fit as "
                f"gemm_eps_max_{suffix}.",
                fitted=True,
            )
        )
    return out


def moe_entries(root: Path, sku: str, chip: str) -> list[str]:
    pinned = COLLECTIONS.get(sku, {}).get("moe")
    if pinned is None:
        return []
    version = root / "data" / sku / "moe" / pinned
    if not version.is_dir():
        raise FileNotFoundError(
            f"{sku}: pinned MoE collection {pinned} is absent; update COLLECTIONS "
            f"rather than falling back to another one silently"
        )
    rows = fg.read_rows(version, "moe_perf")
    if rows is None:
        return []

    def shape(r: dict) -> tuple:
        return tuple(
            r[k] for k in (
                "moe_dtype", "num_tokens", "hidden_size", "inter_size", "topk",
                "num_experts", "moe_tp_size", "moe_ep_size", "kernel_source",
            )
        )

    balanced = {
        shape(r): r["latency"] for r in rows if r["distribution"] == "balanced"
    }
    ratios = sorted(
        r["latency"] / balanced[shape(r)]
        for r in rows
        if r["distribution"] == "power_law_1.01"
        and shape(r) in balanced
        and balanced[shape(r)] > 0
    )
    if len(ratios) < 100:
        return []
    n = len(ratios)
    median = ratios[n // 2]
    p10, p90 = ratios[int(0.10 * n)], ratios[int(0.90 * n)]
    cite = (
        f"NVIDIA AISimulate systems/data/{sku}/moe/{pinned}/moe_perf.parquet, "
        f"distribution power_law_1.01 against balanced ({rows[0]['device']})"
    )
    offref = (
        ""
        if pinned == REFERENCE_COLLECTION
        else f" Measured under {pinned} rather than the {REFERENCE_COLLECTION} the "
        f"other parts in this set use, so not strictly comparable with theirs."
    )
    return [
        entry(
            "moe_routing_imbalance_median", round(median, 3), "dimensionless",
            "measured", chip, cite,
            f"Ratio of measured latency under a power-law routing distribution to "
            f"the same shape under a balanced one, over {n} shapes differing in "
            f"nothing but the distribution. Median {median:.3f}; the 10th and 90th "
            f"percentiles are {p10:.3f} and {p90:.3f} and the maximum observed is "
            f"{ratios[-1]:.1f} -- a spread of the measured ratios, not a confidence "
            f"interval on the median, which is why it is stated here rather than as "
            f"ci95. The spread is the finding: at the median a skewed "
            f"distribution costs a few percent and at the 90th percentile tens of "
            f"percent, so a prediction sensitive to this term should carry the "
            f"band rather than a point. Not marked fitted: an order statistic of "
            f"measured ratios, not a curve fit.{offref}",
        ),
    ]


def descriptor_entries(root: Path, sku: str, chip: str) -> list[str]:
    path = root / f"{sku}.yaml"
    if not path.is_file():
        return []
    d = yaml.safe_load(path.read_text())
    gpu, node, misc = d.get("gpu", {}), d.get("node", {}), d.get("misc", {})
    cite = f"NVIDIA AISimulate systems/{sku}.yaml"
    out = []
    if (derate := gpu.get("mem_bw_empirical_scaling_factor")) is not None:
        out.append(entry(
            "hbm_derate", derate, "dimensionless", "vendor_spec", chip, cite,
            "The fraction of datasheet HBM bandwidth a real access stream "
            "achieves. NVIDIA ships this in its own system descriptor, annotated "
            "there as a correction based on observation. Applied to the catalog's "
            "BwPeakTBs. vendor_spec rather than measured: NVIDIA observed it, this "
            "project did not, and the sweep behind it is not published.",
        ))
    # Parts with a CollectiveX swap_blocks measurement carry a fitted host link
    # (fit_host_link.py); the descriptor's nominal rate stays only where none is measured.
    if (pcie := node.get("pcie_bw")) is not None and chip not in MEASURED_HOST_LINK:
        out.append(entry(
            "host_link_bandwidth", round(pcie / 1e6, 1), "bytes_per_us",
            "vendor_spec", chip, cite,
            f"{pcie / 1e9:.0f} GB/s single-direction over the CPU-to-GPU link, "
            f"which is what an offload tier's transfer crosses in addition to host "
            f"memory. NVIDIA's descriptor states this per part, and the value "
            f"differs by platform: a Grace-Blackwell part connects over NVLink-C2C "
            f"rather than PCIe, so a tier that is a liability on one host can be "
            f"affordable on another. Nominal link rate, not a measured payload "
            f"throughput; a real transfer reaches a fraction of it.",
        ))
    nccl_mem = misc.get("nccl_mem") or {}
    for ranks in sorted(k for k in nccl_mem if isinstance(k, int)):
        if not nccl_mem[ranks]:
            continue
        out.append(entry(
            f"nccl_communicator_bytes_{ranks}rank", nccl_mem[ranks], "bytes_per_rank",
            "vendor_spec", chip, cite,
            f"Bytes a rank reserves for NCCL communicator buffers at {ranks} "
            f"ranks, from NVIDIA's descriptor. It enters the memory breakdown's "
            f"CommBuffers term, and it is a real per-rank-count figure rather than "
            f"the order-of-magnitude constant this model previously carried. It is "
            f"the WHOLE of a rank's collective-buffer reservation in NVIDIA's "
            f"composition: engine_workspace_bytes is a separate allowance that adds "
            f"to it, not part of it (AISimulate sdk/backends/base_backend.py sums "
            f"nccl_mem and other_mem; cost-model-memory.yaml states the full sum). "
            f"The descriptor declares 0 at one rank, which is why no 1-rank entry "
            f"exists, and AISimulate takes the 8-rank figure for any wider group, "
            f"clamping at min(tp, 8).",
        ))
    if (other := misc.get("other_mem")) is not None:
        out.append(entry(
            "engine_workspace_bytes", other, "bytes_per_rank", "vendor_spec", chip,
            cite,
            f"{other / 1e9:.1f} GB of non-weight, non-KV occupancy NVIDIA's "
            f"descriptor reserves, which AISimulate's memory model labels as CUDA "
            f"context, cuBLAS and the like (\"cuda, cublas, etc.\", "
            f"sdk/backends/base_backend.py) and the descriptor annotates as a safety "
            f"reserve that also covers part of the inaccuracy of the rest of its "
            f"memory calculation. It does NOT include CUDA-graph capture: "
            f"AISimulate annotates other_mem as "
            f"excluding the graph reservation (support/config_profile.py) and takes "
            f"that reservation as a separate argument (capacity.py, "
            f"cuda_graph_reserved_bytes), so cudagraph_capture_bytes_<mode> in "
            f"cost-model-memory.yaml adds to this rather than sitting inside it. Nor "
            f"is it part of the collective reservation: NVIDIA's composition sums it "
            f"with nccl_communicator_bytes_<N>rank. It is a lumped allowance rather "
            f"than a decomposition, so a memory breakdown reports it as one term and "
            f"does not claim to split it.",
        ))
    return out


# The families this script OWNS in the committed set: the descriptor entries for every
# part (gb300's descriptor is read like the others; its sweeps are owned elsewhere), and
# A100's MoE median, the one imbalance entry relane_moe_imbalance.py cannot derive.
MEASURED_HOST_LINK = {"h100", "h200", "b200", "b300", "gb200-nvl72", "gb300"}
DESCRIPTOR_NAMES = {"hbm_derate", "host_link_bandwidth", "nccl_communicator_bytes_2rank",
                    "nccl_communicator_bytes_4rank", "nccl_communicator_bytes_8rank",
                    "engine_workspace_bytes"}
DESCRIPTOR_PARTS = sorted({**{s: c for s, c in SKUS.items() if s not in SKIPPED_SKUS},
                           "gb300": "gb300"}.items())


def check(root: Path, committed_path: Path) -> int:
    """Compare every entry this script owns against the committed set, field by field.

    The whole-file output is not committable (the GEMM path is superseded), so the gate is
    entry-level: each owned (name, scope) must be committed exactly as rendered, and no
    committed entry of an owned family may be one this script would not render.
    """
    def parse(blocks: list[str]) -> dict:
        out = {}
        for blk in blocks:
            ((name, body),) = yaml.safe_load(blk.replace("  - ", "- ", 1))[0].items()
            out[(name, tuple(body["scope"]["hardware"]))] = body
        return out

    expected = {}
    for sku, chip in DESCRIPTOR_PARTS:
        expected |= parse(descriptor_entries(root, sku, chip))
    expected |= {k: v for k, v in parse(moe_entries(root, "a100_sxm", "a100-sxm")).items()
                 if k[0] == "moe_routing_imbalance_median"}
    committed = {}
    for e in yaml.safe_load(committed_path.read_text())["coefficients"]:
        ((name, body),) = e.items()
        key = (name, tuple(body["scope"]["hardware"]))
        if name == "host_link_bandwidth" and set(key[1]) <= MEASURED_HOST_LINK:
            continue
        if name in DESCRIPTOR_NAMES or key in expected:
            committed[key] = body
    problems = [f"missing: {k}" for k in expected if k not in committed]
    problems += [f"not rendered by this script: {k}" for k in committed if k not in expected]
    for k in expected.keys() & committed.keys():
        for field in ("value", "units", "method", "fitted", "scope", "sources", "rationale"):
            if expected[k].get(field) != committed[k].get(field):
                problems.append(f"{k}: {field} differs")
    for line in problems:
        print(line, file=sys.stderr)
    print(f"{len(expected)} owned entries checked, {len(problems)} problems")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    if argv[1:] == ["--check"]:
        data = Path(os.environ.get(
            "AISIMULATE_DATA",
            "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data"))
        return check(data.parent, Path(__file__).resolve().parent.parent
                     / "coefficients" / "cost-model-primitives.yaml")
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    root, catalog = Path(argv[1]), Path(argv[2])
    out = [HEADER]
    total = 0
    for sku, chip in sorted(SKUS.items()):
        if sku in SKIPPED_SKUS:
            continue
        block = (
            gemm_entries(root, catalog, sku, chip)
            + moe_entries(root, sku, chip)
            + descriptor_entries(root, sku, chip)
        )
        if not block:
            continue
        out.append(f"\n  # --- {chip} ({sku}) ---\n")
        out.extend(block)
        total += len(block)
    out.append(f"\n# {total} entries generated.\n")
    sys.stdout.write("".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
