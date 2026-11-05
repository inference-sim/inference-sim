"""Every generated coefficient must survive an independent re-derivation.

The two cost-model sets are generated (scripts/emit_primitives.py and
scripts/emit_collectives.py), which removes transcription error but not the risk
that a generator is wrong. So this test re-derives each value by a second path —
the standalone fit scripts, which read the parquet files directly — and compares.

It is a real check rather than a tautology because the two paths differ where it
matters: the generators pin a collection by name and the fit scripts take one as
an argument, so a generator that silently selected a different collection than the
one its citation names would be caught here. That is not hypothetical; it happened
during development, when a "newest version" heuristic picked a 3 564-shape MoE
collection over the 42 174-shape one the citation named, changing the imbalance
median from 1.056 to 1.002.

Skipped when the AISimulate tree is absent, since it is not vendored here.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

DATA = Path(
    os.environ.get(
        "AISIMULATE_DATA",
        "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data",
    )
)
REPO = Path(__file__).resolve().parent.parent

# A directory that EXISTS but is EMPTY must skip too, not run and fail.
#
# The tree is read from $AISIMULATE_DATA, which defaults under /tmp and so is reaped: it
# was found once with 1,427 directories and 0 files. `not DATA.is_dir()` passes for that
# hollow tree, so twenty re-derivation tests ran and failed with "no fit" / "no parts
# fitted" -- indistinguishable, at a glance, from a coefficient regression. Probing for
# one parquet tells the two apart, and the reason string says which state was found.
_PARQUETS = next(DATA.rglob("*.parquet"), None) if DATA.is_dir() else None
pytestmark = pytest.mark.skipif(
    _PARQUETS is None,
    reason=(f"AISimulate data not present at {DATA}" if not DATA.is_dir()
            else f"AISimulate tree at {DATA} holds no parquet: re-fetch with "
                 f"`git clone https://github.com/ai-dynamo/aisimulate`"),
)

# (sku, catalog chip, gemm collection, nccl version)
# (sku, chip, the GEMM collection its coefficients CITE, the NCCL collection).
# The GEMM entries moved to the vLLM lane: vLLM runs its own linear kernels for every
# quantized dtype -- CutlassFP8ScaledMMLinearKernel, FlashInferCuteDslNvFp4LinearKernel,
# and a FlashInfer/DeepGEMM dispatch for fp8_block -- not the single `torch_flow` path the
# TRT-LLM sweep measures. A100 keeps SGLang, the only lane carrying that family for it.
# l40s's newest vLLM gemm collection is 0.24.0; the Hopper and Blackwell parts use 0.25.0
# or 0.27.1 and are covered by the shape-ramp gate below rather than here.
PARTS = [
    ("h200_sxm", "h200", "vllm/0.25.0", "2.29.2"),
    ("h100_sxm", "h100", "vllm/0.25.0", "2.29.2"),
    ("l40s", "l40s", "vllm/0.24.0", "2.27.3"),
    ("gb200", "gb200-nvl72", "vllm/0.27.1", "2.29.2"),
    # a100_sxm's GEMM moved to the vLLM lane too: its vLLM sweep (0.14.0) is smaller
    # than SGLang's but measures the engine being predicted, and scored against the vLLM
    # envelope the vLLM fit wins 0.0673 to 0.1002. Its MoE imbalance stays on SGLang --
    # the vLLM MoE sweep for this part has no `balanced` rows, so no paired ratio exists.
    ("a100_sxm", "a100-sxm", "vllm/0.14.0", "2.27.3"),
]

DTYPE = {
    "bfloat16": "bf16", "float16": "fp16", "fp8": "fp8",
    "fp8_block": "fp8_block", "nvfp4": "nvfp4",
}
NCCL_DTYPE = {"half": "fp16", "int8": "int8"}


def registry_values() -> dict[tuple[str, tuple[str, ...]], float]:
    """Coefficient values keyed by (name, scope tuple), plus one entry per chip.

    The per-chip entries exist because a scope is a SET and the kernel matches
    membership -- `resolve.Scope.Admits` -- so one entry may serve several chips. A
    lookup keyed only on the exact tuple missed every widened scope, and widening is
    legitimate: a100-80 is an alias of a100-sxm and shares its measurements, so both
    names appear in one scope rather than in two entries that could drift apart.

    A chip that appears in more than one scope for the same coefficient would be a
    genuine ambiguity, so that is rejected rather than resolved by order.
    """
    out: dict[tuple[str, tuple[str, ...]], float] = {}
    for name in ("cost-model-primitives", "cost-model-collectives"):
        doc = yaml.safe_load((REPO / "coefficients" / f"{name}.yaml").read_text())
        for item in doc["coefficients"]:
            (key, body), = item.items()
            chips = tuple(body["scope"]["hardware"])
            out[(key, chips)] = body["value"]
            for chip in chips:
                single = (key, (chip,))
                if single in out and out[single] != body["value"]:
                    raise AssertionError(
                        f"{key} resolves to two values for {chip}: "
                        f"{out[single]} and {body['value']}"
                    )
                out[single] = body["value"]
    return out


def run_script(script: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


# Parts whose GEMM ramp is the three-factor shape-aware form, fitted by
# fit_gemm_shape_ramp.py on vllm/0.25.0. Their eps_max and m_half come from that JOINT
# fit, so re-deriving them from the one-factor envelope would check a provenance the
# registry no longer claims. l40s and a100-sxm are absent deliberately: their vLLM gemm
# sweeps are at 0.24.0 and 0.14.0, and pinning one collection per part is what keeps a
# difference between parts attributable to silicon rather than to an engine version.
SHAPE_RAMP_PARTS = {"h200", "h100", "b200", "b300", "gb200-nvl72"}
SHAPE_RAMP_COLLECTION = "vllm/0.25.0"


@pytest.mark.parametrize("sku,chip,collection,nccl", PARTS)
def test_gemm_envelope_survives_independent_refit(sku, chip, collection, nccl):
    if chip in SHAPE_RAMP_PARTS:
        pytest.skip(
            f"{chip} carries the three-factor ramp; "
            f"test_gemm_shape_ramp_survives_independent_refit covers it"
        )
    reg = registry_values()
    out = run_script("fit_gemm_envelope.py", str(DATA / sku / "gemm" / collection))
    checked = 0
    for line in out.splitlines():
        m = re.match(
            r"(\w+)\s+rows=\s*\d+\s+M_values=\s*\d+\s+eps_max=([\d.]+)"
            r"\s+M_half=\s*(\d+)",
            line,
        )
        if not m:
            continue
        suffix = DTYPE[m.group(1)]
        for key, want in (
            (f"gemm_eps_max_{suffix}", float(m.group(2))),
            (f"gemm_m_half_{suffix}", int(m.group(3))),
        ):
            got = reg.get((key, (chip,)))
            assert got is not None, f"{chip}: {key} is not in the registry"
            assert abs(float(got) - want) < 1e-9, (
                f"{chip} {key}: registry holds {got}, an independent refit of "
                f"{collection} gives {want}"
            )
            checked += 1
    assert checked >= 2, f"{chip}: refit produced nothing to compare"


@pytest.mark.parametrize("sku,chip,collection,nccl", PARTS)
def test_collective_floors_and_rates_survive_independent_rederivation(
    sku, chip, collection, nccl
):
    """All three collective parameters must survive an independent re-derivation.

    The transition rate is included because it is the one the step time actually uses:
    the peak is only a ceiling. A generator that emitted a correct floor and peak beside
    a wrong transition rate would price every collective wrongly while looking right.
    """
    reg = registry_values()
    out = run_script("fit_collectives.py", str(DATA / sku / "comm" / "nccl" / nccl))
    checked = 0
    for line in out.splitlines()[2:]:
        parts = line.split()
        # operation dtype ranks floor_us spread peak_GB/s trans_GB/s err2 err3 [note]
        if len(parts) < 9 or parts[3] == "-":
            continue
        op, dtype, ranks = parts[0], parts[1], int(parts[2])
        # all_reduce is fitted from vLLM's own custom kernel, not from NCCL, because vLLM
        # does not call NCCL for a tensor-parallel all-reduce its custom kernel can serve.
        # Re-deriving it from the NCCL sweep would compare the registry against a
        # collection no committed entry cites. The vLLM lane is re-derived separately
        # below, so the coefficient is still checked -- against the right data.
        if op == "all_reduce":
            continue
        floor = float(parts[3])
        # The script reports rates in GB/s for readability; the registry stores
        # bytes per microsecond, which is 1000x. Converting here rather than
        # matching on the printed figure keeps the test honest about units — a
        # mismatch of exactly 1000 is the bug this conversion prevents, and it
        # was the bug that surfaced when the script's columns changed.
        peak = float(parts[5]) * 1000
        transition = float(parts[6]) * 1000
        stem = f"{op}_{NCCL_DTYPE[dtype]}_{ranks}rank_{chip.replace('-', '_')}"
        for key, want, tol in (
            (f"collective_floor_{stem}", round(floor, 2), 0.05),
            (f"collective_peak_rate_{stem}", peak, 50.0),
            (f"collective_transition_rate_{stem}", transition, 50.0),
        ):
            got = reg.get((key, (chip,)))
            assert got is not None, f"{chip}: {key} is not in the registry"
            assert abs(float(got) - want) < tol, (
                f"{chip} {key}: registry holds {got}, an independent re-derivation "
                f"gives {want}"
            )
            checked += 1
    assert checked >= 9, f"{chip}: re-derivation produced too little to compare"


@pytest.mark.parametrize(
    "sku, chip, vllm",
    [
        # h200 and h100 were relaned to vLLM 0.31.0's sweep (relane_allreduce_vllm.py),
        # which AISimulate carries from 7980e95; at the older pinned tree it is absent and
        # the case skips with that reason.
        ("h200_sxm", "h200", "0.31.0"),
        ("h100_sxm", "h100", "0.31.0"),
        ("b200_sxm", "b200", "0.24.0"),
        ("b300_sxm", "b300", "0.24.0"),
    ],
)
def test_all_reduce_survives_rederivation_from_the_vllm_lane(sku, chip, vllm):
    """The all-reduce triple must re-derive from the collection its citations name.

    BLIS prices vLLM serving, and vLLM serves a tensor-parallel all-reduce from its own
    custom kernel rather than from NCCL. The committed values therefore come from
    `custom_allreduce_perf.parquet` in the CUDA-graph lane, which is the lane a decode step
    runs in, and this re-derives them from that file.

    The lane matters enough to pin: on h200 at 8 ranks the NCCL floor is 15.98us against
    4.711us here, and the eager lane is slower than NCCL rather than faster. A test that
    accepted any of the three would accept a 3.4x mispricing.
    """
    # The 0.31.0 sweep is read from the tree relane_allreduce_vllm.py is pinned to
    # (AISIMULATE_ALLREDUCE_DATA, aisimulate 7980e95), the rest from the main pin.
    root = Path(os.environ.get("AISIMULATE_ALLREDUCE_DATA") or DATA) if vllm == "0.31.0" else DATA
    base = root / sku / "comm" / "vllm" / vllm
    if not (base / "custom_allreduce_perf.parquet").is_file():
        pytest.skip(f"{sku}: vllm/{vllm} custom all-reduce sweep absent; set "
                    f"AISIMULATE_ALLREDUCE_DATA to an aisimulate 7980e95 systems/data tree")
    reg = registry_values()
    out = run_script("fit_collectives_vllm.py", str(base))
    checked = 0
    for line in out.splitlines():
        parts = line.replace("#", " ").split()
        # backend ranks floor_us trans_GB/s peak_GB/s err2 err3 flat_n
        if len(parts) < 8 or parts[0] != "vllm_graph":
            continue
        ranks = int(parts[1])
        floor, transition, peak = float(parts[2]), float(parts[3]), float(parts[4])
        stem = f"all_reduce_fp16_{ranks}rank_{chip.replace('-', '_')}"
        for key, want, tol in (
            (f"collective_floor_{stem}", round(floor, 2), 0.05),
            (f"collective_peak_rate_{stem}", peak * 1000, 50.0),
            (f"collective_transition_rate_{stem}", transition * 1000, 50.0),
        ):
            got = reg.get((key, (chip,)))
            assert got is not None, f"{chip}: {key} is not in the registry"
            assert abs(float(got) - want) < tol, (
                f"{chip} {key}: registry holds {got}, an independent re-derivation "
                f"of the vLLM lane gives {want}"
            )
            checked += 1
    assert checked == 9, (
        f"{chip}: expected 3 ranks x 3 coefficients from the vllm_graph lane, "
        f"compared {checked}"
    )


# The one family measured outside AISimulate: AISimulate has no host-link measurement (its
# descriptor states the nominal PCIe rate), and InferenceX CollectiveX swap_blocks times
# vLLM's own copy kernel. A microbenchmark, not an end-to-end row, so it is fit data in
# full (docs/fitting-principles.md section 3). Named here so no other family follows it in
# silently; fit_host_link.py is its writer, gated in test_deepep_ll_graph.py.
#
# The second: the Marlin NVFP4 W4A16 grouped ramp on h100/h200. AISimulate carries no Marlin
# NVFP4 MoE rows (its Hopper Marlin rows are MXFP4 and int4 only, through 7980e95); InferenceX
# OperatorX times vLLM's own MarlinExperts NVFP4 through FusedMoE, an operator microbenchmark.
# It lives in cost-model-primitives because that is the set a scenario loads the ramp from;
# fit_marlin_nvfp4_grouped_ramp.py is its writer, gated in test_operatorx.py.
MICROBENCHMARK_SOURCED = {
    "host_link_bandwidth": "InferenceX CollectiveX swap_blocks",
    "gemm_eps_max_w4a16_nvfp4_grouped": "InferenceX OperatorX",
    "gemm_m_half_w4a16_nvfp4_grouped": "InferenceX OperatorX",
}


def test_no_coefficient_cites_a_non_nvidia_measurement_source():
    """Every MEASURED cost-model entry is sourced from NVIDIA AISimulate.

    Mixing measurement sources across a set makes cross-SKU comparisons
    confounded, since two sources apply different methodologies. The host-overhead
    set is exempt: its entries are assumed, cite no measurement, and say so.

    SCOPED TO MEASUREMENTS, which is what the confound is about. An `assumed` entry
    derived from this part's own AISimulate-measured siblings cites those siblings rather
    than a parquet -- that is the honest citation for a derived quantity, and it introduces
    no second methodology, so requiring it to name a collection it was not read from would
    force either a false citation or a deleted one.

    The distinction is load-bearing rather than a convenience: 186 collective entries are
    extrapolated this way (every rank-width gap closed in a737aff, each holdout-validated
    against the one part with a measured 16-rank value), and all 186 are
    `method: assumed, fitted: false`. A measured entry citing something other than
    AISimulate is still a failure, which is the case this test exists to catch.
    """
    for name in ("cost-model-primitives", "cost-model-collectives"):
        doc = yaml.safe_load((REPO / "coefficients" / f"{name}.yaml").read_text())
        for item in doc["coefficients"]:
            (key, body), = item.items()
            if body.get("method") != "measured":
                continue
            if key in MICROBENCHMARK_SOURCED:
                assert all(MICROBENCHMARK_SOURCED[key] in x["cite"] for x in body["sources"]), key
                continue
            for source in body.get("sources", []):
                cite = source["cite"]
                assert "AISimulate" in cite or "aisimulate" in cite, (
                    f"{name}/{key} cites {cite!r}, which is not an AISimulate "
                    f"collection; this set is single-sourced"
                )


def test_no_assumed_entry_smuggles_in_a_foreign_measurement():
    """An `assumed` entry may cite a derivation, but not a rival measurement source.

    This is the half of the single-sourcing guarantee that survives scoping the test above
    to measured entries. A derived coefficient legitimately cites the siblings it was
    derived from; what it must not do is import a number measured by someone else, which
    would reintroduce exactly the methodology confound -- just under an `assumed` label.

    Enforced by naming the rival sources rather than by requiring an AISimulate citation,
    since a derivation's honest citation names no collection at all.
    """
    # Measurement sources that are NOT AISimulate and would confound a cross-SKU
    # comparison if a value were taken from them. FPM and InferenceX are this project's
    # other two datasets and are the realistic mistakes: FPM cannot identify a
    # per-primitive term at all (docs/methodology.md), and InferenceX is evaluation-only.
    RIVALS = ("fpm", "forward pass model", "inferencex", "semianalysis",
              "aiconfigurator", "vidur", "datasheet-only")
    for name in ("cost-model-primitives", "cost-model-collectives"):
        doc = yaml.safe_load((REPO / "coefficients" / f"{name}.yaml").read_text())
        for item in doc["coefficients"]:
            (key, body), = item.items()
            if body.get("method") == "measured":
                continue  # covered by the test above
            for source in body.get("sources", []):
                low = source["cite"].lower()
                for rival in RIVALS:
                    assert rival not in low, (
                        f"{name}/{key} is method={body.get('method')!r} but cites "
                        f"{rival!r}: {source['cite']!r}. A derived entry may cite the "
                        f"siblings it was derived from; importing a value measured by "
                        f"another source reintroduces the methodology confound this set "
                        f"is single-sourced to avoid."
                    )


@pytest.mark.parametrize("sku,chip", sorted(
    (sku, chip) for sku, chip in {
        "h200_sxm": "h200", "h100_sxm": "h100", "b200_sxm": "b200",
        "b300_sxm": "b300", "gb200": "gb200-nvl72",
    }.items()
))
def test_gemm_shape_ramp_survives_independent_refit(sku, chip):
    """Re-derive all four ramp coefficients from the sweep their citation names.

    SKIPPED while the registry carries only the one-factor ramp. The three-factor form
    is fitted by scripts/fit_gemm_shape_ramp.py and implemented in the kernel, and its
    coefficients are deliberately NOT applied: it is several times closer per shape --
    the one-factor ramp over-predicts efficiency by up to 700x at small output width --
    and it makes the end-to-end score worse, because the error it removes was cancelling
    another. docs/perf-model/hypothesis-log.md has both measurements. Gating on the
    registry rather than on a hardcoded list means this test starts checking the moment
    those coefficients land, and does not fail for their absence.

    The three-factor form fits eps_max, m_half, k_half and n_half TOGETHER, so all four
    have to be checked against one re-run. Checking a subset would let the registry hold
    a mixture of two fits -- which is the defect this replaces: eps_max and m_half were
    the envelope fit's while the shape terms were absent, so every GEMM in a layer read
    the same efficiency.
    """
    reg = registry_values()
    if not any(k.startswith("gemm_k_half_") and chip in scope
               for (k, scope) in reg):
        pytest.skip(
            f"{chip} carries the one-factor ramp; the three-factor coefficients are "
            f"fitted and implemented but not applied"
        )
    out = run_script("fit_gemm_shape_ramp.py", "--emit", f"{sku}:{chip}")
    seen = {}
    name = None
    for line in out.splitlines():
        m = re.match(r"^  - (gemm_\w+):$", line)
        if m:
            name = m.group(1)
            continue
        m = re.match(r"^      value: ([\d.]+)$", line)
        if m and name:
            seen[name] = float(m.group(1))
            name = None
    assert seen, f"{chip}: the fitter emitted no coefficients"
    checked = 0
    for key, want in seen.items():
        got = reg.get((key, (chip,)))
        assert got is not None, f"{chip}: {key} is not in the registry"
        assert abs(float(got) - want) < 1e-9, (
            f"{chip} {key}: registry holds {got}, an independent refit of "
            f"{SHAPE_RAMP_COLLECTION} gives {want}"
        )
        checked += 1
    # Four coefficients per dtype, and a part carries at least bf16, fp8 and fp8_block.
    assert checked >= 12, f"{chip}: only {checked} coefficients re-derived"


# --------------------------------------------------------------------------------------
# DeepSeek-V4.1: the committed entries through the CONTRACT's byte counts
# --------------------------------------------------------------------------------------
#
# The kernel reader evaluates these entries with fixed byte counts (the coefficient contract
# and each rationale). A writer that changed its own W -- dropping the BF16 factor of the
# mHC residual, say -- would regenerate rates that fit its own W and pass its own --check,
# while the kernel priced every site 2x off. So these tests rebuild W from the contract's
# literals, independently of the writers' regressor code, and assert the stated error.

V41 = "deepseek-v4.1-flash"
V41_PARTS = [("h200_sxm", "h200"), ("h100_sxm", "h100")]
MHC_BYTES_PER_HIDDEN = 8      # hc_mult 4 x BF16, per token per hidden element (40,960 B at 5120)
FLASHMLA_BLOCK_BYTES = 64 * 584   # one 64-token FlashMLA block of the 584 B V4 record
V41_WINDOW = 128


def _v41(name: str, chip: str) -> float:
    for setname in ("cost-model-attention", "cost-model-primitives"):
        doc = yaml.safe_load((REPO / "coefficients" / f"{setname}.yaml").read_text())
        for e in doc["coefficients"]:
            ((n, body),) = e.items()
            if n == name and body["scope"] == {"hardware": [chip], "model": [V41]}:
                return float(body["value"])
    raise AssertionError(f"no {name} for {chip}/{V41}")


def _geo(pred, y):
    import math
    logs = [math.log(p / m) for p, m in zip(pred, y)]
    return (math.exp(sum(abs(x) for x in logs) / len(logs)),
            math.exp(sum(logs) / len(logs)))


@pytest.mark.parametrize("sku,chip", V41_PARTS)
@pytest.mark.parametrize("op", ["pre", "post"])
def test_v41_mhc_entries_price_the_sweep_at_the_contracts_bytes(sku, chip, op):
    """W = tokens x hidden x 8 B per mHC site (a row times two sites)."""
    import pyarrow.parquet as pq
    t = pq.read_table(DATA / sku / "mhc" / "vllm" / "0.25.0" / "mhc_module_perf.parquet"
                      ).to_pydict()
    floor, rate = _v41(f"mhc_floor_{op}", chip), _v41(f"mhc_rate_{op}", chip)
    pred, y = [], []
    for i in range(len(t["latency"])):
        if t["op_name"][i] != op:
            continue
        w = t["num_tokens"][i] * t["hidden_size"][i] * MHC_BYTES_PER_HIDDEN
        pred.append(floor + w / rate)
        y.append(t["latency"][i] * 1000.0 / 2)
    err, signed = _geo(pred, y)
    assert len(y) >= 60 and err < 1.12 and 0.9 < signed < 1.1, (
        f"{chip} mhc {op}: {err:.3f}x signed {signed:.3f} over {len(y)} rows at the "
        f"contract's 8 B per hidden element; the committed rate was fitted on other bytes")


def _flashmla_decode_rows(sku):
    import math
    import pyarrow.parquet as pq
    t = pq.read_table(DATA / sku / "sparse_attention" / "vllm" / "0.25.0" /
                      "dsv4_hca_attn_module_perf.parquet").to_pydict()
    out = []
    for i in range(len(t["latency"])):
        if t["isl"][i] != 1 or t["model"][i] != "sgl-project/DeepSeek-V4-Flash-FP8":
            continue
        ctx = t["step"][i] + 1
        blocks = math.ceil(min(ctx, V41_WINDOW) / 64) + math.ceil((ctx // 128) / 64)
        out.append((t["batch_size"][i], blocks, t["latency"][i] * 1000.0))
    return out


@pytest.mark.parametrize("sku,chip", V41_PARTS)
def test_v41_sparse_decode_rate_prices_the_sweep_at_the_contracts_bytes(sku, chip):
    """The single-rate pair: part-wide floor + batch x blocks x 37,376 B / rate."""
    doc = yaml.safe_load((REPO / "coefficients" / "cost-model-attention.yaml").read_text())
    floor = next(float(b["value"]) for e in doc["coefficients"] for n, b in e.items()
                 if n == "attention_decode_floor" and b["scope"] == {"hardware": [chip]})
    rate = _v41("attention_decode_rate_sparse_mla", chip)
    rows = _flashmla_decode_rows(sku)
    err, signed = _geo([floor + r * b * FLASHMLA_BLOCK_BYTES / rate for r, b, _ in rows],
                       [y for *_, y in rows])
    assert err < 1.25 and 0.95 < signed < 1.15, (chip, err, signed)
