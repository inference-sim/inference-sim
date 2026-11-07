"""Properties the committed coefficients must satisfy whatever the sweeps contain.

WHY PROPERTIES RATHER THAN PINNED VALUES. The re-derivation gates
(test_generated_from_aisimulate.py, test_writers_are_idempotent.py) assert that each
committed number is what its fitter produces. That catches drift but says nothing about
whether the number is PHYSICALLY SENSIBLE: a fitter re-run after a unit slip reproduces
its own mistake exactly and both gates stay green.

These tests encode what the cost model's own physics implies, so they hold for any part,
any future sweep, and any refit. Each one below is a defect this project actually shipped
at least once, which is the bar for including it:

  * A rate stated as a fraction of datasheet bandwidth once landed at 2.00 of peak -- a
    physically impossible figure -- because a windowed kernel's byte count was computed
    from the context length rather than the window.
  * A 1000x unit error (ms read as s) produced a peak of 143,314 GB/s on h200 and a
    three-parameter error worse than the two-parameter one.
  * Pooling geometries in a sweep that varies them produced a recurrent rate 6.4x too
    fast (KDA at num_k_heads=12 for a 96-head model) and a GDN rate 1.27x-1.80x off.

They read only the committed files and the catalog, so they run in CI with no AISimulate
tree present. That is deliberate: a property that needs the sweep to check is a
re-derivation, and those live elsewhere.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent

# The chip descriptors these properties check against: the repository's own catalog, so
# they are present wherever the suite runs.
#
# This default used to be /Users/sri/Documents/Projects/blis-catalog, a path that exists on
# one machine. In CI `chip_facts` returned None for every chip, so
# `test_no_bandwidth_rate_exceeds_its_parts_datasheet_peak` examined ZERO rate/chip pairs
# and only its own `checked >= 10` guard turned that into a failure rather than a
# vacuous pass. The companion fraction-of-peak test had the same blind spot, and the
# hardware-coverage test below skipped outright. A physical bound that does not run is
# worse than no bound: it reports the invariant as protected.
#
# 40 KB for the nine chips the committed coefficients scope to, pinned to blis-catalog
# 0.5.0 (b5550f750cb51571498577fbdc1c68f78cda67db),
# the revision blis-latency-kernel's testdata/upstream.lock pins too. BLIS_CATALOG still overrides, so a working copy can be
# checked against a live catalog; a chip rename there is SUPPOSED to break these tests,
# which is why the copy tracks a commit rather than a branch.
#
# `or` rather than a two-argument `os.environ.get`: an EMPTY BLIS_CATALOG must fall back to
# the vendored copy, not be honoured as a root. `BLIS_CATALOG= pytest ...` is the natural
# way to ask for the default, and get() treats the empty string as a real value -- which
# resolves every chip path relative to nothing, returns None for all of them, and lands
# back in the vacuous-pass state this vendoring exists to prevent.
CATALOG = Path(os.environ.get("BLIS_CATALOG")
               or REPO.parent / "catalog")


def committed() -> list[tuple[str, str, dict]]:
    """Every committed entry as (set-name, coefficient-name, body)."""
    out = []
    for path in sorted(glob.glob(str(REPO / "coefficients" / "*.yaml"))):
        doc = yaml.safe_load(Path(path).read_text())
        for e in doc["coefficients"]:
            (name, body), = e.items()
            out.append((os.path.basename(path), name, body))
    return out


ENTRIES = committed()


def chip_facts(chip: str) -> dict | None:
    p = CATALOG / "hardware" / f"{chip}.yaml"
    if not p.is_file():
        return None
    return yaml.safe_load(p.read_text())


# --------------------------------------------------------------------------------------
# Physical bounds
# --------------------------------------------------------------------------------------

def test_no_bandwidth_rate_exceeds_its_parts_datasheet_peak():
    """A measured byte rate cannot beat the memory it reads from.

    `attention_decode_rate*` and the MLA rate are bytes per microsecond, and no kernel
    sustains more than the datasheet HBM figure. An entry above it means the byte count
    the fit divided by was wrong -- which is exactly how an early windowed fit reached
    2.00 of peak.
    """
    checked = 0
    for setname, name, body in ENTRIES:
        if not re.match(r"attention_decode_rate(_|$)", name):
            continue
        for chip in (body.get("scope") or {}).get("hardware", []):
            facts = chip_facts(chip)
            if not facts or "BwPeakTBs" not in facts:
                continue
            peak_bytes_per_us = facts["BwPeakTBs"] * 1e6
            assert body["value"] <= peak_bytes_per_us, (
                f"{setname}: {name} on {chip} is {body['value']:,.0f} B/us, above the "
                f"part's {peak_bytes_per_us:,.0f} B/us datasheet peak. A kernel cannot "
                f"read faster than its memory; the byte count in the fit is wrong."
            )
            checked += 1
    assert checked >= 10, f"only {checked} rate/chip pairs checked; the filter is wrong"


# The families whose bytes_per_us value is a rate through device memory. Link, host and
# interconnect rates (collective_*, moe_a2a_*, host_*) are bounded by their own links, not by
# HBM, and are not in this list.
HBM_RATE_FAMILY = re.compile(r"(attention_|mhc_)\w*rate")
# The work rates committed with rate_basis: work. A work rate may exceed the memory peak, so
# marking an entry "work" would otherwise be a way around SUB-5. The count may only go down:
# lower it when an entry is refitted as a memory rate, never raise it to admit a new one.
# (Four entries: the sparse-MLA and block-index prefill rates on h100 and on h200.)
MAX_WORK_RATES = 4


def memory_rate_problems(entries, facts_of):
    """SUB-5: no memory rate above the chip's datasheet memory bandwidth.

    Returns (problems, examined). A rate is a memory rate unless it says rate_basis: work.
    A work rate must say why in its rationale. A memory rate with no chip to compare against
    is a problem too, so the check cannot pass by examining nothing.
    """
    problems, examined = [], 0
    for setname, name, body in entries:
        if body.get("units") != "bytes_per_us" or not HBM_RATE_FAMILY.match(name):
            continue
        basis = body.get("rate_basis", "memory")
        if basis == "work":
            if not str(body.get("rationale", "")).strip():
                problems.append(f"{setname}: {name} is a work rate with no rationale")
            continue
        for chip in (body.get("scope") or {}).get("hardware", []):
            facts = facts_of(chip)
            if not facts or "BwPeakTBs" not in facts:
                problems.append(f"{setname}: {name} names chip {chip}, which has no datasheet "
                                f"memory bandwidth to compare against")
                continue
            examined += 1
            peak = facts["BwPeakTBs"] * 1e6
            if body["value"] > peak:
                problems.append(
                    f"{setname}: {name} on {chip} is {body['value']:,.0f} B/us, "
                    f"{body['value'] / peak:.2f}x the part's {peak:,.0f} B/us datasheet memory "
                    f"bandwidth. A kernel cannot read faster than its memory: fix the byte "
                    f"count in the fit, or, if the rate counts work rather than memory "
                    f"traffic, give it rate_basis: work and say why in its rationale.")
    return problems, examined


def test_no_memory_rate_exceeds_its_parts_datasheet_peak():
    """SUB-5, every family that reads device memory, not only attention decode."""
    problems, examined = memory_rate_problems(ENTRIES, chip_facts)
    assert not problems, "\n".join(problems)
    assert examined >= 36, f"only {examined} memory rate/chip pairs examined; the filter is wrong"


def test_work_rates_are_few_and_only_go_down():
    work = [(s, n, b["scope"]["hardware"]) for s, n, b in ENTRIES
            if b.get("rate_basis") == "work"]
    assert len(work) <= MAX_WORK_RATES, (
        f"{len(work)} work rates, above the {MAX_WORK_RATES} committed: {work}. A work rate "
        f"is allowed to exceed the memory peak, so each one is a reviewed exception.")


def test_the_memory_rate_check_fails_on_its_known_bad_inputs():
    """X18: the check must fail on an input it exists to refuse, and pass on its neighbour."""
    facts = {"h200": {"BwPeakTBs": 4.8}}.get
    rate = lambda v, **kw: [("s.yaml", "attention_decode_rate",
                             {"units": "bytes_per_us", "value": v, "scope": {"hardware": ["h200"]}, **kw})]
    peak = 4.8e6
    assert memory_rate_problems(rate(peak), facts)[0] == []                     # at peak: allowed
    assert memory_rate_problems(rate(peak * 1.001), facts)[0]                   # just above: refused
    assert memory_rate_problems(rate(2 * peak, rate_basis="work", rationale="counts work"), facts)[0] == []
    assert memory_rate_problems(rate(2 * peak, rate_basis="work"), facts)[0]    # work, no reason
    assert memory_rate_problems(rate(1.0), {}.get)[0]                           # unknown chip
    assert memory_rate_problems(rate(1e9), facts)[0]                            # 1000x slip, high


def test_every_bandwidth_rate_is_a_plausible_fraction_of_peak():
    """And not implausibly far below it either.

    Below ~2% of datasheet peak a 'measured bandwidth' is more likely a unit slip than a
    slow kernel. The lowest committed figure is l40s SWA at ~0.25 of peak, so 0.02 is a
    wide floor that still catches a 1000x error.
    """
    for setname, name, body in ENTRIES:
        if not re.match(r"attention_decode_rate(_|$)", name):
            continue
        for chip in (body.get("scope") or {}).get("hardware", []):
            facts = chip_facts(chip)
            if not facts or "BwPeakTBs" not in facts:
                continue
            frac = body["value"] / (facts["BwPeakTBs"] * 1e6)
            assert frac >= 0.02, (
                f"{setname}: {name} on {chip} is {frac:.4f} of datasheet peak. That is "
                f"below any measured kernel in this registry and is the signature of a "
                f"unit error (ms read as s gives 1/1000)."
            )


def test_gemm_efficiency_asymptotes_are_fractions():
    """`gemm_eps_max_*` is a fraction of peak FLOPs: 0 < eps <= 1."""
    checked = 0
    for setname, name, body in ENTRIES:
        if not name.startswith("gemm_eps_max"):
            continue
        assert 0 < body["value"] <= 1.0, (
            f"{setname}: {name} = {body['value']} is not a fraction. The GEMM ramp's "
            f"asymptote is the share of this dtype's peak a large matmul reaches."
        )
        checked += 1
    assert checked >= 20, f"only {checked} eps_max entries checked"


def test_moe_imbalance_is_near_one():
    """The imbalance multiplier is a ratio of skewed to balanced latency at one shape.

    It cannot be far from 1 -- it is the same work on the same kernel, differing only in
    routing distribution. (A companion p90 family was retired in #36 because nothing
    read it; its value is stated in each median's rationale.)
    """
    med = {}
    for _, name, body in ENTRIES:
        for chip in (body.get("scope") or {}).get("hardware", []):
            if name == "moe_routing_imbalance_median":
                med[chip] = body["value"]
    assert med, "no imbalance medians found"
    for chip, v in sorted(med.items()):
        assert 0.5 <= v <= 2.0, (
            f"moe_routing_imbalance_median on {chip} is {v}, implausible for a ratio of "
            f"the same kernel at the same shape under two routing distributions."
        )


def test_retired_families_stay_retired():
    """Families removed because nothing in BLIS reads them (#36) must not creep back.

    hbm_constant_latency and p2p_latency are NVIDIA-simulator constants whose fixed cost
    the kernel's measured attention and collective floors already carry, so a reader
    would double-count; moe_routing_imbalance_p90 is a tail a mean-step model does not
    price. Re-adding one needs a reader first, and this test is where that is decided.
    """
    retired = {"hbm_constant_latency", "p2p_latency", "moe_routing_imbalance_p90"}
    found = sorted({name for _, name, _ in ENTRIES} & retired)
    assert not found, f"retired families are committed again: {found}"


def test_a_model_scoped_entry_never_shadows_a_part_wide_one():
    """A name may be scoped to one model, or part-wide, on a given part -- not both, unless
    the two resolve the same under either rule a kernel applies.

    blis-latency-kernel through 993ddeb admits every entry whose scope covers the
    deployment and keeps the LAST in file order, so a model-scoped entry and a part-wide
    one of the same name on the same part would make the price depend on where each sits
    in the file. From the DeepSeek-V4.1 reader on, a model-scoped entry wins over an
    unscoped one regardless of order. A model-scoped entry placed AFTER its part-wide
    namesake in the SAME set is chosen by both rules -- last in file order, and scoped --
    so that one arrangement is allowed: it is how a model with its own measured host cost
    (DeepSeek-V4.1's admission) overrides the part-wide estimate without a new name the
    kernel would not read. Any other arrangement still fails.
    """
    partwide: dict = {}
    scoped = []
    for i, (setname, name, body) in enumerate(ENTRIES):
        scope = body.get("scope") or {}
        for hw in scope.get("hardware", []):
            if "model" in scope:
                scoped.append((setname, name, hw, scope["model"], i))
            else:
                partwide.setdefault((name, hw), []).append((setname, i))
    clashes = [s[:4] for s in scoped
               if any(ps != s[0] or pi > s[4] for ps, pi in partwide.get((s[1], s[2]), []))]
    assert not clashes, (
        f"model-scoped entries that a part-wide entry of the same name also admits: "
        f"{clashes}. Give the model-scoped quantity its own name, or scope both."
    )


def test_every_name_resolves_to_one_entry_per_part_and_model():
    """blis-latency-kernel keeps the LAST admitted entry of a name, so two entries that
    admit the same (part, model) make the price depend on file order and leave one copy
    unmaintained -- the a100-80 descriptor duplicates were this, with equal values only by
    luck. Holds for every set, with no sweep."""
    seen: dict = {}
    clashes = []
    for setname, name, body in ENTRIES:
        scope = body.get("scope") or {}
        models = scope.get("model") or [None]
        for hw in scope.get("hardware", []):
            for m in (models if isinstance(models, list) else [models]):
                key = (setname, name, hw, m)
                if key in seen:
                    clashes.append(key)
                seen[key] = True
    assert not clashes, f"names admitted twice for one part and model: {clashes}"


def _single(name: str, chip: str, model: str | None = None) -> float:
    hits = [b["value"] for _, n, b in ENTRIES
            if n == name and (b.get("scope") or {}).get("hardware") == [chip]
            and (b.get("scope") or {}).get("model") == ([model] if model else None)]
    assert len(hits) == 1, f"expected one {name} on {chip} (model {model}), got {hits}"
    return hits[0]


def test_marlin_weight_only_ramps_sit_at_or_below_the_dense_bf16_envelope():
    """Marlin multiplies BF16 activations with mma.sync after dequantizing the weights.

    Its ramp is a fraction of the BF16 peak, so on any part it cannot exceed the dense
    BF16 envelope measured on cuBLAS. A grouped fit above it would mean the routed term's
    FLOP or byte count in the fit was wrong, not that Marlin is faster.
    """
    checked = 0
    for _, name, body in ENTRIES:
        # The W8A16 entry is assumed EQUAL to the BF16 envelope (see the derived-entry test
        # below), so only the fitted grouped ramp can actually violate this bound.
        if name not in ("gemm_eps_max_w4a16_mxfp4_grouped", "gemm_eps_max_w4a16_nvfp4_grouped"):
            continue
        for hw in body["scope"]["hardware"]:
            bf16 = _single("gemm_eps_max_bf16", hw)
            assert body["value"] <= bf16, (
                f"{name} on {hw} is {body['value']}, above the part's dense BF16 envelope "
                f"{bf16}; a weight-only BF16-compute kernel cannot exceed it.")
            checked += 1
    assert checked >= 1, f"only {checked} Marlin ramp entries checked"


@pytest.mark.parametrize("hw", ["h200", "h100"])
def test_the_derived_deepseek_v41_entries_equal_what_they_are_derived_from(hw):
    """The four `assumed` entries fit_deepseek_v41_h200.py writes per part are each a
    function of THAT part's committed siblings and catalog descriptor, so the relation is
    checkable in CI with no sweep. A sibling refitted without re-running the writer, or an
    h100 value copied from h200, fails here rather than leaving an assumed value derived
    from a number the registry no longer carries."""
    m = "deepseek-v4.1-flash"
    bw = chip_facts(hw)["BwPeakTBs"] * 1e6
    assert _single("attention_decode_rate_block_index_scores", hw, m) == round(
        bw * _single("hbm_derate", hw))
    assert _single("gemm_eps_max_w8a16_mxfp8", hw) == _single("gemm_eps_max_bf16", hw)
    assert _single("gemm_m_half_w8a16_mxfp8", hw) == _single("gemm_m_half_bf16", hw)
    # The descriptor's nominal 64 GB/s x 128/130 x useful/fetched bytes. Deliberately not
    # host_link_bandwidth, which is the measured copy-engine payload rate (fit_host_link.py).
    assert _single("host_gather_rate_uva", hw) == round(64000.0 * 128 / 130 * (256 + 8) / (256 + 32))


def test_the_v41_host_entries_re_derive_from_their_vendored_measurements():
    """`fit_deepseek_v41_host.py --check`: the eager launch cost and the admission entries
    re-derive from validator/testdata/anchors/, which needs no AISimulate tree, so this runs
    in every CI job rather than only the rederive one."""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, str(repo / "scripts" / "fit_deepseek_v41_host.py"),
                        "--check"], capture_output=True, text=True, cwd=str(repo),
                       timeout=300)
    assert r.returncode == 0, (
        "the DeepSeek-V4.1 host entries are not what their vendored measurements give:\n"
        f"{r.stdout}\n{r.stderr}")


@pytest.mark.parametrize("hw", ["h200", "h100"])
def test_the_v41_sparse_decode_form_is_physical_and_complete(hw):
    """The two-branch decode form ships all five members or none (the reader is all-or-none),
    its throughput rate is below the part's HBM peak, and its launch floor is below the
    part-wide attention floor it replaces at one block -- a floor above it would mean a
    module, not the kernel, was timed."""
    m = "deepseek-v4.1-flash"
    names = ["launch_floor", "split_floor", "serial_rate", "per_token", "throughput_rate"]
    vals = {n: _single(f"attention_decode_{n}_sparse_mla", hw, m) for n in names}
    peak = chip_facts(hw)["BwPeakTBs"] * 1e6
    assert 0 < vals["throughput_rate"] < peak
    assert 0 < vals["serial_rate"] < vals["throughput_rate"]
    assert 0 < vals["launch_floor"] <= _single("attention_decode_floor", hw)
    assert 0 < vals["split_floor"] < vals["launch_floor"]
    assert 0 < vals["per_token"] < 1.0


@pytest.mark.parametrize("hw", ["h200", "h100"])
def test_the_v41_mhc_rates_are_below_the_parts_hbm_peak(hw):
    """mHC streams the hc_mult residual copies through HBM once a step; a fitted rate over
    those bytes above the datasheet bandwidth would mean the byte count W is wrong."""
    m = "deepseek-v4.1-flash"
    peak = chip_facts(hw)["BwPeakTBs"] * 1e6
    for op in ("pre", "post"):
        assert 0 < _single(f"mhc_rate_{op}", hw, m) < peak


# --------------------------------------------------------------------------------------
# Monotonicity across rank width: the collective floors
# --------------------------------------------------------------------------------------

def _collectives():
    pat = re.compile(
        r"^collective_(floor|peak_rate|transition_rate)_"
        r"(all_reduce|all_gather|reduce_scatter|alltoall)_"
        r"(fp16|int8)_(\d+)rank_(.+)$")
    out = collections.defaultdict(dict)
    for _, name, body in ENTRIES:
        m = pat.match(name)
        if m:
            kind, op, dt, w, chip = m.groups()
            out[(chip, op, dt, kind)][int(w)] = body["value"]
    return out


COLLECTIVES = _collectives()


def test_collective_floors_do_not_fall_as_the_group_widens():
    """A wider collective cannot have a LOWER launch floor.

    More ranks means more synchronisation, and the measured 4->8 all-reduce ratio is
    1.53x-1.91x on every part swept at both widths. A floor that falls with width means
    two widths were fitted from different row sets or different collections.

    Stated as non-decreasing rather than strictly increasing because an alltoall's floor
    is nearly flat in width (median 4->8 ratio 1.090) and an equal reading is credible.
    """
    checked = 0
    for (chip, op, dt, kind), byw in sorted(COLLECTIVES.items()):
        if kind != "floor" or len(byw) < 2:
            continue
        widths = sorted(byw)
        for a, b in zip(widths, widths[1:]):
            assert byw[b] >= byw[a], (
                f"{op} {dt} on {chip}: the {b}-rank floor ({byw[b]}us) is BELOW the "
                f"{a}-rank floor ({byw[a]}us). A wider group synchronises more, so its "
                f"floor cannot be lower."
            )
            checked += 1
    assert checked >= 20, f"only {checked} width pairs checked"


def test_collective_rates_do_not_rise_steeply_as_the_group_widens():
    """Per-GPU throughput does not IMPROVE MATERIALLY with more ranks.

    A loose bound, deliberately, because the strict version is false and the measurements
    say why. Eight committed (part, op, dtype, kind) pairs do rise with width, and they
    are not defects:

      * h100/h200 fp16 all-reduce rises 1.13x from 4 to 8 ranks on the vLLM lane. That
        is vLLM's one-shot custom kernel, not a ring: every rank writes once and reads
        once, so a wider group amortises the launch over more payload instead of adding
        ring hops. The committed entries cite `custom_allreduce_perf.parquet` precisely
        because this kernel behaves differently from NCCL.
      * Several NCCL reduce_scatter and alltoall pairs rise 1.007x-1.041x, which is grid
        resolution on a near-flat curve rather than a measured improvement.

    So the invariant worth asserting is that nothing rises STEEPLY: a rate that nearly
    doubles with width would mean two widths were fitted from different collections,
    which is the defect this guards. 1.25x sits well above the 1.13x the one-shot kernel
    genuinely reaches and well below any plausible cross-collection mix.
    """
    for (chip, op, dt, kind), byw in sorted(COLLECTIVES.items()):
        if kind not in ("peak_rate", "transition_rate") or len(byw) < 2:
            continue
        widths = sorted(byw)
        for a, b in zip(widths, widths[1:]):
            assert byw[b] <= byw[a] * 1.25, (
                f"{op} {dt} {kind} on {chip}: the {b}-rank figure ({byw[b]}) is "
                f"{byw[b] / byw[a]:.2f}x the {a}-rank one ({byw[a]}). A collective does "
                f"not get materially faster per GPU as it widens; this is the signature "
                f"of two widths fitted from different collections."
            )


def test_transition_rate_never_exceeds_peak_rate():
    """The transition rate is the rate in the ramp; the peak is the ceiling.

    `liftCollectiveFloors` prices a collective as max(floor + size/transition,
    size/peak), so a transition above peak makes the second term unreachable and the
    peak coefficient dead.
    """
    checked = 0
    for (chip, op, dt, kind), byw in sorted(COLLECTIVES.items()):
        if kind != "transition_rate":
            continue
        peaks = COLLECTIVES.get((chip, op, dt, "peak_rate"), {})
        for w, v in byw.items():
            if w in peaks:
                assert v <= peaks[w], (
                    f"{op} {dt} on {chip} at {w} ranks: transition {v} exceeds peak "
                    f"{peaks[w]}. The transition rate sits inside the ramp, below the "
                    f"asymptote."
                )
                checked += 1
    assert checked >= 20, f"only {checked} transition/peak pairs checked"


# --------------------------------------------------------------------------------------
# Recurrent families
# --------------------------------------------------------------------------------------

def test_recurrent_rates_are_slower_than_the_convolution_alone():
    """A conv+scan chain is slower than the conv by itself.

    mamba2 is the convolution ALONE (its selective scan is in no collection) while KDA
    and GDN are conv + recurrent scan on the same part. Serial kernels mean reciprocal
    rates add, so the chained families must have the LOWER tokens/us. This is the
    metamorphic relation that would have caught the original KDA fit, whose 1.50 tok/us
    sat above mamba2's conv-only figure and was 6.4x too fast.
    """
    byfam = collections.defaultdict(dict)
    for _, name, body in ENTRIES:
        m = re.match(r"recurrent_decode_rate_(kda|gdn|mamba2)$", name)
        if not m:
            continue
        for chip in (body.get("scope") or {}).get("hardware", []):
            byfam[m.group(1)][chip] = body["value"]
    shared = set(byfam.get("mamba2", {}))
    assert shared, "no mamba2 rates committed; this relation cannot be checked"
    checked = 0
    for fam in ("kda", "gdn"):
        for chip in sorted(shared & set(byfam.get(fam, {}))):
            assert byfam[fam][chip] < byfam["mamba2"][chip], (
                f"{fam} rate on {chip} ({byfam[fam][chip]} tok/us) is NOT slower than "
                f"the conv-only mamba2 figure ({byfam['mamba2'][chip]}). {fam} is that "
                f"same convolution plus a sequential scan, and serial kernels add "
                f"reciprocal rates, so it must be slower."
            )
            checked += 1
    assert checked >= 6, f"only {checked} family/chip comparisons made"


def test_recurrent_floors_are_positive_and_bounded():
    """A per-layer state update is microseconds, not nanoseconds or milliseconds."""
    for setname, name, body in ENTRIES:
        if not re.match(r"recurrent_decode_floor_", name):
            continue
        assert 0.1 <= body["value"] <= 1000.0, (
            f"{setname}: {name} = {body['value']} us is outside the range a single "
            f"recurrent layer's launch can plausibly take."
        )


# --------------------------------------------------------------------------------------
# Scope and provenance hygiene
# --------------------------------------------------------------------------------------

def test_hardware_independent_sets_cover_every_catalog_part():
    """A set that claims to apply everywhere must name every part the catalog holds.

    The host overheads are CPU work in the engine's Python path, so none has a per-part
    value. They are scoped to an explicit list because the validator rejects an empty
    scope -- which means adding a chip to the catalog silently leaves it priced at ZERO
    until someone edits that list. That has now happened twice: once scoped to
    [h100, h200] (zero on all of Blackwell) and once omitting gb300.
    """
    cat = {os.path.basename(p)[:-5]
           for p in glob.glob(str(CATALOG / "hardware" / "*.yaml"))}
    if not cat:
        pytest.skip(f"catalog not present at {CATALOG}")
    # A PART-WIDE entry (no model) must scope every part on its own, as before: a part-wide
    # host term has no per-part value in its source, so splitting it is never allowed.
    # A MODEL-SCOPED term may be split across entries -- DeepSeek-V4.1's eager launch cost is
    # derived on h200 and copied to the other parts -- as long as together they cover every
    # part AND the split carries one value: every entry of the (set, name, model) has the
    # same value, and any entry beyond the first is `copied` with copied_from naming a
    # sibling's scope. A split may not smuggle in a per-part value.
    split: dict = {}
    for setname, name, body in ENTRIES:
        if setname not in ("cost-model-host-overheads.yaml", "cost-model-memory.yaml"):
            continue
        scope = body.get("scope") or {}
        models = tuple(scope.get("model") or ())
        # `host_*` in cost-model-host-overheads only. `host_link_bandwidth` lives in
        # cost-model-primitives and is a genuinely PER-PART figure -- NVIDIA's descriptor
        # states a different PCIe or NVLink-C2C rate per platform -- so it is scoped to
        # one chip by design and must not be swept into this rule by its name prefix.
        # cost-model-memory holds engine-structural memory terms with no per-part value in
        # their sources, scoped to every part for the same reason the host set is.
        if not models:
            missing = sorted(cat - set(scope.get("hardware", [])))
            assert not missing, (
                f"{setname}: {name} does not scope {missing}, so it resolves to NOTHING "
                f"there and the term is priced at zero. Every part-wide term in this set has "
                f"no per-part value in its source, so the scope must list every catalog part."
            )
            continue
        split.setdefault((setname, name, models), []).append(body)
    for (setname, name, models), bodies in split.items():
        # The model-scoped capture form is read all or none, and where it is absent the
        # kernel charges the part-wide cudagraph_capture_bytes_<mode> (blis-latency-kernel
        # captureMemory), so a part it does not scope is priced, not zeroed.
        if name.startswith(("cudagraph_capture_floor_bytes_",
                            "cudagraph_capture_bytes_per_mixed_token_",
                            "cudagraph_capture_bytes_per_decode_token_")):
            continue
        covered = set().union(*(set(b["scope"].get("hardware", [])) for b in bodies))
        missing = sorted(cat - covered)
        assert not missing, (
            f"{setname}: {name} for {list(models)} does not scope {missing}, so the term is "
            f"priced at zero there.")
        if len(bodies) == 1:
            continue
        values = {b["value"] for b in bodies}
        assert len(values) == 1, (
            f"{setname}: {name} for {list(models)} is split across entries with values "
            f"{sorted(values)}; a host term has no per-part value, so a split must carry one.")
        originals = [b for b in bodies if b.get("method") != "copied"]
        assert len(originals) == 1, (
            f"{setname}: {name} for {list(models)}: a split needs exactly one original entry "
            f"and every other `copied`, found {len(originals)} originals.")
        orig = originals[0]["scope"].get("hardware", [])
        for b in bodies:
            if b is originals[0]:
                continue
            cf = b.get("copied_from", "")
            assert cf.startswith(f"hardware={','.join(orig)}"), (
                f"{setname}: {name} for {list(models)}: copied_from {cf!r} does not name the "
                f"original entry's scope (hardware={','.join(orig)}).")


def test_every_measured_entry_cites_a_source():
    """`measured` means someone measured it, and the citation is where.

    A registry VALUE/QUALITY property, not a schema rule: blis-schemas requires sources
    only for `literature`/`vendor_spec` and deliberately allows a `measured` entry to be
    uncited, so pre-existing measured-but-uncited data can be adopted. This repository holds
    its own committed data to the stricter bar, which the schema does not enforce.
    """
    for setname, name, body in ENTRIES:
        if body.get("method") != "measured":
            continue
        srcs = body.get("sources") or []
        assert srcs, (
            f"{setname}: {name} is method: measured with no sources. A measurement "
            f"whose provenance is absent cannot be re-derived or audited."
        )


def test_every_assumed_entry_explains_itself():
    """`assumed` is a declared gap, so the rationale carries the whole argument.

    A registry VALUE/QUALITY property: blis-schemas (migrated) requires a rationale to be
    PRESENT for a non-measured value; the >200-character thoroughness bar — the house style
    that states the dimension, the magnitude's caveat, and the experiment that would replace
    it — is this repository's own, and the schema does not impose it.
    """
    for setname, name, body in ENTRIES:
        if body.get("method") != "assumed":
            continue
        rat = body.get("rationale") or ""
        assert len(rat) > 200, (
            f"{setname}: {name} is method: assumed with a {len(rat)}-character "
            f"rationale. An assumed value is only as good as its stated reasoning; "
            f"this set's house style states the dimension, the magnitude's caveat, and "
            f"the experiment that would replace it."
        )


# NOTE: schema-shape rules that used to live here — `fitted: true` => `method: measured`,
# the bare presence of a rationale for a non-measured value, and `assumed` excludes `fitted`
# (transitively covered by fitted => measured) — were removed when the registry collapsed to
# a single validator; blis-schemas owns them (migrated in blis-schemas#38). The two checks
# ABOVE are kept deliberately: they are registry value/quality properties on this repo's own
# committed data that the schema does NOT enforce (blis-schemas allows measured-uncited and
# imposes no rationale-length bar), so they are a conscious keep, not a silent drop.

# --------------------------------------------------------------------------------------
# Kind-specific terms against their part-wide siblings
# --------------------------------------------------------------------------------------

# How far a kind-specific decode FLOOR may sit above the same part's part-wide floor.
#
# A floor is a launch-and-setup cost. Changing the attention kind changes what the setup
# reads -- an MLA kernel reads a latent cache, a windowed kernel a bounded block -- which
# moves the floor by a modest factor. It does not multiply it by six: a figure that large
# is the signature of a MODULE measurement, one that includes the projection GEMMs the
# catalog prices as separate nodes in the same layer.
#
# THE DEFECT THIS ENCODES. `attention_decode_floor_mla` shipped at 4.7x-6.2x its part's
# attention floor (51.5-89.5 us against 9.5-14.5 us), fitted cleanly from
# `mla_generation_module_perf.parquet` and labelled `method: measured, fitted: true`. The
# fit was sound and measured the wrong quantity: scripts/fit_attention_mla.py had already
# recorded that the module floor IS the projection weight read (293.6 MB at fp8, 61.2 us
# at H200's 4.80 TB/s, against a 44.0-63.8 us measured module floor), so the kernel charged
# it twice. End-to-end on the InferenceX corpus it cost 2.45 points of overall TPOT
# (14.98% -> 17.43%) and 8.2 points on kimi-k2.5 (6.38% -> 14.57%), flipping the kernel
# from beating AISimulate to losing to it. scripts/correct_mla_floor.py is the correction.
#
# 3.0 rather than something tighter: the measured windowed/full floor ratios span
# 0.77-1.00, so every legitimate kind-specific floor fitted so far sits at or BELOW its
# part-wide sibling. The bound only has to separate "a different kernel's setup" from "a
# different quantity entirely", and the rejected values were above 4.7.
MAX_KIND_FLOOR_RATIO = 3.0


def _by_hardware(prefix: str) -> dict[str, float]:
    """hardware name -> value, for every entry whose name is exactly `prefix`."""
    out: dict[str, float] = {}
    for _, name, body in ENTRIES:
        if name != prefix or "model" in (body.get("scope") or {}):
            continue
        for hw in (body.get("scope") or {}).get("hardware", []):
            out[hw] = body["value"]
    return out


def test_kind_specific_decode_floors_are_not_module_measurements():
    """A per-kind decode floor must stay near its part's attention-kernel floor.

    A floor many times larger is measuring a different quantity -- a module including its
    projections -- not a different kernel.
    """
    partwide = _by_hardware("attention_decode_floor")
    assert partwide, "no attention_decode_floor entries; this test would prove nothing"

    checked = 0
    for setname, name, body in ENTRIES:
        if not name.startswith("attention_decode_floor_"):
            continue
        for hw in (body.get("scope") or {}).get("hardware", []):
            base = partwide.get(hw)
            if base is None or base <= 0:
                continue
            ratio = body["value"] / base
            assert ratio <= MAX_KIND_FLOOR_RATIO, (
                f"{setname}: {name} on {hw} is {body['value']} us, {ratio:.2f}x this "
                f"part's attention_decode_floor of {base} us. A kind changes what the "
                f"setup reads, not its order of magnitude; a floor this large is a "
                f"MODULE measurement including the projection GEMMs the catalog prices "
                f"as separate nodes, so the kernel would charge them twice. See "
                f"scripts/correct_mla_floor.py."
            )
            checked += 1
    assert checked, "no kind-specific decode floor was checked against a part-wide floor"


# --------------------------------------------------------------------------------------
# Memory occupancy: activation multiples and CUDA-graph capture
# --------------------------------------------------------------------------------------

def _memory(name: str) -> float:
    for setname, n, body in ENTRIES:
        if setname == "cost-model-memory.yaml" and n == name:
            return body["value"]
    raise AssertionError(f"cost-model-memory.yaml carries no {name}")


# vLLM's CUDAGraphMode members (vllm/config/compilation.py). A consumer is handed one of
# these as a deployment's cudagraph_mode, so each must resolve to an entry: a missing one
# would read as "no capture cost" on exactly the deployment that pays it.
CUDAGRAPH_MODES = ("none", "piecewise", "full", "full_decode_only", "full_and_piecewise")


def test_every_cudagraph_mode_has_a_capture_entry():
    for mode in CUDAGRAPH_MODES:
        _memory(f"cudagraph_capture_bytes_{mode}")


def test_capture_bytes_respect_the_pool_overlay():
    """No mode may cost more than FULL_AND_PIECEWISE, and only NONE may cost nothing.

    vLLM captures FULL_AND_PIECEWISE as the PIECEWISE set plus the FULL decode set in one
    overlaid pool (gpu_model_runner.profile_cudagraph_memory), so every other capturing
    mode records a subset of its graphs and cannot exceed it for the same deployment.
    """
    bound = _memory("cudagraph_capture_bytes_full_and_piecewise")
    assert _memory("cudagraph_capture_bytes_none") == 0
    for mode in ("piecewise", "full", "full_decode_only"):
        v = _memory(f"cudagraph_capture_bytes_{mode}")
        assert 0 < v <= bound, (
            f"cudagraph_capture_bytes_{mode} is {v}, outside (0, {bound}]: a mode that "
            f"captures a subset of FULL_AND_PIECEWISE's graphs into the same pool cannot "
            f"cost more, and a capturing mode cannot cost nothing.")
    # A capture is a fraction of a GiB to a few GiB on every part the samples cover; a
    # value outside this is a unit slip (MiB read as GiB, or bytes as KiB).
    assert 64 * 2**20 <= bound <= 16 * 2**30


def test_activation_multiples_fall_with_width_and_moe_carries_more():
    """The live-buffer multiple shrinks as tensor parallelism shards the buffers, and an
    MoE layer holds more of them than a dense one at every width."""
    widths = (1, 2, 4, 8)
    for family in ("dense", "moe"):
        row = [_memory(f"activation_buffer_count_{family}_{n}rank") for n in widths]
        assert all(a >= b for a, b in zip(row, row[1:])), (
            f"activation_buffer_count_{family} rises with width: {row}")
        assert all(v >= 1 for v in row), f"activation_buffer_count_{family}: {row}"
    for n in widths:
        assert (_memory(f"activation_buffer_count_moe_{n}rank")
                >= _memory(f"activation_buffer_count_dense_{n}rank"))
    floor = _memory("activation_scratch_floor_bytes")
    assert 2**20 <= floor <= 2**30, f"activation_scratch_floor_bytes {floor} is not MiB-scale"


def test_capture_bytes_re_derive_from_the_committed_samples():
    """The capture half of emit_memory.py, re-run from the committed CSV.

    Needs no AISimulate tree, so unlike the writer gate it runs in CI: an edited CSV or a
    hand-edited value fails here.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "emit_memory", REPO / "scripts" / "emit_memory.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    text = (REPO / "coefficients" / "cost-model-memory.yaml").read_text()
    entries = mod.capture_entries()
    assert len(entries) == len(CUDAGRAPH_MODES)
    for e in entries:
        assert e in text, (
            "cost-model-memory.yaml no longer carries what emit_memory.py renders from "
            "docs/vllm-cudagraph-capture-samples.csv:\n" + e[:400])


def _capture_writer():
    import importlib.util
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "fit_deepseek_v41_capture", REPO / "scripts" / "fit_deepseek_v41_capture.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_the_v41_capture_entries_re_derive_from_their_vendored_start_ups():
    """`fit_deepseek_v41_capture.py --check`: the floor and per-token capture entries are what
    the vendored InferenceX H200 TP8 start-ups give."""
    import subprocess
    import sys

    r = subprocess.run([sys.executable, str(REPO / "scripts" / "fit_deepseek_v41_capture.py"),
                        "--check"], capture_output=True, text=True, cwd=str(REPO), timeout=120)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"


def test_the_v41_capture_form_reproduces_each_fitted_start_up():
    """With the committed values, the kernel's form (floor + mixed x ceiling + decode x the
    uniform-decode ceiling) gives each start-up's logged estimate within 0.15 GiB, and vLLM's
    own extrapolation, E - M, within the decode term's share: the split is the source's."""
    w = _capture_writer()
    floor, mixed, decode = (_memory(n) for n in w.NAMES)
    for s in json.loads(w.ANCHOR.read_text())["startups"]:
        c = s["max_cudagraph_capture_size"]
        d = w.decode_graphs(c, s["max_num_batched_tokens"], s["max_num_seqs"],
                            1 + s["num_spec_tokens"])[-1]
        assert abs((floor + mixed * c + decode * d) / 2**30 - s["estimated_gib"]) < 0.15, s
        assert abs(decode * d / 2**30 - (s["estimated_gib"] - s["profiling_took_gib"])) < 0.15, s


def test_emit_memory_keeps_the_capture_writers_entries():
    """emit_memory.py renders the whole set; re-running it must not drop the entries
    fit_deepseek_v41_capture.py owns, or the kernel silently falls back to the flat figure."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("emit_memory", REPO / "scripts" / "emit_memory.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    import coefficient_blocks as cb

    text = (REPO / "coefficients" / "cost-model-memory.yaml").read_text()
    stripped = cb.join([b for b in cb.split(text)
                        if not (cb.name_of(b) or "").startswith(mod.FOREIGN)])
    assert stripped != text
    assert mod.keep_foreign(stripped) == text


def test_the_h100_host_link_is_the_catalog_cpu_tier_fit_not_a_second_number():
    """host_link_bandwidth caps a CPU tier's transfer rate (kernel TierTime), and blis-catalog's
    cpu_dram is fitted to the same CollectiveX swap_blocks rows. The cap is that fit's larger
    direction, so it never binds below a fitted tier rate and the two cannot drift apart."""
    path = REPO.parent / "catalog" / "devices" / "storage.yaml"
    tier = yaml.safe_load(path.read_text())["cpu_dram"]
    assert _single("host_link_bandwidth", "h100") == max(
        tier["read_bandwidth_mb_s"], tier["write_bandwidth_mb_s"])


# --------------------------------------------------------------------------------------
# VAL-7: single-model coefficients only go down
# --------------------------------------------------------------------------------------

# A coefficient scoped to one model marks a cost the cost model's general forms do not yet
# explain. All 46 committed ones are for deepseek-v4.1-flash. Lower this number when a
# general form absorbs some of them; never raise it to admit a new one -- add the missing
# general form instead (design VAL-7).
MAX_SINGLE_MODEL_ENTRIES = 46


def single_model_entries(entries):
    return [(s, n, tuple(b["scope"]["model"])) for s, n, b in entries
            if len((b.get("scope") or {}).get("model") or []) == 1]


def test_single_model_coefficients_only_go_down():
    found = single_model_entries(ENTRIES)
    assert len(found) <= MAX_SINGLE_MODEL_ENTRIES, (
        f"{len(found)} coefficients are scoped to a single model, above the "
        f"{MAX_SINGLE_MODEL_ENTRIES} committed. A single-model coefficient marks a cost the "
        f"general forms do not explain; add the general form instead of another one.")
    assert found, "no single-model entries found; the filter is wrong"


def test_the_single_model_ratchet_counts_a_new_entry():
    """X18: the ratchet must count an entry added beyond the committed set."""
    extra = ENTRIES + [("s.yaml", "gemm_eps_max_bf16",
                        {"scope": {"hardware": ["h200"], "model": ["some-other-model"]}})]
    assert len(single_model_entries(extra)) == len(single_model_entries(ENTRIES)) + 1
    assert len(single_model_entries(ENTRIES)) == MAX_SINGLE_MODEL_ENTRIES, (
        "the committed count differs from the ratchet: lower MAX_SINGLE_MODEL_ENTRIES to match")
