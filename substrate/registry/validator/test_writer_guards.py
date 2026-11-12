"""The writers' own refusals and the block rewriter they share, tested without the sweeps.

The re-derivation gates (test_writers_are_idempotent.py) show that a writer reproduces the
committed file today. They cannot show that a writer REFUSES when its premises break,
because at the pinned data every premise holds. These tests feed stub inputs to the
guards that keep a fit honest, so a guard that cannot fire -- the geometry check in
fit_deepseek_v41_h200.py once could not, for the very case it exists for -- fails here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import coefficient_blocks as cb  # noqa: E402


def entry(name: str, scope: str, value: str = "1.0", rationale: str = "text") -> str:
    return (f"  - {name}:\n      value: {value}\n      units: dimensionless\n"
            f"      method: measured\n      fitted: true\n      scope: {scope}\n"
            f"      rationale: >\n        {rationale}")


# --------------------------------------------------------------------------------------
# coefficient_blocks
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted((REPO / "coefficients").glob("*.yaml")),
                         ids=lambda p: p.name)
def test_split_and_join_round_trip_every_committed_set(path):
    text = path.read_text()
    assert cb.join(cb.split(text)) == text


def test_a_paragraph_break_stays_inside_its_entry():
    """A blank line in a folded rationale must not end the entry: replacing it would leave
    the second paragraph behind, glued under the new rationale, and --check would accept
    it from then on."""
    text = "\n".join([
        "coefficients:",
        entry("a", "{hardware: [h200]}", rationale="para one"),
        "",
        "        para two OLD",
        "  - b:",
        "      value: 2.0",
    ])
    blocks = cb.split(text)
    a = next(b for b in blocks if cb.name_of(b) == "a")
    assert "        para two OLD" in a
    cb.upsert(blocks, entry("a", "{hardware: [h200]}", rationale="fresh").split("\n"),
              after=lambda b: False)
    assert "OLD" not in cb.join(blocks)


def test_a_blank_line_before_the_next_section_is_not_absorbed():
    text = "\n".join([entry("a", "{hardware: [h200]}"), "", "  # --- next ---",
                      entry("b", "{hardware: [h200]}")])
    blocks = cb.split(text)
    a = next(b for b in blocks if cb.name_of(b) == "a")
    assert a[-1].strip() == "text"


def test_a_reordered_scope_is_the_same_entry_not_a_new_one():
    text = "\n".join(["coefficients:", entry("a", "{model: [m], hardware: [h200]}")])
    blocks = cb.split(text)
    cb.upsert(blocks, entry("a", "{hardware: [h200], model: [m]}", value="2.0").split("\n"),
              after=lambda b: True)
    joined = cb.join(blocks)
    assert joined.count("  - a:") == 1 and "value: 2.0" in joined


def test_two_entries_with_one_name_and_scope_refuse():
    text = "\n".join([entry("a", "{hardware: [h200]}"), entry("a", "{hardware: [h200]}")])
    with pytest.raises(SystemExit):
        cb.upsert(cb.split(text), entry("a", "{hardware: [h200]}").split("\n"),
                  after=lambda b: True)


def test_entries_of_different_methods_in_one_scope_are_two_entries_each_kept_in_place():
    """An `assumed` bound and the `transferred` entry that displaces it share a name and a
    scope (design VAL-4): a writer that owns one rewrites it and leaves the other byte for byte."""
    def with_method(m, value):
        return entry("a", "{hardware: [h200]}", value=value).replace("method: measured", f"method: {m}")
    text = "\n".join([with_method("assumed", "1.0"), with_method("transferred", "0.25")])
    blocks = cb.split(text)
    cb.upsert(blocks, with_method("assumed", "0.9").split("\n"), after=lambda b: False)
    joined = cb.join(blocks)
    assert "value: 0.9" in joined and "value: 0.25" in joined and "value: 1.0" not in joined
    assert joined.count("  - a:") == 2


def test_a_new_entry_with_no_anchor_refuses():
    blocks = cb.split(entry("a", "{hardware: [h200]}"))
    with pytest.raises(SystemExit):
        cb.upsert(blocks, entry("b", "{hardware: [h100]}").split("\n"),
                  after=lambda b: False)


def test_a_new_entry_lands_after_the_last_anchor():
    text = "\n".join([entry("g1", "{hardware: [h200]}"), entry("g2", "{hardware: [h200]}"),
                      entry("x", "{hardware: [h200]}")])
    blocks = cb.split(text)
    cb.upsert(blocks, entry("g3", "{hardware: [h200]}").split("\n"),
              after=lambda b: (cb.name_of(b) or "").startswith("g"))
    assert [cb.name_of(b) for b in blocks if cb.name_of(b)] == ["g1", "g2", "g3", "x"]


# --------------------------------------------------------------------------------------
# fit_deepseek_v41_h200.check_geometry
# --------------------------------------------------------------------------------------

V41 = {"num_attention_heads": 64, "head_dim": 512, "qk_rope_head_dim": 64,
       "sliding_window": 128, "compress_ratios": [0, 1, 2], "index_n_heads": 32,
       "index_head_dim": 128, "index_topk": 512}
FLASH = {"num_attention_heads": 64, "head_dim": 512, "qk_rope_head_dim": 64,
         "sliding_window": 128, "index_n_heads": 64, "index_head_dim": 128}
PRO = dict(FLASH, num_attention_heads=128)


def geometry(tmp_path, v41=None, flash=None, pro=None):
    pytest.importorskip("pyarrow")
    import fit_deepseek_v41_h200 as w

    configs = tmp_path / "model_configs"
    configs.mkdir()
    for name, cfg in (("deepseek-ai--DeepSeek-V4.1-Flash", v41 or V41),
                      ("sgl-project--DeepSeek-V4-Flash-FP8", flash or FLASH),
                      ("sgl-project--DeepSeek-V4-Pro-FP8", pro or PRO)):
        (configs / f"{name}_config.json").write_text(json.dumps({"text_config": cfg}))
    data = tmp_path / "systems" / "data"
    data.mkdir(parents=True)
    return w.check_geometry(data)


def test_the_committed_geometry_passes_and_derives_the_v41_read(tmp_path):
    g = geometry(tmp_path)
    assert (g["record"], g["key"], g["v41_blocks"], g["v41_tokens"]) == (584, 132, 10, 640)


@pytest.mark.parametrize("change", [
    ({"num_attention_heads": 128}, None),   # V4.1 on h128, fitted on h64 rows
    ({"head_dim": 576}, None),              # a different KV record
    ({"sliding_window": 256}, None),        # a different window than the fitted rows
    ({"compress_ratios": [0, 4]}, None),    # a geometry this fit does not describe
    ({"index_head_dim": 64}, None),         # a different indexer key
    (None, {"num_attention_heads": 128}),   # the fit rows move to h128
], ids=["v41-h128", "record", "window", "ratio", "key", "fit-rows-h128"])
def test_check_geometry_refuses_a_kernel_v41_does_not_share(tmp_path, change):
    v41_change, flash_change = change
    with pytest.raises(SystemExit):
        geometry(tmp_path, v41=dict(V41, **(v41_change or {})),
                 flash=dict(FLASH, **(flash_change or {})))


def test_check_geometry_refuses_two_indexer_configurations(tmp_path):
    with pytest.raises(SystemExit):
        geometry(tmp_path, pro=dict(PRO, index_n_heads=32))


def test_a_fit_at_its_grid_edge_refuses():
    pytest.importorskip("pyarrow")
    import fit_deepseek_v41_h200 as w

    grid = [1, 2, 3]
    w.interior(2, grid, "x")
    for edge in (1, 3, float("nan")):
        with pytest.raises(SystemExit):
            w.interior(edge, grid, "x")


# --------------------------------------------------------------------------------------
# fit_deepseek_v41_host.kernel_basis
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("tamper", ["anchor", "device", "count", "scenario", "step",
                                    "prefill_count", "dirty", "no_prefill_row"])
def test_the_v41_eager_basis_refuses_a_record_that_does_not_match(tmp_path, monkeypatch,
                                                                   tamper):
    """The eager constant divides the anchor by the KERNEL's device time and launch count;
    a basis recorded for another anchor, or edited by hand, must refuse rather than price."""
    import fit_deepseek_v41_host as w

    anchor = json.loads(w.ANCHOR_JSON.read_text())
    kb = json.loads(w.KERNEL_BASIS_JSON.read_text())
    if tamper == "anchor":
        kb["anchor_sha256"] = "0" * 64
    elif tamper == "device":
        kb["device_ms"] += 0.5
    elif tamper == "count":
        kb["kernel_count"] += 46
    elif tamper == "step":
        kb["step"] = "prefill"
    elif tamper == "prefill_count":
        # Review H2: a count that includes the 46 phase: prefill launches, with a host time
        # made consistent with it, is the known error and must still refuse.
        kb["kernel_count"] += 46
        kb["per_resource_ms"]["host"] += 46 * 3.2 / 1e3
        kb["prefill_row_host_ms"] += 46 * 3.2 / 1e3
    elif tamper == "dirty":
        kb["kernel_commit"] += "-dirty"
    elif tamper == "no_prefill_row":
        del kb["prefill_row_host_ms"]
    else:
        kb["scenario"] = "TP8-baseline"
    path = tmp_path / "basis.json"
    path.write_text(json.dumps(kb))
    monkeypatch.setattr(w, "KERNEL_BASIS_JSON", path)
    with pytest.raises(SystemExit):
        w.kernel_basis(anchor, 3.2)


@pytest.mark.parametrize("tamper", ["median", "levels", "sha"])
def test_the_v41_eager_anchor_inputs_refuse_an_inconsistent_record(tamper):
    """Only derived inputs of the enforce-eager anchor are vendored; without the source
    report (V41_EAGER_ANCHOR_REPORT) --check still refuses inputs that contradict themselves."""
    import fit_deepseek_v41_host as w

    anchor = json.loads(w.ANCHOR_JSON.read_text())
    if tamper == "median":
        anchor["plateau"]["median_itl_ms"] = anchor["plateau"]["max_itl_ms"] + 1
    elif tamper == "levels":
        anchor["plateau"]["levels"] = anchor["plateau"]["levels"][:-1]
    else:
        anchor["sha256"] = "not-a-sha"
    with pytest.raises(SystemExit):
        w.verify_anchor(anchor, {})


@pytest.mark.parametrize("composition", [None, "launch_bound: step = launches x (eager + "
                                                "dispatch) >= device, device hidden"])
def test_the_v41_eager_constant_refuses_a_non_additive_basis(tmp_path, monkeypatch,
                                                              composition):
    """The constant is backed out under the kernel's ADDITIVE eager composition; a basis
    recorded under another (the rejected launch-overlap form) or none must refuse."""
    import fit_deepseek_v41_host as w

    anchor = json.loads(w.ANCHOR_JSON.read_text())
    kb = json.loads(w.KERNEL_BASIS_JSON.read_text())
    if composition is None:
        del kb["composition"]
    else:
        kb["composition"] = composition
    path = tmp_path / "basis.json"
    path.write_text(json.dumps(kb))
    monkeypatch.setattr(w, "KERNEL_BASIS_JSON", path)
    with pytest.raises(SystemExit):
        w.eager(anchor)


def test_the_v41_eager_constant_reproduces_the_anchor_additively():
    """launches x (eager + dispatch) + device is the anchor's ITL, as the kernel composes it."""
    import fit_deepseek_v41_host as w

    ea = w.eager(json.loads(w.ANCHOR_JSON.read_text()))
    step = ea["n"] * (ea["e"] + ea["launch"]) / 1e3 + ea["dev"]
    assert abs(step / ea["median"] - 1) < 1e-4
    assert 92.75 <= ea["e"] <= 93.7  # the kernel's TestTheEagerDecodeAnchorIsAdditive band


# --------------------------------------------------------------------------------------
# fit_attention: the bandwidth fraction (SUB-5, blis-substrate #16)
# --------------------------------------------------------------------------------------

def test_the_attention_fit_never_offers_a_fraction_above_the_datasheet_peak():
    """relane_attention_decode.py writes fraction * peak unchanged, so a grid that reaches
    2.0 would let a wrong byte count commit a rate above the part's HBM peak."""
    import fit_attention

    assert max(fit_attention.FRACTION_GRID) == 1.0
    assert min(fit_attention.FRACTION_GRID) > 0.0
    for f in fit_attention.FRACTION_GRID:
        assert fit_attention.check_fraction(f) == f


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.02, 2.0, float("inf"), float("nan")])
def test_the_attention_fit_refuses_a_fraction_outside_zero_to_one(bad):
    import fit_attention

    with pytest.raises(ValueError, match=r"outside \(0, 1\.0\]"):
        fit_attention.check_fraction(bad)


# --------------------------------------------------------------------------------------
# fit_deepseek_v41_host.rust_ttft_bound: a one-sided check on the Rust-frontend rate
# --------------------------------------------------------------------------------------

def _bound_row(ttft, isl, split="validation"):
    q = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
    return {"point_id": "x", "split": split, "gpus": 8, "concurrency": 1,
            "ttft_ms": dict(zip(q, ttft)), "isl_tokens": dict(zip(q, isl))}


def test_the_rust_rate_bound_is_the_tightest_ttft_over_prompt_pair_of_percentiles():
    """Nine tenths of the requests answer within 500 ms (p90) and a quarter have prompts of at
    least 1,000,000 tokens (p75 of prompt): at least 15% have both, so the rate is at most 0.5
    us/token. A looser pair (p95 of TTFT, 600 ms, over the same prompt) must not be chosen."""
    import fit_deepseek_v41_host as w

    ttft = [100, 150, 200, 250, 300, 400, 500, 600, 9000]
    isl = [10, 20, 30, 40, 50, 1_000_000, 1_100_000, 1_200_000, 1_300_000]
    b = w.rust_ttft_bound({"rows": [_bound_row(ttft, isl)]})
    assert b["a"] == "p90" and b["b"] == "p75"  # 500 ms over 1,000,000 tokens
    assert abs(b["rate"] - 0.5) < 1e-9
    assert b["share"] == 15


def test_the_rust_rate_bound_refuses_a_row_that_is_not_a_development_point():
    import fit_deepseek_v41_host as w

    row = _bound_row([1] * 9, [1] * 9, split="test")
    with pytest.raises(SystemExit):
        w.rust_ttft_bound({"rows": [row]})


def test_the_vendored_rows_bound_the_rust_rate_above_the_registry_value_and_below_the_python_rate():
    """The registry's Rust rate must sit under the bound the H200 rows give, and the
    Python-path rate must not: that is what the rationale says the rows show."""
    import fit_deepseek_v41_host as w

    ad = w.admission(json.loads(w.BENCH_JSON.read_text()))
    bound = w.rust_ttft_bound(json.loads(w.RUST_BOUND_JSON.read_text()))["rate"]
    assert ad["value"] / w.FASTOKENS_SPEEDUP < bound < ad["value"]
