"""Behavioural checks of draw_partition.py on a synthetic inventory (no InferenceX data)."""
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent


def run(inv):
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "inv.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inv))
    subprocess.run([sys.executable, "-I", str(HERE / "draw_partition.py"), "--inventory", str(d / "inv.jsonl"),
                    "--out", str(d / "p.json")], check=True, capture_output=True)
    return json.loads((d / "p.json").read_text())


def rec(i, model, gpu, concs, cfg=1, scen="single_turn", seed_run=None):
    return {"run_id": f"r{i}", "model_slug": model, "gpu": gpu, "engine": "vllm", "engine_image": "img",
            "precision": "fp8", "parallel": {"disagg": False, "prefill": {"tp": 8, "ep": 1}, "decode": {"tp": 8, "ep": 1}},
            "spec_method": "none", "offload": "off", "offload_backend": None, "scenario": scen, "isl": 1024,
            "osl": 1024, "recipe_fingerprint": None, "blocker": "ok", "config_id": cfg,
            "points": [{"point_id": f"{seed_run or i}-{c}-{i}", "conc": c} for c in concs]}


def base():
    inv = []
    for i, (m, g) in enumerate([(m, g) for m in ("a", "b", "c") for g in ("h100", "h200", "b200")]):
        inv.append(rec(i, m, g, [1, 2, 4, 8, 16, 32, 64]))
    inv.append(rec(100, "a", "h100", [1, 2, 4, 8, 16, 32, 64]))  # a repeat of run 0's configuration
    inv.append(rec(200, "d", "h100", [1, 2], cfg=2570, scen="agentic_traces"))  # an anchor
    return inv


def test_deterministic_and_repeats_share_a_set():
    p1, p2 = run(base()), run(base())
    assert p1["assignment"] == p2["assignment"]
    a = p1["assignment"]
    for c in (1, 2, 4, 8, 16, 32, 64):  # same configuration and level -> same set
        assert a[f"0-{c}-0"] == a[f"100-{c}-100"]


def test_anchor_held_and_every_level_in_both_sets():
    p = run(base())
    assert all(p["assignment"][f"200-{c}-200"] == "HELD" for c in (1, 2))
    for fac in ("model", "gpu", "conc_bin"):
        for level, x in p["per_factor_coverage"][fac]["per_level"].items():
            if x["groups"] >= 2 and level not in ("d",):  # model d is anchors only: forced HELD
                assert x["FIT"] and x["HELD"], (fac, level, x)


def test_adding_runs_does_not_move_existing_unrepaired_points():
    p1 = run(base())
    p2 = run(base() + [rec(300, "e", "b300", [1, 2, 4, 8])])
    moved = [k for k, v in p1["assignment"].items() if p2["assignment"][k] != v]
    repaired = {m["moved"] for m in p2["repair_log"]} | {m["moved"] for m in p1["repair_log"]}
    groups = {g["group_key"]: g["points"] for g in p2["groups"]}
    allowed = {pid for gk in repaired for pid in groups.get(gk, [])}
    assert set(moved) <= allowed, moved


def test_forced_held_beats_the_hash():
    """With 40 anchor groups, about 16 would hash to FIT; every one must be HELD."""
    inv = [rec(500, "x", "h100", list(range(1, 41)), cfg=2560, scen="agentic_traces")]
    p = run(base() + inv)
    assert all(p["assignment"][f"500-{c}-500"] == "HELD" for c in range(1, 41))


def test_repair_puts_two_group_levels_in_both_sets():
    """Thirty models with exactly two groups each: by chance about 1 in 3 would land wholly in
    one set; repair must leave every one in both."""
    inv = [rec(1000 + i, f"m{i}", "h200", [4, 8]) for i in range(30)]
    p = run(base() + inv)
    for i in range(30):
        x = p["per_factor_coverage"]["model"]["per_level"][f"m{i}"]
        assert x["FIT"] and x["HELD"], (i, x)


def test_freeze_keeps_every_existing_assignment():
    import tempfile as tf
    d = pathlib.Path(tf.mkdtemp())
    inv1 = base()
    (d / "inv1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inv1))
    subprocess.run([sys.executable, "-I", str(HERE / "draw_partition.py"), "--inventory", str(d / "inv1.jsonl"),
                    "--out", str(d / "p1.json")], check=True, capture_output=True)
    inv2 = inv1 + [rec(i, f"n{i}", "h200", [4, 8]) for i in range(400, 420)] + [rec(101, "a", "h100", [1, 2])]
    (d / "inv2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inv2))
    subprocess.run([sys.executable, "-I", str(HERE / "draw_partition.py"), "--inventory", str(d / "inv2.jsonl"),
                    "--out", str(d / "p2.json"), "--freeze", str(d / "p1.json")], check=True, capture_output=True)
    p1, p2 = json.loads((d / "p1.json").read_text()), json.loads((d / "p2.json").read_text())
    # A frozen assignment the hash would not reproduce must still be kept: flip every point of p1.
    flipped = {k: ("FIT" if v == "HELD" else "HELD") for k, v in p1["assignment"].items()}
    flipped.update({k: "HELD" for k in p1["assignment"] if k.startswith("200-")})
    (d / "p1f.json").write_text(json.dumps({"assignment": flipped}))
    subprocess.run([sys.executable, "-I", str(HERE / "draw_partition.py"), "--inventory", str(d / "inv2.jsonl"),
                    "--out", str(d / "p3.json"), "--freeze", str(d / "p1f.json")], check=True, capture_output=True)
    p3 = json.loads((d / "p3.json").read_text())
    assert all(p3["assignment"][k] == v for k, v in flipped.items())
    assert all(p2["assignment"][k] == v for k, v in p1["assignment"].items())
    assert p2["assignment"]["101-1-101"] == p1["assignment"]["0-1-0"]  # a new repeat joins its group
