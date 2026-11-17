"""The NCCL lane of the fp16 tensor-parallel all-reduce, read from the committed set alone.

The kernel prices a tensor-parallel all-reduce from the lane of the kernel the engine resolved
(kernel/collective_lane_test.go proves the selection). These tests check the data the NCCL lane
carries: that it is a complete, ordered triple per width, that it is slower than the custom
kernel's lane where both are measured (the direction the issue's acceptance test names), and that
the 2.30.7 comparison of inference-sim/blis-substrate#137 is on the entries it concerns. They
read only the committed file, so they run in CI with no AISimulate tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

SET = Path(__file__).resolve().parent.parent / "coefficients" / "cost-model-collectives.yaml"
NAMES = {}
for e in yaml.safe_load(SET.read_text(encoding="utf-8"))["coefficients"]:
    (name, body), = e.items()
    NAMES.setdefault(name, []).append(body)

LANE = re.compile(r"^collective_(floor|peak_rate|transition_rate)_all_reduce_nccl_fp16_(\d+)rank_(\w+)$")


def lane():
    out = {}
    for name, bodies in NAMES.items():
        m = LANE.match(name)
        if m:
            kind, w, part = m.groups()
            out.setdefault((part, int(w)), {})[kind] = bodies[0]
    return out


def test_every_width_of_the_lane_is_a_complete_triple_citing_a_version_column():
    got = lane()
    assert got, "no NCCL lane entries"
    for (part, w), triple in got.items():
        assert set(triple) == {"floor", "peak_rate", "transition_rate"}, (part, w, sorted(triple))
        for body in triple.values():
            assert body["method"] == "measured"
            assert "version column:" in body["sources"][0]["cite"], (part, w)
        assert triple["transition_rate"]["value"] <= triple["peak_rate"]["value"], (part, w)


def test_the_nccl_floor_is_above_the_custom_kernels_where_both_are_measured():
    checked = 0
    for (part, w), triple in lane().items():
        custom = NAMES.get(f"collective_floor_all_reduce_fp16_{w}rank_{part}")
        if not custom or custom[0]["method"] != "measured":
            continue
        # a100 and l40s have no NVLink custom lane worth the name here; the comparison the issue
        # asks for is on the parts where vLLM's custom kernel is measured.
        if custom[0]["sources"][0]["cite"].find("/comm/nccl/") >= 0:
            continue
        assert triple["floor"]["value"] > custom[0]["value"], (
            f"{part} {w} ranks: NCCL floor {triple['floor']['value']} us is not above the custom "
            f"kernel's {custom[0]['value']} us; one of the two lanes is from the wrong sweep")
        checked += 1
    assert checked >= 15, checked


def test_a_lane_floor_does_not_fall_as_the_group_widens():
    by_part = {}
    for (part, w), triple in lane().items():
        by_part.setdefault(part, {})[w] = triple["floor"]["value"]
    for part, widths in by_part.items():
        ws = sorted(widths)
        for a, b in zip(ws, ws[1:]):
            assert widths[b] >= widths[a], (part, a, b, widths)


def test_the_2_30_7_comparison_is_recorded_where_the_issue_reports_it():
    """#137: the >= 16 MiB latency ratio of the 2.30.7 table to the cited sweep."""
    expect = {("b200", 2): 1.376, ("b200", 4): 1.252, ("gb200_nvl72", 2): 1.739,
              ("gb200_nvl72", 4): 1.670, ("h100", 2): 1.009, ("h100", 4): 1.029,
              ("h200", 2): 1.032, ("h200", 4): 1.049}
    for (part, w), want in expect.items():
        text = lane()[(part, w)]["floor"]["rationale"]
        m = re.search(r"a median latency ([0-9.]+)x", text)
        assert m, f"{part} {w} ranks records no 2.30.7 comparison"
        assert abs(float(m.group(1)) - want) <= 0.05, (part, w, m.group(1), want)
