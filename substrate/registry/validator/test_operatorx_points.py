"""The digest the Marlin NVFP4 entries cite covers the OperatorX rows the fit reads, and nothing else.

The entries once cited the sha256 of the response's bytes. The URL then served a different
response with the same rows, and the re-derivation test could not run for a change that moved no
value. These tests pin both halves of the replacement: a change
the fit does not read leaves the digest alone, and a change to a row it reads moves it.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import operatorx_points as op  # noqa: E402


def _key(tokens, hidden=6144, inter=3072, top_k=4, num=128, w="e2m1", sc="e4m3", grp=16, parallel=None):
    return json.dumps({
        "tokens": tokens, "hidden": hidden, "parallel": parallel,
        "experts": {"inter": inter, "top_k": top_k, "num": num,
                    "w1": {"dtype": w, "scale": {"dtype": sc, "group": [1, grp]}}},
        "shared": {"count": 1, "inter": 3072, "w1": {"dtype": "e4m3", "scale": {"dtype": "ue8m0"}}},
    })


def _body():
    toks = [1, 2, 4]
    meas = {}
    for chip in op.CHIPS:
        meas[chip] = {"status": ["ok"] * 3, "kernel": [0] * 3, "runId": [op.RUNS[chip]] * 3,
                      "cudaGraph": [1, 1, 0], "latencyUs": [10.0, 11.0, 12.0]}
    return {"cases": [{"key": _key(t)} for t in toks], "kernels": ["MarlinExperts", "Other"],
            "measurements": meas, "workloads": ["anything"]}


BODY = _body()


def test_digest_is_a_sha256():
    assert len(op.digest(BODY)) == 64


def test_cases_in_another_order_leave_the_digest_alone():
    d = op.digest(BODY)
    b = copy.deepcopy(BODY)
    for k in ("cases",):
        b[k].reverse()
    for chip in op.CHIPS:
        for f in b["measurements"][chip].values():
            f.reverse()
    assert op.digest(b) == d


def test_what_the_fit_does_not_read_leaves_the_digest_alone():
    d = op.digest(BODY)
    b = copy.deepcopy(BODY)
    b["workloads"].append("new")
    b["kernels"].append("Third")
    b["measurements"]["h100"]["status"].append("ok")
    b["cases"].append({"key": _key(8, w="e4m3")})            # another dtype: not kept
    b["measurements"]["h100"]["kernel"].append(1)
    b["measurements"]["h100"]["runId"].append("9")
    b["measurements"]["h100"]["cudaGraph"].append(0)
    b["measurements"]["h100"]["latencyUs"].append(99.0)
    b["measurements"]["h200"]["status"].append("ok")
    b["measurements"]["h200"]["kernel"].append(1)
    b["measurements"]["h200"]["runId"].append("9")
    b["measurements"]["h200"]["cudaGraph"].append(0)
    b["measurements"]["h200"]["latencyUs"].append(99.0)
    assert op.digest(b) == d


@pytest.mark.parametrize("edit", [
    lambda b: b["measurements"]["h100"]["latencyUs"].__setitem__(0, 10.01),
    lambda b: b["measurements"]["h200"]["latencyUs"].__setitem__(2, 12.5),
    lambda b: b["measurements"]["h100"]["cudaGraph"].__setitem__(0, 0),
    lambda b: b["cases"].__setitem__(0, {"key": _key(3)}),
    lambda b: b["measurements"]["h100"]["status"].__setitem__(1, "failed"),
    lambda b: b["measurements"]["h200"]["kernel"].__setitem__(1, 1),
], ids=["latency-h100", "latency-h200", "graph", "tokens", "status", "kernel"])
def test_a_change_to_a_row_the_fit_reads_moves_the_digest(edit):
    b = copy.deepcopy(BODY)
    edit(b)
    assert op.digest(b) != op.digest(BODY)


@pytest.mark.parametrize("key", [_key(1, hidden=7168), _key(1, inter=2048), _key(1, top_k=8), _key(1, num=64)],
                         ids=["hidden", "inter", "top_k", "experts"])
def test_a_moved_geometry_is_an_error_not_a_skip(key):
    b = copy.deepcopy(BODY)
    b["cases"][0] = {"key": key}
    with pytest.raises(SystemExit):
        op.digest(b)


def test_a_row_of_another_run_is_not_one_the_fit_reads():
    b = copy.deepcopy(BODY)
    b["measurements"]["h100"]["runId"][1] = "1"
    assert op.digest(b) != op.digest(BODY)
    parallel = copy.deepcopy(BODY)
    parallel["cases"][0] = {"key": _key(1, parallel={"tp": 2})}
    assert op.digest(parallel) != op.digest(BODY)
