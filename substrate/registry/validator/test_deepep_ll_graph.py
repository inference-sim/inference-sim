"""The CollectiveX-fitted entries are what their writers derive from CollectiveX.

Runs `fit_deepep_ll_graph.py`, `fit_host_link.py` and `fit_deepep_collectivex.py --check`
against $COLLECTIVEX_DATA, the directory of InferenceX
CollectiveX run bodies the entries cite; skipped when that data is not present, since it is
fetched, not vendored.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATA = os.environ.get("COLLECTIVEX_DATA", "")

# A writer exits 3 when the bodies its committed entries cite are not in $COLLECTIVEX_DATA:
# CollectiveX serves only its latest 15 runs, so a run cited once can stop being fetchable.
# That is a gate that cannot run, not a pass, so it skips under this exact reason, which
# scripts/rederive.sh names as the one data skip it accepts.
CITED_ABSENT = "the cited CollectiveX bodies are not present"


def _check(script: str) -> subprocess.CompletedProcess:
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), "--check"],
        capture_output=True, text=True, cwd=str(REPO), timeout=600,
        env=dict(os.environ, COLLECTIVEX_DATA=DATA))
    if r.returncode == 3:
        pytest.skip(f"{script}: {CITED_ABSENT}: {r.stderr.strip()}")
    return r


@pytest.mark.skipif(not DATA or not any(Path(DATA).glob("cx_run_*.body")),
                    reason="CollectiveX run bodies not present ($COLLECTIVEX_DATA)")
def test_graph_mode_pair_reproduces_from_collectivex():
    r = _check("fit_deepep_ll_graph.py")
    assert r.returncode == 0, (
        "the committed moe_a2a_*_deepep_low_latency_graph entries are not what "
        f"fit_deepep_ll_graph.py derives.\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")


@pytest.mark.skipif(not DATA or not any(Path(DATA).glob("cx_run_*.body")),
                    reason="CollectiveX run bodies not present ($COLLECTIVEX_DATA)")
def test_host_link_reproduces_from_collectivex():
    r = _check("fit_host_link.py")
    assert r.returncode == 0, (
        "the committed host_link_bandwidth entries are not what fit_host_link.py derives.\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")


@pytest.mark.skipif(not DATA or not any(Path(DATA).glob("cx_run_*.body")),
                    reason="CollectiveX run bodies not present ($COLLECTIVEX_DATA)")
def test_gb200_and_h200_ep16_deepep_reproduce_from_collectivex():
    r = _check("fit_deepep_collectivex.py")
    assert r.returncode == 0, (
        "the committed GB200 / H200-EP16 moe_a2a_* entries are not what "
        f"fit_deepep_collectivex.py derives.\n--- stdout ---\n{r.stdout}\n"
        f"--- stderr ---\n{r.stderr}")
