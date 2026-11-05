"""The Marlin NVFP4 W4A16 grouped ramp is what its writer derives from InferenceX OperatorX.

Runs `fit_marlin_nvfp4_grouped_ramp.py --check --no-refit` against $OPERATORX_DATA, the
directory holding the OperatorX MoE response the entries cite (moe_MiniMax-M3.json); skipped
when that data is not present, since it is fetched, not vendored. `--no-refit` skips the
comparison candidate that needs the AISimulate tree; it does not change what is written.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATA = os.environ.get("OPERATORX_DATA", "")


@pytest.mark.skipif(not DATA or not (Path(DATA) / "moe_MiniMax-M3.json").is_file(),
                    reason="OperatorX MoE response not present ($OPERATORX_DATA)")
def test_marlin_nvfp4_grouped_ramp_reproduces_from_operatorx():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "fit_marlin_nvfp4_grouped_ramp.py"), "--check",
         "--no-refit"], capture_output=True, text=True, cwd=str(REPO), timeout=900,
        env=dict(os.environ, OPERATORX_DATA=DATA))
    assert r.returncode == 0, (
        "the committed gemm_*_w4a16_nvfp4_grouped entries are not what "
        f"fit_marlin_nvfp4_grouped_ramp.py derives.\n--- stdout ---\n{r.stdout}\n"
        f"--- stderr ---\n{r.stderr}")
