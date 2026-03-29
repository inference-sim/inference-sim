"""BLIS trained-roofline adapter — roofline basis functions with learned corrections."""

from __future__ import annotations

import os
import subprocess
import tempfile

from experiment.adapters.base import BaseBLISAdapter
from experiment.data_model import Experiment, SimulatorResult


class BLISTrainedRooflineAdapter(BaseBLISAdapter):
    """BLIS simulator with ``--latency-model trained-roofline``.

    Uses globally-fitted roofline correction coefficients from
    ``defaults.yaml`` (loaded by the BLIS binary automatically).
    Works for any model (no per-model profiling required).
    """

    @property
    def name(self) -> str:
        return "blis-trained-roofline"

    def run(self, experiment: Experiment) -> SimulatorResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = os.path.join(tmpdir, "workload_spec.yaml")
            self._write_workload_spec(experiment, spec_path)

            results_path = os.path.join(tmpdir, "results.json")
            args = self._build_common_args(experiment, spec_path, results_path)
            args.extend(["--latency-model", "trained-roofline"])

            try:
                subprocess.run(args, capture_output=True, check=True, cwd=self._blis_dir)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"BLIS trained-roofline failed (rc={exc.returncode}) for "
                    f"{experiment.model}: {stderr}"
                ) from exc

            return self._parse_blis_results(results_path, experiment)
