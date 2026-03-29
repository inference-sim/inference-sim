"""BLIS blackbox adapter — uses trained alpha/beta regression coefficients."""

from __future__ import annotations

import os
import subprocess
import tempfile

import yaml

from experiment.adapters.base import BaseBLISAdapter
from experiment.data_model import Experiment, SimulatorResult


class BLISBlackboxAdapter(BaseBLISAdapter):
    """BLIS simulator in blackbox mode (trained coefficients).

    ``can_run()`` returns True only when ``defaults.yaml`` contains trained
    coefficients matching the experiment's (model, GPU, TP) tuple.
    """

    def __init__(self, blis_binary: str, defaults_yaml: str | None = None):
        super().__init__(blis_binary)
        if defaults_yaml is None:
            defaults_yaml = os.path.join(os.path.dirname(blis_binary), "defaults.yaml")
        self._defaults_yaml = defaults_yaml

    @property
    def name(self) -> str:
        return "blis-blackbox"

    def can_run(self, experiment: Experiment) -> bool:
        """True if defaults.yaml has trained coefficients for (model, GPU, TP) tuple."""
        try:
            with open(self._defaults_yaml) as fh:
                data = yaml.safe_load(fh)
        except (FileNotFoundError, PermissionError, yaml.YAMLError):
            return False

        model_lower = experiment.model.lower()
        normalized_hw = self._normalize_hardware(experiment.hardware)

        for entry in (data or {}).get("models", []):
            if (
                entry.get("id", "").lower() == model_lower
                and entry.get("tensor_parallelism") == experiment.tp
                and entry.get("GPU") == normalized_hw
                and any(c != 0 for c in entry.get("alpha_coeffs", []))
            ):
                return True
        return False

    def run(self, experiment: Experiment) -> SimulatorResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = os.path.join(tmpdir, "workload_spec.yaml")
            self._write_workload_spec(experiment, spec_path)

            results_path = os.path.join(tmpdir, "results.json")
            args = self._build_common_args(experiment, spec_path, results_path)

            try:
                subprocess.run(args, capture_output=True, check=True, cwd=self._blis_dir)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"BLIS blackbox failed (rc={exc.returncode}) for "
                    f"{experiment.model}: {stderr}"
                ) from exc

            return self._parse_blis_results(results_path, experiment)
