"""scripts/emit_host_overheads_scope.py keeps the part-wide host entries' scope equal to the catalog's parts (#242).

What is checked is the behaviour a new part depends on: it appears in every part-wide entry, a chip with its own
entry of a name is left to that entry, a model-scoped entry is untouched, and the committed file is already what
the script would write.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REG = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("emit_host_overheads_scope", REG / "scripts" / "emit_host_overheads_scope.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

TEXT = """kind: CoefficientSet
coefficients:
  - host_completion:
      value: 1.0
      scope: {hardware: [a, b]}
  - host_launch_per_kernel:
      value: 3.2
      scope: {hardware: [a, b, c]}
  - host_launch_per_kernel:
      value: 2.0
      scope: {hardware: [c]}
  - host_admission_per_token:
      value: 0.1
      scope: {hardware: [a], model: [m]}
"""


def test_a_new_part_reaches_every_part_wide_entry_and_a_chip_with_its_own_value_is_left_to_it():
    got = mod.render(TEXT, ["a", "b", "c", "d"])
    assert "host_completion:\n      value: 1.0\n      scope: {hardware: [a, b, c, d]}" in got
    assert "value: 3.2\n      scope: {hardware: [a, b, d]}" in got      # c has its own entry
    assert "value: 2.0\n      scope: {hardware: [c]}" in got           # the chip's own entry is not rewritten
    assert "scope: {hardware: [a], model: [m]}" in got                  # a model-scoped entry is not touched


def test_the_committed_file_is_what_the_script_writes():
    r = subprocess.run([sys.executable, str(REG / "scripts" / "emit_host_overheads_scope.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
