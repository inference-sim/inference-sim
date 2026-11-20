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


CPUS = {"a": "x86", "b": "x86", "c": "grace", "d": "grace"}
IDS = {"host": {"vllm": "1" * 64, "sglang": "2" * 64}, "graphs": {"vllm": "3" * 64, "sglang": "3" * 64}}


def test_a_new_part_reaches_every_part_wide_entry_and_a_chip_with_its_own_value_is_left_to_it():
    got = mod.render(TEXT, ["a", "b", "c", "d"], CPUS, IDS)
    assert "host_completion:\n      value: 1.0\n      scope: {hardware: [a, b, c, d], host_cpu: [x86, grace]}" in got
    assert "value: 3.2\n      scope: {hardware: [a, b, d], host_cpu: [x86, grace]}" in got   # c has its own entry
    assert "value: 2.0\n      scope: {hardware: [c], host_cpu: [grace]}" in got              # the chip's own entry keeps its chip
    assert "scope: {hardware: [a], model: [m], host_cpu: [x86]}" in got                       # a model-scoped part list is not touched


def test_each_entry_is_keyed_by_the_cpus_of_its_chips_and_the_code_ids_of_its_path():
    got = mod.render(TEXT, ["a", "b", "c", "d"], CPUS, IDS)
    host = "      code_ids: [" + "1" * 64 + ", " + "2" * 64 + "]"
    graphs = "      code_ids: [" + "3" * 64 + "]"          # two engines that state one id list it once
    assert "host_completion:\n      value: 1.0\n      scope: {hardware: [a, b, c, d], host_cpu: [x86, grace]}\n" + host in got
    assert "host_admission_per_token:\n      value: 0.1\n      scope: {hardware: [a], model: [m], host_cpu: [x86]}\n" + host in got
    assert "host_launch_per_kernel:\n      value: 2.0\n      scope: {hardware: [c], host_cpu: [grace]}\n" + graphs in got


def test_keying_is_idempotent_and_follows_a_changed_code_id():
    once = mod.render(TEXT, ["a", "b", "c", "d"], CPUS, IDS)
    assert mod.render(once, ["a", "b", "c", "d"], CPUS, IDS) == once
    newer = {**IDS, "host": {"vllm": "4" * 64, "sglang": "2" * 64}}
    again = mod.render(once, ["a", "b", "c", "d"], CPUS, newer)
    assert "1" * 64 not in again and again.count("4" * 64) == 2    # replaced, not appended; once per host entry
    assert again.count("code_ids:") == once.count("code_ids:")


def test_a_chip_the_catalog_does_not_hold_or_a_name_bound_to_no_path_is_refused():
    import pytest
    with pytest.raises(SystemExit):
        mod.render(TEXT, ["a", "b"], {"a": "x86"}, IDS)
    with pytest.raises(SystemExit):
        mod.render(TEXT.replace("host_completion", "gemm_eps_max_bf16"), ["a", "b", "c", "d"], CPUS, IDS)


def test_the_committed_file_is_what_the_script_writes():
    r = subprocess.run([sys.executable, str(REG / "scripts" / "emit_host_overheads_scope.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
