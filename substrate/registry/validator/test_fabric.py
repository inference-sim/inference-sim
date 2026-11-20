"""Every inter-node fabric the catalog defines carries exactly one bandwidth correction.

blis-latency-kernel multiplies a fabric's nominal InterNodeBwGBps by
`fabric_bandwidth_efficiency` (fabric.go), and refuses a value outside (0, 1]. A fabric with
no entry is silently priced at its datasheet line rate, which is the error the term exists
to correct; two entries for one fabric leave the kernel's choice to file order. So the set
must cover the vendored catalog's networks/ exactly once each, and `emit_fabric.py --check`
must reproduce it.
"""

import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
NETWORKS = REPO.parent / "catalog" / "networks"
NAME = "fabric_bandwidth_efficiency"


def _entries():
    out = []
    for path in sorted((REPO / "coefficients").glob("*.yaml")):
        for e in yaml.safe_load(path.read_text())["coefficients"]:
            (name, body), = e.items()
            if name == NAME:
                out.append((path.name, body))
    return out


def test_every_catalog_fabric_has_exactly_one_efficiency_in_the_unit_interval():
    fabrics = sorted(p.stem for p in NETWORKS.glob("*.yaml"))
    assert fabrics, f"no vendored fabrics under {NETWORKS}"
    seen: dict[str, list[float]] = {f: [] for f in fabrics}
    for setname, body in _entries():
        scope = body["scope"]
        assert set(scope) == {"fabric"}, (
            f"{setname}: {NAME} must be scoped by fabric alone, got {scope}; a hardware or "
            f"model scope would apply one fabric's correction to another")
        for f in scope["fabric"]:
            assert f in seen, f"{setname}: {NAME} names fabric {f!r}, not in the catalog"
            seen[f].append(body["value"])
    for f, values in seen.items():
        assert len(values) == 1, f"fabric {f} has {len(values)} {NAME} entries: {values}"
        assert 0 < values[0] <= 1, f"fabric {f}: {NAME} {values[0]} is outside (0, 1]"


def test_fabric_set_is_what_its_writer_emits():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "emit_fabric.py"), "--check"],
                       capture_output=True, text=True, cwd=str(REPO), timeout=120)
    assert r.returncode == 0, (
        "coefficients/cost-model-fabric.yaml is not what scripts/emit_fabric.py writes.\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
