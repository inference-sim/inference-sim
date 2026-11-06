#!/usr/bin/env bash
# The re-derivation gates, with the data present: every fitted value must be what its
# writer derives from the pinned upstream data, and no test may skip except the by-design
# ramp-form ones (a part carries the one- or three-factor ramp, so the other gate does not
# apply) and the tests on the quarantine list (.github/quarantine.yaml, X25), each with an
# owning issue and an expiry. A skip is a gate that did not run, so any other skip fails
# (X16), as does an expired quarantine entry.
#
# Run from registry/ after `pip install --require-hashes -r scripts/requirements-fit.txt`,
# with the Go toolchain of contracts/toolchain on PATH (for tools/prcheck):
#
#   export AISIMULATE_DATA=<aisimulate>/python/aisimulate/src/aisimulate_core/systems/data
#   export BLIS_CATALOG=<the repository's catalog/ directory>
#   export AISIMULATE_ALLREDUCE_DATA=<aisimulate @7980e95>/.../systems/data  # relane_allreduce_vllm.py
#   python -I scripts/fetch_cited_inferencex.py <dir>
#   export COLLECTIVEX_DATA=<dir>                       # test_deepep_ll_graph.py
#   export OPERATORX_DATA=<dir of moe_<Model>.json>     # test_operatorx.py; quarantined, optional
#   export INFERENCEX_GOLDEN=<inferencex>/inferencex-e2e/infx/golden_al_distribution  # optional
#   scripts/rederive.sh
#
# The pinned commits (AISIMULATE_SHA, INFERENCEX_SHA) live in
# .github/workflows/rederive.yml, which runs this same script.
set -euo pipefail

: "${AISIMULATE_DATA:?set AISIMULATE_DATA to the AISimulate systems/data directory}"
: "${BLIS_CATALOG:?set BLIS_CATALOG to the catalog/ directory of this repository}"
: "${AISIMULATE_ALLREDUCE_DATA:?set AISIMULATE_ALLREDUCE_DATA to the systems/data of aisimulate at AISIMULATE_ALLREDUCE_SHA (rederive.yml), the h100/h200 vLLM 0.31.0 all-reduce sweep}"
: "${COLLECTIVEX_DATA:?set COLLECTIVEX_DATA to the directory scripts/fetch_cited_inferencex.py fills}"

n=$(find "$AISIMULATE_DATA" -name '*.parquet' | wc -l)
echo "AISimulate parquets present: $n"
test "$n" -gt 500

# With an InferenceX checkout, the vendored golden curves must be its curves, all of
# them, and the emitter must cite INFERENCEX_SHA (when given) and reproduce the set.
if [ -n "${INFERENCEX_GOLDEN:-}" ]; then
  vendored=validator/testdata/inferencex/golden_al_distribution
  diff <(cd "$vendored" && ls -1 *.yaml) <(cd "$INFERENCEX_GOLDEN" && ls -1 *.yaml)
  for f in "$vendored"/*.yaml; do diff "$f" "$INFERENCEX_GOLDEN/${f##*/}"; done
  if [ -n "${INFERENCEX_SHA:-}" ]; then
    grep -q "^INFERENCEX_SHA = \"$INFERENCEX_SHA\"$" scripts/emit_acceptance_profiles.py
  fi
  python scripts/emit_acceptance_profiles.py --check
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python -m pytest validator/ -q -rs --junitxml="$tmp/results.xml" -o junit_family=xunit2 | tee "$tmp/pytest.log"

# Every skip, with its full reason, from the JUnit report (the terminal truncates reasons):
# a test as "<file>::<test> SKIPPED (<reason>)", a module skipped at import as a summary
# line for its file, the two forms `prcheck skips` reads.
python -I - "$tmp/results.xml" > "$tmp/skips.txt" <<'PY'
import os
import sys
import xml.etree.ElementTree as ET


def node(dotted: str) -> tuple[str, list[str]]:
    """The test file and the classes within it that a JUnit classname names."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        if os.path.isfile("/".join(parts[:i]) + ".py"):
            return "/".join(parts[:i]) + ".py", parts[i:]
    raise SystemExit(f"no test file for JUnit classname {dotted!r}")


for tc in ET.parse(sys.argv[1]).iter("testcase"):
    s = tc.find("skipped")
    if s is None:
        continue
    reason = " ".join((s.get("message") or "").split())
    cls, name = tc.get("classname") or "", tc.get("name") or ""
    if cls:
        f, classes = node(cls)
        print(f"{f}::{'::'.join(classes + [name])} SKIPPED ({reason})")
    else:
        print(f"SKIPPED [1] {node(name)[0]}:0: {reason}")
PY
echo "skipped: $(wc -l < "$tmp/skips.txt")"
cat "$tmp/skips.txt"
# The by-design ramp-form skips are not gates that failed to run: on a part that carries one
# ramp form, the other form's gate has nothing to check.
grep -Ev 'SKIPPED \(.*carries the (one|three)-factor ramp' "$tmp/skips.txt" > "$tmp/gated.txt" \
  || test $? -eq 1
# Every other skip must be quarantined, with an owning issue and an unexpired date (X16, X25).
results=$tmp/gated.txt
(cd .. && go run ./tools/prcheck skips -results "$results" -python-package registry \
  -list .github/quarantine.yaml)
echo "rederive: every gate ran and passed, but for the quarantined tests listed above"
