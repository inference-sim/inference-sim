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
#   export AISIM_GLM53_DATA=<aisim @AISIM_GLM53_SHA>/python/aisim/src/aisim_core/systems/data  # fit_glm53_attention.py, fit_glm53_fp8_prefill_scale.py
#   python -I scripts/fetch_cited_inferencex.py <dir>
#   export COLLECTIVEX_DATA=<dir>                       # test_deepep_ll_graph.py
#   export OPERATORX_DATA=<dir>                         # test_operatorx.py; the same dir: the fetcher
#                                                       # writes moe_MiniMax-M3.json there too
#   export INFERENCEX_GOLDEN=<inferencex>/inferencex-e2e/infx/golden_al_distribution  # optional
#   scripts/rederive.sh
#
# REDERIVE_JOBS=N (default 1) runs the tests in N pytest processes at once, on one machine. The
# tests are independent subprocess runs of the writers (each reads the data and writes only a
# scratch directory, or nothing). tools/shard splits the collected tests into N shares by their
# weights in scripts/rederive-weights.txt, so each test is in exactly one share; each process
# deselects the other shares, and the script checks that what it will run is its own share
# before it runs it. Skips and failures of every process are judged exactly as with one.
#
# The pinned commits (AISIMULATE_SHA, INFERENCEX_SHA) live in
# .github/workflows/rederive.yml, which runs this same script.
set -euo pipefail

: "${AISIMULATE_DATA:?set AISIMULATE_DATA to the AISimulate systems/data directory}"
: "${BLIS_CATALOG:?set BLIS_CATALOG to the catalog/ directory of this repository}"
: "${AISIMULATE_ALLREDUCE_DATA:?set AISIMULATE_ALLREDUCE_DATA to the systems/data of aisimulate at AISIMULATE_ALLREDUCE_SHA (rederive.yml), the h100/h200 vLLM 0.31.0 all-reduce sweep}"
: "${COLLECTIVEX_DATA:?set COLLECTIVEX_DATA to the directory scripts/fetch_cited_inferencex.py fills}"
: "${AISIM_GLM53_DATA:?set AISIM_GLM53_DATA to the systems/data of aisim at AISIM_GLM53_SHA (rederive.yml), the GLM-5.3 attention-module table fit_glm53_attention.py reads}"

n=$(find "$AISIMULATE_DATA" -name '*.parquet' | wc -l)
echo "AISimulate parquets present: $n"
test "$n" -gt 500

# With an InferenceX checkout, its golden curves must be the files curves.json pins, all of
# them (the emitter refuses otherwise), curves.json must name INFERENCEX_SHA (when given), and
# the emitter must reproduce the set.
if [ -n "${INFERENCEX_GOLDEN:-}" ]; then
  if [ -n "${INFERENCEX_SHA:-}" ]; then
    grep -q "\"commit\": \"$INFERENCEX_SHA\"" validator/testdata/inferencex/curves.json
  fi
  python scripts/emit_acceptance_profiles.py --check
fi

# The rest-of-module prices fit_glm53_attention.py subtracts are a generated file: it must be
# what the kernel charges now, or the GLM-5.3 fit below is checked against a stale input.
(cd .. && go test ./kernel -run '^TestGLM53AttentionRestIsCurrent$' -count=1)

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
jobs="${REDERIVE_JOBS:-1}"
if [ "$jobs" -le 1 ]; then
  python -m pytest validator/ -q -rs --junitxml="$tmp/results.0.xml" -o junit_family=xunit2 | tee "$tmp/pytest.log"
else
  (cd .. && go build -o "$tmp/shard" ./tools/shard)
  weights=registry/scripts/rederive-weights.txt
  # Every collected test, one node id per line (a module skipped at import collects none and
  # is reported by its process below, as with one process).
  python -m pytest validator/ --collect-only -q -p no:cacheprovider | grep '::' > "$tmp/all.txt"
  (cd .. && "$tmp/shard" cover -of "$jobs" -min-items 100 -weights "$weights") < "$tmp/all.txt"
  pids=()
  for ((i = 0; i < jobs; i++)); do
    (cd .. && "$tmp/shard" select -of "$jobs" -index "$i" -weights "$weights") < "$tmp/all.txt" > "$tmp/mine.$i"
    # The others' tests, as --deselect arguments.
    grep -vxFf "$tmp/mine.$i" "$tmp/all.txt" > "$tmp/others.$i" || test $? -eq 1
    deselect=()
    while IFS= read -r id; do deselect+=(--deselect "$id"); done < "$tmp/others.$i"
    # --deselect matches by prefix: fail here, not silently, if it drops or keeps a wrong test.
    python -m pytest validator/ --collect-only -q -p no:cacheprovider "${deselect[@]}" | grep '::' | sort > "$tmp/will-run.$i"
    sort "$tmp/mine.$i" | diff - "$tmp/will-run.$i"
    echo "process $i runs $(wc -l < "$tmp/mine.$i") of $(wc -l < "$tmp/all.txt") tests"
    python -m pytest validator/ -q -rs -p no:cacheprovider "${deselect[@]}" \
      --junitxml="$tmp/results.$i.xml" -o junit_family=xunit2 > "$tmp/pytest.$i.log" 2>&1 &
    pids+=($!)
  done
  failed=0
  for ((i = 0; i < jobs; i++)); do
    wait "${pids[$i]}" || failed=1
    echo "::group::pytest process $i"
    cat "$tmp/pytest.$i.log"
    echo "::endgroup::"
  done
  test "$failed" -eq 0
fi

# Every skip, with its full reason, from the JUnit report (the terminal truncates reasons):
# a test as "<file>::<test> SKIPPED (<reason>)", a module skipped at import as a summary
# line for its file, the two forms `prcheck skips` reads.
python -I - "$tmp"/results.*.xml > "$tmp/skips.txt" <<'PY'
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


seen = set()
for report in sys.argv[1:]:
    for tc in ET.parse(report).iter("testcase"):
        s = tc.find("skipped")
        if s is None:
            continue
        reason = " ".join((s.get("message") or "").split())
        cls, name = tc.get("classname") or "", tc.get("name") or ""
        if cls:
            f, classes = node(cls)
            line = f"{f}::{'::'.join(classes + [name])} SKIPPED ({reason})"
        else:
            line = f"SKIPPED [1] {node(name)[0]}:0: {reason}"
        # A module skipped at import is reported by every process; list it once.
        if line not in seen:
            seen.add(line)
            print(line)
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
