#!/usr/bin/env bash
# The re-derivation gates, with the data present: every fitted value must be what its
# writer derives from the pinned upstream data, and no test may skip except the eight
# by-design ramp-form ones and a named cited-bodies-absent skip (below). A skip is a gate
# that did not run, so it fails here.
#
# Run from registry/ after `pip install --require-hashes -r scripts/requirements-fit.txt`:
#
#   export AISIMULATE_DATA=<aisimulate>/python/aisimulate/src/aisimulate_core/systems/data
#   export BLIS_CATALOG=<the repository's catalog/ directory>
#   export AISIMULATE_ALLREDUCE_DATA=<aisimulate @7980e95>/.../systems/data  # relane_allreduce_vllm.py
#   export COLLECTIVEX_DATA=<dir of cx_run_*.body>     # test_deepep_ll_graph.py
#   export OPERATORX_DATA=<dir of moe_<Model>.json>    # test_operatorx.py
#   export INFERENCEX_GOLDEN=<inferencex>/inferencex-e2e/infx/golden_al_distribution  # optional
#   scripts/rederive.sh
#
# The pinned commits (AISIMULATE_SHA, INFERENCEX_SHA) live in
# .github/workflows/rederive.yml, which runs this same script on manual dispatch.
set -euo pipefail

: "${AISIMULATE_DATA:?set AISIMULATE_DATA to the AISimulate systems/data directory}"
: "${BLIS_CATALOG:?set BLIS_CATALOG to the repository's catalog/ directory}"
: "${AISIMULATE_ALLREDUCE_DATA:?set AISIMULATE_ALLREDUCE_DATA to the systems/data of aisimulate at AISIMULATE_ALLREDUCE_SHA (rederive.yml), the h100/h200 vLLM 0.31.0 all-reduce sweep}"

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

log=$(mktemp)
trap 'rm -f "$log"' EXIT
python -m pytest validator/ -q -rs | tee "$log"
# With the data present, the only skips are the eight by-design ramp-form ones (a part
# carries the one- or three-factor ramp, so the other gate does not apply). Any other
# skip -- missing data, a missing catalog, a gate whose premise quietly disappeared --
# is a gate that did not run, and fails.
# One data skip is accepted, by name: a CollectiveX writer whose cited run bodies are no
# longer served (the API keeps its latest 15 runs) refuses to re-cite them and skips with
# "the cited CollectiveX bodies are not present". Its entries then cannot be re-derived from
# public data, and the skip line says which runs are missing.
unexpected=$(grep '^SKIPPED' "$log" \
  | grep -Ev 'carries the (one|three)-factor ramp|the cited CollectiveX bodies are not present' \
  || true)
if [ -n "$unexpected" ]; then
  echo "unexpected skips with the data present:" >&2
  echo "$unexpected" >&2
  exit 1
fi
echo "rederive: every gate ran and passed"
