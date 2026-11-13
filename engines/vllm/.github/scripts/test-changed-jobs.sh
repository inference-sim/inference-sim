#!/usr/bin/env bash
# Which jobs run for which change. Each case below is a change list and the exact set of ci.yml jobs it
# selects (changed_jobs.py), in the order the outputs print. A job outside the set is skipped.
# Then the audit (audit_triggers.py) checks that the lists are complete for the files the jobs read, and
# the shard tool's own tests run. Needs go on the PATH and a checkout.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here/../.."
fail=0
names=()
wants=()
legs_wants=()
input=""
# case NAME "JOBS" "LEGS" -- file...   JOBS: the jobs that are selected, sorted, space separated ("-" for none);
# LEGS: the race legs that run ("-" for none). A job outside the list is skipped.
case_() {
	local name="$1" want="$2" legs="$3"; shift 4
	names+=("$name"); wants+=("$want"); legs_wants+=("$legs")
	[[ -z "$input" ]] || input+=$'\n--\n'
	input+="$(printf '%s\n' "$@")"
}
ALL="bench go goldens inpackage kerneltests lint metrics oracle packs_v0_29 packs_v0_31 packs_v0_32 race race_guard rest"
CENSUS="kernelmodel-census-0 kernelmodel-census-1 kernelmodel-census-2 kernelmodel-census-3"
ENGINE="vllmengine-0 vllmengine-1"
EVERY_LEG="core compat $ENGINE $CENSUS"

case_ "a documents-only change"                 "-" "-" -- docs/testing.md docs/feature-matrix.md README.md AGENTS.md
case_ "a licence file"                          "-" "-" -- LICENSE NOTICE .gitignore
case_ "the pull-request template"               "-" "-" -- .github/pull_request_template.md
case_ "another workflow (blis, mutation)"       "-" "-" -- .github/workflows/blis.yml .github/workflows/mutation.yml .github/workflows/oracle-traces.yml .github/workflows/private-names.yml
case_ "a Python file of the oracle"             "goldens oracle" "-" -- oracle/gen_fuzz.py
case_ "the oracle's fuzz cache action"          "oracle" "-" -- .github/actions/oracle-fuzz/action.yml
case_ "fields.json only"                        "bench go inpackage kerneltests metrics oracle race rest" "$EVERY_LEG" -- fields.json
case_ "a fields generator"                      "metrics" "-" -- fields/gen_fields.py
case_ "an oracle golden only"                   "bench goldens oracle race rest" "core" -- core/sched/testdata/golden/dashboard-deepseek-v4-pro-layers.json
case_ "an oracle scenario"                      "goldens oracle race rest" "$EVERY_LEG" -- oracle/scenarios/connector/nixl-d-abort-waiting.json
case_ "a rules pack's source"                   "bench go inpackage kerneltests lint oracle packs_v0_32 race race_guard rest" "core $ENGINE $CENSUS" -- rules/v0_32/pack.go
case_ "a rules pack's test"                     "go lint packs_v0_32 rest" "-" -- rules/v0_32/source_test.go
case_ "a kernelmodel Go file"                   "go kerneltests lint race race_guard rest" "$ENGINE $CENSUS" -- compat/kernelmodel/acceptance.go
case_ "a kernelmodel test"                      "go lint race rest" "$CENSUS" -- compat/kernelmodel/blocksize_census_test.go
case_ "a kernel-test fixture"                   "inpackage kerneltests race rest" "$EVERY_LEG" -- kerneltests/testdata/direct/x.yaml
case_ "a kernel test in the kernel package"     "go inpackage" "-" -- kerneltests/_inpackage/caches_test.go
case_ "the kernel tests' documents"             "kerneltests" "-" -- kerneltests/docs/index.md
case_ "a vllmengine Go file"                    "go lint race race_guard rest" "$ENGINE" -- vllmengine/engine.go
case_ "go.mod"                                  "bench go inpackage kerneltests lint metrics oracle packs_v0_29 packs_v0_31 packs_v0_32 race rest" "$EVERY_LEG" -- go.mod
case_ "the lint configuration"                  "lint" "-" -- .golangci.yml
case_ "the CI workflow"                         "$ALL" "$EVERY_LEG" -- .github/workflows/ci.yml
case_ "the shard tool"                          "inpackage kerneltests oracle race rest" "$EVERY_LEG" -- .github/scripts/shard.py
case_ "the inertness shard weights"             "oracle" "-" -- .github/shards/weights-inertness.txt
case_ "the shard weights"                       "inpackage kerneltests race rest" "$EVERY_LEG" -- .github/shards/weights.txt
case_ "the race legs file"                      "race race_guard" "$EVERY_LEG" -- .github/shards/race-legs.json
case_ "the selection script"                    "$ALL" "$EVERY_LEG" -- .github/scripts/changed_jobs.py
case_ "the Go setup action"                     "$ALL" "$EVERY_LEG" -- .github/actions/go-setup/action.yml
case_ "the oracle environment action"           "oracle" "-" -- .github/actions/oracle-env/action.yml
case_ "a script only people run"                "-" "-" -- .github/scripts/measure_weights.py
case_ "a docs change beside a workflow it does not run" "-" "-" -- docs/testing.md .github/workflows/blis.yml

got="$(printf '%s' "$input" | python3 -I "$here/changed_jobs.py" --batch)"
i=0
while IFS= read -r line; do
	jobs="${line%%|*}"; legs="${line##*|}"
	want="${wants[$i]}"; wl="${legs_wants[$i]}"
	# the wanted job list, sorted
	want="$(printf '%s\n' $want | sort | tr '\n' ' ' | sed 's/ $//')"
	wl="$(printf '%s\n' $wl | sort | tr '\n' ' ' | sed 's/ $//')"
	gl="$(printf '%s\n' $legs | sort | tr '\n' ' ' | sed 's/ $//')"
	if [[ "$jobs" == "$want" && "$gl" == "$wl" ]]; then echo "ok   ${names[$i]}: $jobs"; else
		echo "FAIL ${names[$i]}"; echo "  want jobs: $want"; echo "  got jobs:  $jobs"; echo "  want legs: $wl"; echo "  got legs:  $gl"; fail=1; fi
	i=$((i + 1))
done <<< "$got"
[[ "$i" -eq "${#names[@]}" ]] || { echo "FAIL: $i results for ${#names[@]} cases"; fail=1; }

# Every job runs on a push to main and on a run by hand.
all="$(printf '' | python3 -I "$here/changed_jobs.py" --all | awk -F= '$1!="race_matrix" && $2!="true" {print $1}')"
if [[ -z "$all" ]]; then echo "ok   --all selects every job"; else echo "FAIL --all leaves out: $all"; fail=1; fi

# The pull-request path of the `changes` job, run as written in ci.yml on a real merge commit: the checkout of
# a pull request is the merge of its head into its base, and the files the pull request changes are the
# difference from the first parent. A pull request that changes one document selects no job; one that changes a
# golden selects the oracle and the tests that read it.
pr_event() {
	local tmp step want got
	tmp="$(mktemp -d)"
	git clone -q --shared . "$tmp/repo"
	python3 -I - "$here/../workflows/ci.yml" "$tmp/step.sh" <<'PY'
import re, sys, textwrap
s = open(sys.argv[1]).read()
i = s.index("      - id: filter")
j = s.index("        run: |\n", i) + len("        run: |\n")
k = s.index("\n\n", j)
open(sys.argv[2], "w").write(textwrap.dedent(s[j:k]) + "\n")
PY
	(
		cd "$tmp/repo"
		git config user.email ci@example.invalid; git config user.name ci; git config commit.gpgsign false
		git checkout -q --detach "$(git -C "$here/../.." rev-parse HEAD)"
		# the base is this checkout's commit with its working tree's CI files, which may not be committed yet
		cp -R "$here/../scripts/." .github/scripts/ && cp -R "$here/../shards/." .github/shards/ && cp "$here/../workflows/ci.yml" .github/workflows/ci.yml
		git add -A; git commit -q --allow-empty -m "base"
		base="$(git rev-parse HEAD)"
		for case in "docs/testing.md:-" "core/sched/testdata/golden/dashboard-deepseek-v4-pro-layers.json:goldens oracle race rest"; do
			f="${case%%:*}"; want="${case#*:}"
			git checkout -q -B "pr" "$base"; echo "edit" >> "$f"; git add -A; git commit -q -m "pr"
			git checkout -q -B "merge" "$base"; git merge --no-ff -q pr -m "merge"
			: > "$tmp/out"
			EVENT=pull_request BASE="$base" CHANGED= GITHUB_OUTPUT="$tmp/out" RUNNER_TEMP="$tmp" bash "$tmp/step.sh" > "$tmp/log" 2>&1
			got="$(awk -F= '$2=="true" && $1!="race_matrix" && $1!="bench" {print $1}' "$tmp/out" | sort | tr '\n' ' ' | sed 's/ $//')"
			[[ -n "$got" ]] || got="-"
			want="$(printf '%s\n' $want | sort | tr '\n' ' ' | sed 's/ $//')"
			if [[ "$got" == "$want" ]]; then echo "ok   a pull request that changes $f: $got"; else echo "FAIL a pull request that changes $f: want $want, got $got"; cat "$tmp/log"; exit 1; fi
		done
	) || fail=1
	rm -rf "$tmp"
}
pr_event

python3 -I "$here/audit_triggers.py" || fail=1
python3 -I "$here/test_shard.py" || fail=1
python3 -I "$here/check_race_legs.py" || fail=1
exit "$fail"
