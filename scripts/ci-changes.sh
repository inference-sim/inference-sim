#!/usr/bin/env bash
# ci-changes.sh — which ci.yml jobs a change needs.
#
# Usage: ci-changes.sh [--all] < changed-paths      (one repository-relative path per line)
#
# Prints one `name=true|false` line per job group, in a fixed order, ready for $GITHUB_OUTPUT.
# --all (a push to main, a workflow_dispatch, a tag) turns every group on: the delivery loop
# dispatches ci.yml and reads a green run as "the full suite passed" (docs/contributing/
# automated-delivery.md), so a dispatch never filters.
#
# The rules are fail-closed. A path is matched against the classes below; a path that no class
# names turns EVERYTHING on, so a new directory costs a full run until someone classifies it. A
# class names the jobs that READ the path, with the evidence that they do:
#   - a Go package's own directory (all its files, and its testdata/) feeds the jobs that test it;
#   - a package a job's tests import feeds that job through its non-test .go files only (a
#     _test.go file of an imported package is never compiled into another package's test);
#   - a data file feeds the jobs whose tests open it.
# scripts/ci_changes_test.go checks the Go side against `go list -deps -test` for every job, so a
# new import cannot make a job read a directory this table does not send to it. The data side
# (what opens docs/, testdata/, ...) is grep evidence, confirmed by deleting each tree on a
# scratch branch and running the whole suite: only the jobs listed here failed.
#
# A _test.go file can only change the job that tests its package, `lint`, and `cmd` (cmd's
# repository guards scan every Go source). A path under .github/ is NOT "shared": only the
# workflow files are read, by the scripts tests and actionlint.
#
# Groups: build lint core cluster kernel workload cmd root scripts llmd bytegate other stack
#         selected actionlint claudemd
set -euo pipefail

groups="build lint core cluster kernel workload cmd root scripts llmd bytegate other stack selected actionlint claudemd"

all=0
if [ "${1:-}" = "--all" ]; then all=1; shift; fi
if [ "$#" -ne 0 ]; then echo "usage: ci-changes.sh [--all] < changed-paths" >&2; exit 2; fi

on() { eval "f_$1=1"; }
for g in $groups; do eval "f_$g=0"; done
if [ "$all" -eq 1 ]; then for g in $groups; do on "$g"; done; fi

p=""
# direct DIR: a file directly in DIR ("." is the repository root, where only .go files belong to
# the package; the data files there are classed below).
direct() { if [ "$1" = . ]; then [[ $p != */* && $p == *.go ]]; else [[ $p == "$1"/* && $p != "$1"/*/* ]]; fi; }
# own DIR: a file of the package in DIR, or its testdata/.
own() { direct "$1" || { [ "$1" != . ] && [[ $p == "$1"/testdata/* ]]; }; }
# dep DIR: a non-test file of a package that the job's tests or binary import.
dep() { direct "$1" && [[ $p != *_test.go ]]; }
# any_of FUNC DIR...: FUNC holds for one of the directories.
any_of() { local f=$1 d; shift; for d in "$@"; do "$f" "$d" && return 0; done; return 1; }

# classify: set the groups path $p needs; return 1 when no class names the path.
classify() {
  local hit=0
  mark() { hit=1; local g; for g in "$@"; do on "$g"; done; }

  # -- Go packages: the job that tests the package, plus the jobs whose tests import it. ----------
  # Closures are from `go list -deps -test`; scripts/ci_changes_test.go re-derives them.
  any_of own sim sim/trace                  && mark core
  any_of dep internal/invariantscan         && mark core cluster
  any_of own sim/cluster                    && mark cluster
  any_of own tools/harness tools/harness/corpora tools/harness/internal/artifacts \
         tools/kernelscore tools/metricscore tools/sensitivity tools/signedprobe && mark kernel
  any_of own sim/workload                   && mark workload
  any_of own cmd                            && mark cmd
  any_of own . sim/internal/testutil        && mark root
  any_of own scripts                        && mark scripts
  any_of own sim/llmd                       && mark llmd
  any_of own internal/bytegate              && mark bytegate
  any_of own internal/invariantscan internal/kernelfixtures internal/stack internal/stack/cmd/stackgen \
         internal/stack/cmd/versions-doc internal/substratedata sim/check sim/saturation && mark other

  # The stack job runs internal/stack's tests and its two generators.
  case "$p" in internal/stack/*) mark stack ;; esac

  # Imported packages (non-test files only).
  any_of dep sim                            && mark core cluster kernel workload cmd root llmd other bytegate
  any_of dep sim/trace                      && mark core cluster cmd root llmd other bytegate
  any_of dep sim/check                      && mark cluster kernel workload cmd root llmd other bytegate
  any_of dep sim/workload                   && mark cluster kernel cmd root llmd other bytegate
  any_of dep sim/cluster                    && mark cmd root llmd other bytegate
  any_of dep sim/llmd                       && mark cmd root other bytegate
  any_of dep sim/saturation                 && mark cmd root other bytegate
  any_of dep sim/internal/testutil          && mark core cluster llmd
  any_of dep internal/kernelfixtures        && mark cluster kernel
  any_of dep internal/stack                 && mark cmd root other bytegate
  any_of dep internal/substratedata         && mark cmd scripts bytegate other
  any_of dep cmd                            && mark root other bytegate
  any_of dep .                              && mark root bytegate cmd
  # The blis binary: the byte gate builds it, and selected-deps lists the engine and router
  # packages it links (an added import changes the list). tools/ and scripts/ are not linked.
  any_of dep . cmd internal/stack sim sim/check sim/cluster sim/llmd sim/saturation sim/trace \
         sim/workload                       && mark bytegate selected
  # main.go and defaults.yaml (embedded) are the binary every blis-run job builds.
  case "$p" in
    main.go|defaults.yaml) mark root bytegate cmd cluster workload kernel build selected ;;
  esac

  # -- Any Go file: build and lint see it; cmd's repository guards scan every Go source. ----------
  case "$p" in
    *_test.go) mark lint cmd ;;
    *.go)      mark lint cmd build ;;
  esac

  # -- Modules and the stack record. -----------------------------------------------------------
  case "$p" in
    go.mod|go.sum)           for g in $groups; do mark "$g"; done ;;
    stack.json|guides.lock)  mark cmd other stack selected root ;;
  esac

  # -- Data trees (grep evidence + deletion run in the header). --------------------------------
  case "$p" in
    testdata/bytegate/*)                  mark bytegate workload cmd ;;
    testdata/kernel/*)                    mark cmd core cluster kernel other scripts ;;
    testdata/scenarios/*)                 mark cmd bytegate kernel ;;
    testdata/inferencex-workloads/*)      mark kernel cmd ;;
    testdata/trace-v3/*)                  mark workload ;;
    testdata/acceptance/*)                mark cmd ;;
    testdata/recurrent/*)                 mark cmd ;;
    testdata/goldendataset.json)          mark cluster llmd root ;;
    testdata/README.md)                   mark other ;;
    docs/*|README.md|CONTRIBUTING.md|mkdocs.yml)
                                          mark core cluster cmd scripts stack ;;
    AGENTS.md)                            mark core cluster cmd scripts stack workload other ;;
    CLAUDE.md)                            mark core cluster cmd scripts stack workload other claudemd ;;
    examples/*|examplescenarios/*)        mark workload cmd ;;
    # The filter and the shard tools decide what runs and whether it all ran: a change to one is
    # a change to every job.
    scripts/ci-*.sh|.github/workflows/ci.yml) for g in $groups; do mark "$g"; done ;;
    scripts/test-selected-deps.sh)        mark selected scripts ;;
    scripts/claude-md-size.sh)            mark claudemd scripts ;;
    scripts/*)                            mark scripts ;;
    tools/validation/*)                   mark cmd ;;
    tools/*)                              mark kernel ;;
    .github/workflows/*|.github/actionlint.yaml) mark scripts actionlint ;;
    # Read by the scripts package's tests (archon_workflow, dockerfile_go_version, pr_review).
    Dockerfile|k8s/*|.github/archon-trigger-anchor.patch|.archon-version)
                                          mark scripts ;;
    .claude/*)                            mark scripts cmd ;;
    # Deleting these left every job green (scratch run with the whole tree present otherwise).
    LICENSE|.gitignore|.github/ISSUE_TEMPLATE/*|.github/pull_request_template.md|\
    .github/docs-requirements.txt|specs/*)
                                          hit=1 ;;
  esac
  [ "$hit" -eq 1 ]
}

seen=0
while IFS= read -r p || [ -n "$p" ]; do
  [ -n "$p" ] || continue
  seen=1
  if ! classify; then
    echo "ci-changes: '$p' is in no class, so every job runs (add a class to scripts/ci-changes.sh)" >&2
    for g in $groups; do on "$g"; done
  fi
done

# An empty change list (a re-run with nothing to diff) is not evidence that nothing changed.
if [ "$seen" -eq 0 ]; then for g in $groups; do on "$g"; done; fi

for g in $groups; do eval "v=\$f_$g"; if [ "$v" -eq 1 ]; then echo "$g=true"; else echo "$g=false"; fi; done
