#!/usr/bin/env bash
# catalog-load-gate.sh — the R1/C6 strict catalog-load gate (#1750).
#
# Loads EVERY entry in the authoritative blis-catalog through the same loader code path
# `blis run` uses, and fails on any rejection. C6 states: "There is no validate command: the
# simulator validates whatever it reads and fails with a message naming the file and the
# problem. CI runs a load over every catalog entry, which is the same code path." So the gate
# is this script driving cmd.TestCatalogStrictLoad_RealCatalog — not a `blis validate`
# subcommand, and not a bespoke validator that could accept what a run rejects.
#
# The catalog revision is PINNED here, not floating. A green run must mean "this commit loads
# THAT catalog": tracking blis-catalog's default branch would let an unrelated catalog commit
# redden an unrelated PR, and would make a past green run unreproducible. To adopt catalog
# changes, bump CATALOG_REVISION in its own PR — that PR is then the review of the catalog
# change, which is the point of the gate.
#
# It lives here rather than inline in a workflow for the reason every other script in this
# directory does: a workflow step cannot be tested, and this one has a failure mode that is
# invisible from its exit code (see the SKIP guard below). scripts/catalog_load_gate_test.go
# drives it against throwaway catalogs.
#
# CI WIRING — TRACKED BY #1823, a human step. The delivery loop's GITHUB_TOKEN has no
# `workflows` permission (there is no such permission to grant in a `permissions:` block), so a
# push touching .github/workflows/* is rejected outright; see
# docs/contributing/automated-delivery.md. Until #1823 lands, this script runs on demand rather
# than on every PR, and the authoritative-catalog leg of the gate is NOT enforced pre-merge — the
# unconditional fixture leg (cmd.TestCatalogStrictLoad_CommittedFixtureCatalog) still is. Add this
# job to .github/workflows/ci.yml:
#
#   catalog-load:
#     runs-on: ubuntu-latest
#     steps:
#       - name: Checkout code
#         uses: actions/checkout@v4
#       - name: Setup Go
#         uses: actions/setup-go@v5
#         with:
#           go-version: '1.21'
#           cache: true
#       - name: Load every catalog entry through the loader
#         run: scripts/catalog-load-gate.sh
#
# The job needs no `with: repository:` checkout of the catalog and no env: block — this script
# clones the pinned revision itself, so the pinned SHA has exactly one home and the workflow
# cannot drift from it.
#
# Usage:
#   scripts/catalog-load-gate.sh
#
# | Variable           | Default                              | Meaning                                           |
# |--------------------|--------------------------------------|---------------------------------------------------|
# | BLIS_CATALOG       | (unset)                              | An EXISTING catalog checkout to load. When set, no clone happens and the revision is whatever that checkout holds — for a local run against a work-in-progress catalog. |
# | CATALOG_REPO       | the blis-catalog GitHub URL          | Clone source. Overridden by the tests to point at a throwaway local repository instead of the network. |
# | CATALOG_REVISION   | the pinned SHA below                 | Revision to check out. Overridden by the tests, whose fixture repositories have their own SHAs. |
# | CATALOG_CHECKOUT   | a fresh mktemp directory             | Where to clone. |
# | GATE_TIMEOUT       | 5m                                   | `go test -timeout`. |
set -euo pipefail

CATALOG_REPO="${CATALOG_REPO:-https://github.com/inference-sim/blis-catalog.git}"
CATALOG_REVISION="${CATALOG_REVISION:-6fe4664576bc43d601bba6619aca4fbb42c5087b}"
GATE_TIMEOUT="${GATE_TIMEOUT:-5m}"

# Run from the repository root whatever the caller's working directory is, so `./cmd` resolves.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${BLIS_CATALOG:-}" ]]; then
  echo "catalog-load gate: loading the catalog already checked out at ${BLIS_CATALOG}"
else
  checkout="${CATALOG_CHECKOUT:-$(mktemp -d)}"
  echo "catalog-load gate: cloning ${CATALOG_REPO} at ${CATALOG_REVISION} into ${checkout}"
  # A full clone (not --depth 1): a pinned SHA is not fetchable by a shallow clone of a branch
  # tip once the branch has moved past it, which is precisely the situation pinning creates.
  git clone --quiet "$CATALOG_REPO" "$checkout"
  git -C "$checkout" checkout --quiet "$CATALOG_REVISION"
  export BLIS_CATALOG="$checkout"
fi

log="$(mktemp)"
status=0
go test -v -count=1 -timeout "$GATE_TIMEOUT" -run TestCatalogStrictLoad_RealCatalog ./cmd \
  2>&1 | tee "$log" || status=$?

# The did-it-actually-run guard is the load-bearing part. TestCatalogStrictLoad_RealCatalog SKIPS
# when BLIS_CATALOG is unset — deliberately, so a local `go test ./...` needs no catalog
# checkout — and `go test` reports a skip as a PASS. Without this check, a broken clone or a lost
# environment variable would turn the gate green having loaded nothing at all.
#
# Both patterns are ANCHORED to the top-level test name. An unanchored `--- SKIP` would also match
# a SKIPPED SUBTEST (go test indents those: `    --- SKIP: TestX/sub`) and any log line quoting
# the string, failing the gate on a load that ran fine.
if grep -qE -- '^--- SKIP: TestCatalogStrictLoad_RealCatalog([[:space:]]|$)' "$log"; then
  echo "ERROR: the catalog-load gate SKIPPED — BLIS_CATALOG did not reach the test, so no catalog entry was loaded" >&2
  exit 1
fi
if [[ "$status" -ne 0 ]]; then
  echo "ERROR: the catalog at ${BLIS_CATALOG} did not load clean (see the failure above)" >&2
  exit "$status"
fi
# The positive half: the load must have PASSED, not merely not-failed. A `-run` pattern matching
# nothing (a renamed test) exits 0 with "no tests to run", which the skip check above cannot see.
if ! grep -qE -- '^--- PASS: TestCatalogStrictLoad_RealCatalog([[:space:]]|$)' "$log"; then
  echo "ERROR: the catalog-load gate did not RUN TestCatalogStrictLoad_RealCatalog (no PASS line) — the -run pattern matched no test" >&2
  exit 1
fi
echo "catalog-load gate: every catalog entry loaded through the production loader"
