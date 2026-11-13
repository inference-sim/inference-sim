#!/usr/bin/env bash
# ci-shard.sh — run one shard of one package's tests, and prove the shards together run every test once.
#
# Usage: ci-shard.sh INDEX COUNT OUTDIR PACKAGE [go test flags...]
#
#   INDEX   1-based shard number, 1..COUNT
#   COUNT   number of shards the package is split into
#   OUTDIR  directory for the three evidence files (created if missing):
#             list.txt      every test, example and fuzz target the package declares (go test -list)
#             assigned.txt  the ones this shard owns
#             ran.txt       the top-level tests the run started (from the -v "=== RUN" lines)
#   PACKAGE one import path or ./relative path (one package per call)
#
# The partition is a pure function of the test's name: a test belongs to shard
# (cksum(name) mod COUNT) + 1. Every shard computes it from the same `go test -list`, so the
# shards are disjoint and cover the list by construction, and a new test lands in exactly one
# shard without anyone editing a list. cksum is POSIX, so the macOS and Linux values agree.
# CI_SHARD_SALT, when set, is put before the name before it is summed. It changes which shard a
# test falls in, never whether it falls in exactly one: the workflow picks a salt that spreads the
# few slow tests of a package over the shards, for speed only.
#
# Two checks keep the construction honest. This script fails when the tests that ran differ from
# the tests assigned (a bad -run pattern, a test that does not start), and
# scripts/ci-shard-verify.sh, run once after all shards, fails unless the assigned sets are
# pairwise disjoint and their union is the package's full list. A shard that is assigned no test
# while the package has tests is fine (COUNT may exceed the test count); an empty list is not.
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: ci-shard.sh INDEX COUNT OUTDIR PACKAGE [go test flags...]" >&2
  exit 2
fi
index="$1"; count="$2"; out="$3"; pkg="$4"; shift 4

case "$index$count" in *[!0-9]*|'') echo "ci-shard: INDEX and COUNT must be positive integers" >&2; exit 2 ;; esac
if [ "$index" -lt 1 ] || [ "$count" -lt 1 ] || [ "$index" -gt "$count" ]; then
  echo "ci-shard: INDEX $index is outside 1..$count" >&2
  exit 2
fi

mkdir -p "$out"

# vet is off for the listing only: the run below vets the package as every go test does.
# Test, Example and Fuzz functions are what `go test` with no -run starts (a fuzz target runs
# its seed corpus). Benchmarks are listed by -list but never started, so they are left out.
go test -vet=off -list '.' "$pkg" | grep -E '^(Test|Example|Fuzz)' | LC_ALL=C sort -u > "$out/list.txt" || true
if [ ! -s "$out/list.txt" ]; then
  echo "ci-shard: go test -list found no test in $pkg; refusing to report an empty shard as a pass" >&2
  exit 1
fi

: > "$out/assigned.txt"
while IFS= read -r name; do
  sum="$(printf '%s' "${CI_SHARD_SALT:-}$name" | cksum | cut -d' ' -f1)"
  if [ $(( sum % count + 1 )) -eq "$index" ]; then
    printf '%s\n' "$name" >> "$out/assigned.txt"
  fi
done < "$out/list.txt"

echo "ci-shard: shard $index of $count of $pkg owns $(wc -l < "$out/assigned.txt" | tr -d ' ') of $(wc -l < "$out/list.txt" | tr -d ' ') tests"
: > "$out/ran.txt"
if [ ! -s "$out/assigned.txt" ]; then
  exit 0
fi

pattern="^($(paste -sd'|' "$out/assigned.txt"))\$"
status=0
# -v is always on: the "=== RUN" lines are the evidence of what ran.
go test -v -run "$pattern" "$@" "$pkg" 2>&1 | tee "$out/go-test.log" || status=$?
# A top-level test's RUN line has no "/" in its name; subtests run under their parent.
sed -n 's/^=== RUN[[:space:]]*\([^[:space:]/]*\)[[:space:]]*$/\1/p' "$out/go-test.log" | LC_ALL=C sort -u > "$out/ran.txt"
rm -f "$out/go-test.log"

if [ "$status" -ne 0 ]; then
  exit "$status"
fi
if ! diff <(LC_ALL=C sort -u "$out/assigned.txt") "$out/ran.txt" > "$out/diff.txt"; then
  echo "ci-shard: the tests that ran are not the tests assigned to shard $index of $count (< assigned, > ran):" >&2
  cat "$out/diff.txt" >&2
  exit 1
fi
rm -f "$out/diff.txt"
