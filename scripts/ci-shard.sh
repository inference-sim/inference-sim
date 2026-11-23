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
# Two optional rules change the assignment, again as a pure function of the list, so the shards
# stay disjoint and covering:
#   CI_SHARD_SOLO       test names, space separated, each given a shard of its own: with K names,
#                       the j-th owns shard COUNT-K+j alone and every other test is hashed over
#                       shards 1..COUNT-K. A name the package does not declare is an error (a
#                       renamed test must not leave its shard running nothing). K < COUNT.
#   CI_SHARD_DURATIONS  a file of "name<TAB>seconds" lines (# starts a comment). The tests it
#                       names are dealt longest first, each to the shard with the least recorded
#                       time so far (ties: by name, then the lower shard); a test it does not
#                       name is hashed as above. A stale entry (no such test) is ignored.
# CI_SHARD_LIST_FLAGS, when set, is added to the `go test -list` call (for example -race, so the
# listing builds the binary the run then reuses).
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
# shellcheck disable=SC2086
go test -vet=off ${CI_SHARD_LIST_FLAGS:-} -list '.' "$pkg" | grep -E '^(Test|Example|Fuzz)' | LC_ALL=C sort -u > "$out/list.txt" || true
if [ ! -s "$out/list.txt" ]; then
  echo "ci-shard: go test -list found no test in $pkg; refusing to report an empty shard as a pass" >&2
  exit 1
fi

# The solo tests, each checked against the list, take the last shards.
read -r -a solo <<< "${CI_SHARD_SOLO:-}"
hashed="$count"
if [ "${#solo[@]}" -gt 0 ]; then
  if [ "${#solo[@]}" -ge "$count" ]; then
    echo "ci-shard: ${#solo[@]} solo tests need more than $count shards" >&2
    exit 2
  fi
  hashed=$(( count - ${#solo[@]} ))
  for name in "${solo[@]}"; do
    if ! grep -qxF "$name" "$out/list.txt"; then
      echo "ci-shard: solo test $name is not a test of $pkg" >&2
      exit 1
    fi
  done
fi
durations="${CI_SHARD_DURATIONS:-/dev/null}"
if [ ! -r "$durations" ]; then
  echo "ci-shard: CI_SHARD_DURATIONS file $durations cannot be read" >&2
  exit 2
fi

# One awk program assigns every test (so every shard computes the same table) and prints the
# tests of this shard: solo first, then the recorded tests longest first onto the least-loaded
# shard, then the rest by checksum. cksum's checksum is computed per name beforehand.
while IFS= read -r name; do
  printf '%s\t%s\n' "$name" "$(printf '%s' "${CI_SHARD_SALT:-}$name" | cksum | cut -d' ' -f1)"
done < "$out/list.txt" > "$out/sums.txt"
awk -F'\t' -v index_="$index" -v hashed="$hashed" -v solo="${solo[*]:-}" '
  FILENAME == ARGV[1] { if ($0 !~ /^#/ && NF >= 2) dur[$1] = $2 + 0; next }
  { names[++n] = $1; sum[$1] = $2 }
  END {
    k = split(solo, s, " ")
    for (j = 1; j <= k; j++) { shard[s[j]] = hashed + j; taken[s[j]] = 1 }
    m = 0
    for (i = 1; i <= n; i++) if (!(names[i] in taken) && (names[i] in dur)) rec[++m] = names[i]
    # Longest first, ties by name (insertion sort: the recorded lists are short).
    for (i = 2; i <= m; i++) {
      v = rec[i]; j = i - 1
      while (j >= 1 && (dur[rec[j]] < dur[v] || (dur[rec[j]] == dur[v] && rec[j] > v))) { rec[j + 1] = rec[j]; j-- }
      rec[j + 1] = v
    }
    for (b = 1; b <= hashed; b++) load[b] = 0
    for (i = 1; i <= m; i++) {
      best = 1
      for (b = 2; b <= hashed; b++) if (load[b] < load[best]) best = b
      shard[rec[i]] = best; load[best] += dur[rec[i]]; taken[rec[i]] = 1
    }
    for (i = 1; i <= n; i++) {
      t = names[i]
      if (!(t in taken)) shard[t] = sum[t] % hashed + 1
      if (shard[t] == index_) print t
    }
  }' "$durations" "$out/sums.txt" > "$out/assigned.txt"
rm -f "$out/sums.txt"

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
