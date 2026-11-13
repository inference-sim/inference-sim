#!/usr/bin/env bash
# ci-shard-verify.sh — fail unless the shards of a package ran every one of its tests exactly once.
#
# Usage: ci-shard-verify.sh EXPECTED_SHARDS DIR
#
# DIR holds one subdirectory per shard, each with the evidence files scripts/ci-shard.sh wrote
# (list.txt, assigned.txt, ran.txt). EXPECTED_SHARDS is the shard count the workflow declared, so
# a shard whose evidence never arrived (it was cancelled, or its upload failed) is an error and
# not a smaller partition that happens to cover itself.
#
# Checks: exactly EXPECTED_SHARDS subdirectories; every shard saw the same list; the assigned sets
# are pairwise disjoint; their union is the list; what each shard ran is what it was assigned.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: ci-shard-verify.sh EXPECTED_SHARDS DIR" >&2
  exit 2
fi
want="$1"; dir="$2"
fail=0
err() { echo "ci-shard-verify: $*" >&2; fail=1; }

shards=()
for d in "$dir"/*/; do [ -d "$d" ] && shards+=("${d%/}"); done
if [ "${#shards[@]}" -ne "$want" ]; then
  echo "ci-shard-verify: found ${#shards[@]} shard directories in $dir, expected $want" >&2
  exit 1
fi

first="${shards[0]}"
for s in "${shards[@]}"; do
  for f in list.txt assigned.txt ran.txt; do
    [ -f "$s/$f" ] || { err "$s/$f is missing"; continue; }
  done
done
[ "$fail" -eq 0 ] || exit 1
[ -s "$first/list.txt" ] || { echo "ci-shard-verify: the test list is empty" >&2; exit 1; }

all="$(mktemp)"; trap 'rm -f "$all"' EXIT
for s in "${shards[@]}"; do
  cmp -s "$first/list.txt" "$s/list.txt" || err "$s saw a different test list than $first"
  diff <(LC_ALL=C sort -u "$s/assigned.txt") <(LC_ALL=C sort -u "$s/ran.txt") >/dev/null \
    || err "$s ran a different set of tests than it was assigned"
  cat "$s/assigned.txt" >> "$all"
done
dup="$(LC_ALL=C sort "$all" | uniq -d)"
[ -z "$dup" ] || err "tests assigned to more than one shard: $(echo "$dup" | tr '\n' ' ')"
diff <(LC_ALL=C sort -u "$all") "$first/list.txt" >/dev/null \
  || err "the shards' union is not the package's test list (< assigned, > list): $(diff <(LC_ALL=C sort -u "$all") "$first/list.txt" | tr '\n' ' ')"
[ "$fail" -eq 0 ] || exit 1
echo "ci-shard-verify: $(wc -l < "$first/list.txt" | tr -d ' ') tests, each in exactly one of $want shards, all run"
