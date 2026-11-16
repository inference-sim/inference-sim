#!/usr/bin/env bash
# deadcode-check.sh — no code that none of the binaries BLIS ships can reach (design LIF-7).
#
# Usage: deadcode-check.sh ALLOW_LIST          (run from the root of the module to check)
#
# It runs Go's whole-program dead-code analysis (golang.org/x/tools/cmd/deadcode, at the version
# below) over every package of the module, rooted at every main package in it, with test files
# not counted as reaching code. Every main package of a module is a binary that module ships:
# a main package nobody builds or runs is itself code no shipped binary reaches, and is deleted
# rather than rooted. A package no main package links is reported too, function by function.
#
# It fails when
#   - a function no main package reaches, exported or not, is not on the allow-list;
#   - an allow-list entry has no reason, or is listed twice;
#   - an allow-list entry names a function that is no longer unreachable, so the list cannot go
#     stale.
# Packages under contracts/ are exempt, as the design says (inference-sim has none of its own).
#
# The allow-list has one entry per line: the package's directory relative to the module root
# ("." for the root package), the function as deadcode names it (Func, Type.Method), and the
# reason it must stay. Blank lines and lines starting with # are ignored.
#
# DEADCODE_REPORT, when set, is a file holding deadcode's output in this script's -f format, read
# instead of running deadcode (scripts/deadcode_check_test.go drives the rules with it).
set -euo pipefail

# The one place this version is written (X21, X23).
deadcode_version=v0.52.0

if [ "$#" -ne 1 ]; then echo "usage: deadcode-check.sh ALLOW_LIST" >&2; exit 2; fi
allow=$1
if [ ! -f "$allow" ]; then echo "deadcode-check: allow-list $allow is missing" >&2; exit 2; fi

module=$(go list -m)
report=$(mktemp)
trap 'rm -f "$report"' EXIT

if [ -n "${DEADCODE_REPORT:-}" ]; then
  cp "$DEADCODE_REPORT" "$report"
else
  # -test is off (the default): a function only a test calls is unreachable.
  filter="^$(printf '%s' "$module" | sed 's/[.]/\\./g')(/|\$)"
  go run "golang.org/x/tools/cmd/deadcode@${deadcode_version}" -filter "$filter" \
    -f '{{range .Funcs}}{{$.Path}} {{.Name}} {{.Position}}{{"\n"}}{{end}}' ./... > "$report"
fi

awk -v module="$module" '
  function dir_of(path) {
    if (path == module) return "."
    if (index(path, module "/") == 1) return substr(path, length(module) + 2)
    return path
  }
  # ARGV order, not FNR, tells the files apart, so an empty allow-list still works.
  { file = (FILENAME == ARGV[1]) ? 1 : 2 }
  # The allow-list: directory, function, reason.
  file == 1 {
    line = $0
    sub(/^[ \t]+/, "", line)
    if (line == "" || substr(line, 1, 1) == "#") next
    key = $1 " " $2
    if (NF < 3) {
      printf "deadcode-check: allow-list line %d (%s) gives no reason\n", FNR, key
      bad = 1
      next
    }
    if (key in allowed) {
      printf "deadcode-check: allow-list line %d lists %s again\n", FNR, key
      bad = 1
    }
    allowed[key] = FNR
    next
  }
  # deadcode: import path, function, position.
  file == 2 {
    if (NF == 0) next
    d = dir_of($1)
    if (d == "contracts" || index(d, "contracts/") == 1) next
    key = d " " $2
    dead[key] = 1
    if (!(key in allowed)) {
      printf "deadcode-check: %s: \"%s %s\" is reached by no main package and is not on the allow-list\n", $3, d, $2
      bad = 1
    }
  }
  END {
    for (key in allowed) {
      if (!(key in dead)) {
        split(key, k, " ")
        printf "deadcode-check: allow-list line %d names \"%s %s\", which a main package now reaches (or which no longer exists); remove the entry\n", allowed[key], k[1], k[2]
        bad = 1
      }
    }
    exit bad
  }
' "$allow" "$report" || {
  echo "deadcode-check: FAILED (design LIF-7). Delete the code, move a test-only helper into a _test.go file of its package, or allow-list it with its reason in $allow." >&2
  exit 1
}
echo "deadcode-check: every function is reached by a main package of $module or allow-listed with a reason"
