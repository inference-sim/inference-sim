#!/usr/bin/env bash
# List the items a Go test group is made of, one per line: a package ("compat/build"; "." is the
# module's root package), or, for a package named in the split file, each of its top-level tests
# ("compat/kernelmodel TestName"). Only packages that hold a test are listed.
#
# Usage: go-test-items.sh [-exclude REGEX] [-split FILE] PATTERN...
#   PATTERN  go list patterns (./... for every package)
#   -exclude drops the packages whose import path matches the regular expression
#   -split   packages to list by test (default .github/shards/split.txt; one relative path per line)
set -euo pipefail
exclude=""
split=".github/shards/split.txt"
while [[ "${1:-}" == -* ]]; do
	case "$1" in
	-exclude) exclude="$2"; shift 2 ;;
	-split) split="$2"; shift 2 ;;
	*) echo "go-test-items: unknown flag $1" >&2; exit 2 ;;
	esac
done
[[ $# -ge 1 ]] || { echo "usage: go-test-items.sh [-exclude REGEX] [-split FILE] PATTERN..." >&2; exit 2; }
module="$(go list -m)"
splitset=" $(grep -vE '^\s*(#|$)' "$split" 2>/dev/null | tr '\n' ' ')"
go list -f '{{if or .TestGoFiles .XTestGoFiles}}{{.ImportPath}}{{end}}' "$@" | sed '/^$/d' | sort -u | while read -r pkg; do
	if [[ -n "$exclude" ]] && [[ "$pkg" =~ $exclude ]]; then continue; fi
	rel="${pkg#"$module"}"; rel="${rel#/}"; rel="${rel:-.}"
	if [[ "$splitset" == *" $rel "* ]]; then
		# -list builds the test binary and prints its tests, examples and fuzz targets (and an "ok" line).
		go test -list '.*' "$pkg" | grep -E '^(Test|Example|Fuzz)[A-Za-z0-9_]*$' | sort -u | sed "s|^|$rel |"
	else
		echo "$rel"
	fi
done
