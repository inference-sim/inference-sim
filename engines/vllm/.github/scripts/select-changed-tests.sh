#!/usr/bin/env bash
# Read the directories a layer changed (one per line, on stdin) and say how each is tested:
#   direct <dir>   go test ./<dir>
#   inpackage      kerneltests/scripts/run-inpackage.sh
# A directory named _inpackage, or below one, holds the kernel tests that read the kernel's unexported
# state: Go compiles them only inside blis-substrate's kernel package, so `go test` of that directory
# fails on an internal import whatever the change. They are run once, by the script, however many files
# changed there. Every other directory is tested directly.
set -euo pipefail
inpackage=0
while IFS= read -r d; do
	[[ -n "$d" ]] || continue
	case "/$d/" in
	*/_inpackage/*) inpackage=1 ;;
	*) echo "direct $d" ;;
	esac
done
[[ "$inpackage" == 1 ]] && echo "inpackage"
exit 0
