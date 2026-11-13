#!/usr/bin/env bash
# Behaviour of select-changed-tests.sh: a change under _inpackage selects the script and no go test of
# that directory; other directories are tested directly; nothing changed selects nothing.
set -euo pipefail
sel="$(dirname "${BASH_SOURCE[0]}")/select-changed-tests.sh"
fail() { echo "FAIL: $*" >&2; exit 1; }
got="$(printf '%s\n' kerneltests/_inpackage core/vllm | "$sel")"
[[ "$got" == $'direct core/vllm\ninpackage' ]] || fail "mixed change: got $(printf '%q' "$got")"
got="$(printf '%s\n' kerneltests/_inpackage | "$sel")"
[[ "$got" == "inpackage" ]] || fail "only _inpackage: got $(printf '%q' "$got")"
got="$(printf '%s\n' kerneltests/_inpackage/x kerneltests/_inpackage | "$sel")"
[[ "$got" == "inpackage" ]] || fail "two _inpackage dirs must select the script once: got $(printf '%q' "$got")"
got="$(printf '%s\n' kerneltests rules/engineconfig | "$sel")"
[[ "$got" == $'direct kerneltests\ndirect rules/engineconfig' ]] || fail "plain dirs: got $(printf '%q' "$got")"
got="$(printf '' | "$sel")"
[[ -z "$got" ]] || fail "nothing changed: got $(printf '%q' "$got")"
echo ok
