#!/usr/bin/env bash
# Run the kernel tests that read the kernel's unexported state (kerneltests/_inpackage, package
# latencykernel) inside blis-substrate's kernel package, as they ran before they moved here.
#
# They open kernels from vLLM engine blocks, which blis-vllm decodes, so they live in this
# repository; and they read the kernel's unexported fields and functions, so Go compiles them
# only inside its package. This copies the blis-substrate source this module pins (go list -m)
# into a scratch module, adds the tests to its kernel/ package, points it at this checkout for
# blis-vllm and at the kernel tests' fixtures, and runs `go test` there. Nothing is edited in
# the substrate or in this checkout.
#
# Usage: kerneltests/scripts/run-inpackage.sh [go test flags...]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$(cd "$here/.." && pwd)"
substrate="$(cd "$repo" && go list -m -f '{{.Dir}}' github.com/inference-sim/blis-substrate)"
[[ -n "$substrate" && -d "$substrate/kernel" ]] || { echo "run-inpackage: no blis-substrate module source (go mod download first)" >&2; exit 1; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp -R "$substrate/." "$work/"
chmod -R u+w "$work"
# The kernel's own tests stay as the substrate runs them; these run beside them, against the
# fixtures under kerneltests/testdata and the substrate's own catalog/ and registry/.
cp -R "$here/testdata" "$work/kernel/testdata"
for f in "$here"/_inpackage/*_test.go; do
	# A file named like one of the kernel's own tests goes beside it, not over it.
	cp "$f" "$work/kernel/vllm_$(basename "$f")"
done
cd "$work"
# The tests import blis-vllm (the vLLM rules packs, vllmsettings, the artifact check): use this
# checkout, at whatever it holds now. go mod tidy adds the requirement the replacement needs.
go mod edit -replace=github.com/inference-sim/blis-vllm="$repo"
GOFLAGS=-mod=mod GOWORK=off go mod tidy >/dev/null
GOFLAGS=-mod=mod GOWORK=off go test -count=1 "$@" ./kernel/
