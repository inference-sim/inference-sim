#!/usr/bin/env bash
# Run the kernel tests that read the kernel's unexported state (kerneltests/_inpackage, package
# latencykernel) inside the kernel package, as they ran before they moved here.
#
# They open kernels from vLLM engine blocks, which blis-vllm decodes, so they live in this
# repository; and they read the kernel's unexported fields and functions, so Go compiles them
# only inside its package. This copies the kernel source this module pins (go list -m) into a
# scratch module, adds the tests to its root package, points it at this checkout for blis-vllm
# and at the kernel tests' testdata, and runs `go test` there. Nothing is edited in the kernel
# or in this checkout.
#
# Usage: kerneltests/scripts/run-inpackage.sh [go test flags...]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$(cd "$here/.." && pwd)"
kernel="$(cd "$repo" && go list -m -f '{{.Dir}}' github.com/inference-sim/blis-latency-kernel)"
[[ -n "$kernel" && -d "$kernel" ]] || { echo "run-inpackage: no kernel module source (go mod download first)" >&2; exit 1; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp -R "$kernel/." "$work/"
chmod -R u+w "$work"
# The pinned catalog, registry and fixtures the kernel tests read, from kerneltests/testdata.
rm -rf "$work/testdata"
cp -R "$here/testdata" "$work/testdata"
cp "$here"/_inpackage/*_test.go "$work/"
# The kernel's own go.mod check reads the kernel's go.mod, which the scratch module rewrites
# below; the kernel's CI runs it against the real one.
rm -f "$work/gomod_test.go"
cd "$work"
# The tests import blis-vllm (the vLLM rules packs, vllmsettings, the artifact check): use this
# checkout, at whatever it holds now.
go mod edit -require=github.com/inference-sim/blis-vllm@v0.0.0-00010101000000-000000000000 \
	-replace=github.com/inference-sim/blis-vllm="$repo"
GOFLAGS=-mod=mod GOWORK=off go mod tidy >/dev/null
GOFLAGS=-mod=mod GOWORK=off go test -count=1 "$@" .
