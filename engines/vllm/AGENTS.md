# Agent instructions for blis-vllm

These rules apply to every AI-assisted change to this repository. Human contributors follow the
same rules; they are written here so an agent reads them first.

## Cross-repository rules (all BLIS repositories)

blis-vllm sits between blis-schemas (and blis-latency-kernel) and inference-sim. A change that
touches more than one repository follows
[Changes across the repositories](https://github.com/inference-sim/blis-schemas/blob/main/docs/contributing/cross-repo-changes.md).
In short:

1. **Public artifacts.** Never write internal project, customer or colleague names in a commit,
   code, comment, document, issue or pull request. Refer to private measurements by what they
   are ("internal deployment reports"). Run the name scan before every push.
2. **Data.** Never commit third-party data. Commit derived values only, each with its source URL
   plus commit or sha256.
3. **Versions.** Pin dependencies by release tag, or by the exact pushed head marked `TEMPORARY`
   with its pull request. Never commit a `replace` in `go.mod`.
4. **Tests.** Every behaviour change has a test that fails without it. Mutation-check it:
   revert the fix, see the test fail, restore, and say so in the commit message.

## This repository

- **One release.** A build models one vLLM release (`Upstream` in `provenance.go`). Every
  statement about vLLM cites `vllm/<path>:<lines>@<commit8>` at that release.
- **Ported files.** A Go file translated from vLLM keeps vLLM's licence and records its source:
  ```go
  // SPDX-License-Identifier: Apache-2.0
  // SPDX-FileCopyrightText: Copyright contributors to the vLLM project
  // Derived from vllm/v1/core/sched/scheduler.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
  ```
  `TestPortedFilesCarryAttribution` enforces it.
- **No clock, no goroutines.** Nothing reads `time.Now`, sleeps or starts a goroutine;
  `TestNoClockNoGoroutines` enforces it.
- **The oracle decides.** Behaviour is checked against vLLM's own classes in CI (`oracle/`).
  The oracle may patch only the GPU, the clock, transports and external services, never a
  decision; every patch is listed in `oracle/mocks.json`, and CI fails on a patch of decision
  code. Never overwrite a golden trace without human review (`docs/testing.md`).
- Tests: `go test ./...`; lint: `golangci-lint run ./...` (v2.14.0); the oracle tiers:
  `make oracle` (needs the pinned Python environment, `oracle/requirements.lock`).
