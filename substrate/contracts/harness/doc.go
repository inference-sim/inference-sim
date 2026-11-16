// Package harness holds the machinery every engine repository shares with every other: the checks
// an engine port runs in its own tests, the oracle's guard scripts, and the lint settings (design
// §5, §12.6). An engine repository imports it and nothing from another engine repository.
//
// It has four parts:
//
//   - Upstream, the record of the upstream release a repository models.
//   - ProvenanceProblems, the check that every Go file states where its logic comes from (VLM-2,
//     SGL-2).
//   - ClockAndGoroutineUses, the check that engine code reads no clock and starts no goroutine
//     (X6).
//   - The files an engine repository keeps as copies (Copies): the golangci-lint settings (for now
//     golangci-lint's default set; the X6 map-iteration and fused multiply-add lints are added by
//     inference-sim/inference-sim#2078's cross-platform determinism item), and the
//     oracle's lock check (FID-2), manifest check (FID-1, FID-8) and golden-trace check (FID-3).
//     WriteCopies writes them and StaleCopies reports a copy that differs from this version's
//     (X7); cmd/harnessfiles runs either from the command line.
//
// The oracle scripts read the engine's settings, such as its decision modules and golden
// directories, from oracle/harness.json in the engine repository. They name no engine, so the
// same scripts serve every engine (X2).
package harness
