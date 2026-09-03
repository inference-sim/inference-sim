# Archon PR Review

Archon is an architecture analysis tool that extracts package-level dependency graphs from Go repositories and diffs them between commits. The `/archon-pr-review` CI command runs Archon on a pull request and reports whether the PR changes the architecture or is purely internal.

## How to Use

Comment on any PR:

```
/archon-pr-review
```

The GitHub Action runs automatically and posts the results as a PR comment. Re-running
`/archon-pr-review` **updates that same comment in place** rather than adding a new one, so
a long-running PR keeps one current report instead of a stack of near-identical ones.

Each round's full body is also written to its own workflow run's job summary. That is
retention-bound, not an archive: it disappears when the run is aged out under the
repository's retention policy. If you need a round preserved, quote it in a PR comment.

## What It Reports

### Empty Delta (No Architectural Change)

If the PR only changes internal implementation (file renames, function body rewrites, new files within existing packages without new exports), Archon reports:

> **No architectural change detected.** Internal-only PR — fast-track eligible.

If tests (invariants) were added, removed, or modified, those are listed separately — even when the structure is unchanged.

### Non-Empty Delta (Architecture Changed)

If the PR changes package boundaries, Archon reports three sections:

**1. Architectural Delta** — What structurally changed:

- `+ arrow A -> B [import]` — new dependency between packages
- `- arrow A -> B [import]` — removed dependency
- `+ surface pkg.Func` — package surface widened (new export)
- `- surface pkg.Func` — package surface narrowed (export removed)
- `+ box pkg` / `- box pkg` — new or removed package

**2. Blast Radius** — For each changed package, how many other packages depend on it (directly and transitively). High blast radius means a change there affects many things.

**3. Contract Evidence** — For each interface (contract) in the codebase, which implementations are covered by a contract test and which are not. Gaps are flagged.

## How It Works

1. Determines the merge-base and PR head SHA
2. Runs `archon-go delta --json` to detect structural changes
3. If empty: posts fast-track message
4. If non-empty: runs `archon-go delta` (human report), `archon-go impact` (blast radius per changed package), and `archon-go evidence` (contract test coverage)
5. Posts combined output as a PR comment, editing the existing archon comment if one exists

## Key Properties

- **Witness-only changes are invisible.** A file rename or function body edit within a package does not appear in the delta. Only boundary-crossing changes (new dependencies, new/removed exports, new/removed packages) are reported.
- **Deterministic.** Same commits always produce the same output.
- **No LLM involved.** Pure static analysis. For LLM-interpreted results, see `/archon-pr-review-claude` ([#1540](https://github.com/inference-sim/inference-sim/issues/1540)).

## Requirements

- The repository must be a Go module
- The commenter must have write access to the repository
- Archon is built from [archon v0.1.0](https://github.com/AI-native-Systems-Research/archon/releases/tag/v0.1.0) during the CI run

## Related

- **Issue:** [#1541](https://github.com/inference-sim/inference-sim/issues/1541)
- **Future:** [#1540](https://github.com/inference-sim/inference-sim/issues/1540) — `/archon-pr-review-claude` feeds Archon output to Claude for a reasoned architectural review
