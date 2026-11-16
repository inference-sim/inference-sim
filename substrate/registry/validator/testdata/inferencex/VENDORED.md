# Vendored InferenceX golden acceptance curves

`golden_al_distribution/*.yaml` are the curve files and `identities.yaml` of
[SemiAnalysis InferenceX]'s `inferencex-e2e/infx/golden_al_distribution/`, copied verbatim
at commit `67c3ed26f9c196b3e8286053a44838b4f0024756` (2026-11-16). InferenceX is
Apache-2.0 licensed; the files are unmodified. The directory's Python package and READMEs
are not copied: the emitter reads none of them.

They are the only input `scripts/emit_acceptance_profiles.py` reads values from. Each curve
file states the Actions run that measured it and what it ran under, and `identities.yaml`
states how each was collected; the emitter cites each file at this commit, with its
SHA-256. Vendoring them lets `emit_acceptance_profiles.py --check` run in every CI job, not
only one with network access.

## Updating

`.github/workflows/acceptance-refresh.yml` does this on a schedule and opens a pull request
when the output changes. By hand: re-copy every `*.yaml` from an upstream checkout, change
the commit above, `INFERENCEX_SHA` in the emitter and in `.github/workflows/rederive.yml`,
and re-run the emitter. A new curve is carried with no script edit when its model and
drafter resolve in the catalog; one that does not is named in the set's caveats with the
reason.

## Kept in step with CI

`.github/workflows/rederive.yml` checks out InferenceX at `INFERENCEX_SHA` and fails if
its curve files differ from this copy in content or in number.

[SemiAnalysis InferenceX]: https://github.com/SemiAnalysisAI/InferenceX
