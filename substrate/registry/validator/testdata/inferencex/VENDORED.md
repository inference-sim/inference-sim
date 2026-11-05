# Vendored InferenceX golden acceptance curves

`golden_al_distribution/*.yaml` are the eleven curve files of
[SemiAnalysis InferenceX]'s `inferencex-e2e/infx/golden_al_distribution/`, copied verbatim
at commit `9fa76586ed1ef66caadc6b95a9b1448acde94c6b` (2026-10-13). InferenceX is
Apache-2.0 licensed; the files are unmodified.

They are the only input `scripts/emit_acceptance_profiles.py` reads values from. Each file
states the Actions run that measured it and what it ran under; the emitter cites each file
at this commit. Vendoring them lets `emit_acceptance_profiles.py --check` run in every CI
job, not only one with network access.

## Updating

Re-copy every `*.yaml` from an upstream checkout, change the commit above and
`INFERENCEX_SHA` in the emitter, and re-run the emitter. A new file fails the emitter
until it is classified in its `CURVES` table, so a new curve is a decision, not a silent
addition or omission.

## Kept in step with CI

`.github/workflows/rederive.yml` checks out InferenceX at `INFERENCEX_SHA` and fails if
its curve files differ from this copy in content or in number.

[SemiAnalysis InferenceX]: https://github.com/SemiAnalysisAI/InferenceX
