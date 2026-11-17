# InferenceX golden acceptance curves: the pin

`curves.json` pins the input of `scripts/emit_acceptance_profiles.py`: [SemiAnalysis
InferenceX]'s `inferencex-e2e/infx/golden_al_distribution/` at one commit, the last that changed
a curve, with the SHA-256 of every curve file and of `identities.yaml`. No InferenceX file is
committed (X12); only these identifiers and hashes, and the profiles derived from them.

`scripts/fetch_inferencex_curves.py` downloads the files at the pinned commit into `cache/`
(ignored by git) and fails on a file whose hash differs. The emitter reads them from there, or
from `$INFERENCEX_GOLDEN`, and refuses to run unless the files are exactly the pinned ones. Every
CI job that runs the acceptance tests fetches them first.

## Moving the pin

`.github/workflows/acceptance-refresh.yml` does it on a schedule and opens a pull request when
the output changes. By hand: check out InferenceX at the commit, then run
`scripts/refresh_inferencex_curves.py <checkout>/inferencex-e2e/infx/golden_al_distribution
<commit> <date>`. It rewrites `curves.json` and `INFERENCEX_SHA` in `.github/workflows/rederive.yml`
together and re-runs the emitter. A new curve is carried with no script edit when its model and
drafter resolve in the catalog; one that does not is named in the set's caveats with the reason.

## Kept in step with CI

`.github/workflows/rederive.yml` checks out InferenceX at `INFERENCEX_SHA` and runs the emitter
against that checkout, which fails unless its files are the ones `curves.json` pins.

[SemiAnalysis InferenceX]: https://github.com/SemiAnalysisAI/InferenceX
