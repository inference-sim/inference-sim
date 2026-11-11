# The data ledger

This directory is the data ledger of design §10. It lists every external data unit BLIS
uses, and holds the rule that gives each unit its role. The code that reads and checks it is
the Go package `ledger/` at the repository root, with its command `go run ./ledger/cmd/ledger`.

A unit's role is computed when it is needed, never stored. The only exceptions are the three
lists the release job publishes, and the FIT and HELD sides of the legacy table (D4). The
check `ledger scan-roles` refuses any other committed file that pairs a unit with a role.

## Files

| File | What it holds | Written by |
|---|---|---|
| `ledger.yaml` | the partition version, the seed's SHA-256 and who made it, and each source's cut-points | a ledger owner |
| `sources.yaml` | each source, pinned to an exact revision (D3), with its licence, the quoted evidence and a ledger owner's sign-off (X12, D6) | a ledger owner |
| `units/<source>.jsonl.gz` | the units: unit_id, role key, identifying fields, and each measurement's source record and the hash of its measured values. No measured value is committed. | the ingest tool |
| `reports/<source>.yaml` | the ingest report: records read, units, role keys, duplicates, merged and joined records, and every record left out, by reason | the ingest tool |
| `legacy-map.yaml` | the legacy table: each legacy unit's side, the units whose records carry both labels, and the new units whose role key holds legacy units | the ingest tool |
| `legacy-points.jsonl.gz` | each point of the old partition, its label and the units its records fell in | the ingest tool |
| `aliases/<source>-<n>.yaml` | alias tables, old to new unit_id and role key, after a change to how identities are computed | `ledger generate-aliases`, in the pull request that makes the change |
| `published/{fit,anchor,report}.yaml` | the three published lists | the release job, at each substrate release |

`published/` does not exist yet. The release job writes it once a ledger owner has created the
seed. Nothing else may write it.

A tree without the lists can merge, but it cannot be released. `ledger check -release catalog/ledger`
and the release tool (`tools/release`, at `-tag`) refuse a tree that lacks any of the three lists
or whose fit or anchor list holds no unit; the release job itself refuses to write such lists.
The report list may be empty, because the cut-points give the report role no range. The
fit list is what every fitted coefficient cites (D1) and the anchor list is what the dashboard
scores (COV-2), so a release without them has nothing to fit from or score.

No other file may sit in this directory. `ledger check` refuses one, so a role list cannot be
added under another name.

## The InferenceX source

The pinned revision is the InferenceX-app release `db-dump/2026-09-28`: 17 parts, each named in
`sources.yaml` with its SHA-256. To regenerate the unit list:

```sh
ledger/inferencex/restore.sh ix               # 34 GB download, checked part by part
psql -d ix -At -f ledger/inferencex/export.sql -o rows.jsonl
curl -fsSLO https://raw.githubusercontent.com/inference-sim/inference-sim/8e344596c35272e6f505df65c0935aeb5e0000ec/tools/validation/inferencex/partition/partition.v5.json.gz
go run ./ledger/cmd/ledger ingest-inferencex -rows rows.jsonl -partition partition.v5.json.gz catalog/ledger
```

`ledger check-generated` runs the same steps into memory and requires every generated file to
equal the committed one byte for byte (D4). It needs the dump, so it runs in the nightly
`gate-nightly` job of `.github/workflows/ledger.yml`, not on each pull request. Each pull request
runs `ledger check`, `ledger scan-roles`, `ledger lint-workflows` and `ledger check-base`, which
need no source data (`.github/workflows/ci.yml`, job `ledger`).

The rules the ingest tool applies are those of design §10.2:

- The identity is the source plus ten fields: model, hardware, precision, framework,
  benchmark_type, configuration_hash, recipe_fingerprint, conc, measured_date and
  benchmark_run. The configuration hash covers 17 serving fields, with the image's
  `@sha256:` digest removed. The digest is kept on the measurement as `image_digest`.
- The measured date of a record with a run id is the earliest date among all records of its
  benchmark run. A record with no run id keeps its own date.
- The role key is the source, model, hardware, precision, framework, benchmark_type and
  configuration_hash.
- A record is not a measurement when it reports an error or its metrics are empty. Such a
  record is left out and counted. The run's conclusion is recorded on the measurement and
  never removes it.
- Records with the same identity and the same metrics are one measurement. The record with
  the smallest `benchmark_results.id` is kept, and the others are its duplicates.
- Records with the same identity and different metrics are one unit that holds each
  measurement.
- The same metrics under two identities with one role key are one unit, under the identity
  of the smallest row id. With two different role keys, both records are left out.

## Changing how identities are computed

If a pull request changes the ingest tool so that some unit_ids or role keys change, the
units would be drawn again, and a held unit could become a fit unit. To prevent that, the
pull request regenerates the unit list from the same pinned revision. It then commits the
alias table generated by pairing the old and new identities of the same records:

```sh
git archive "$(git merge-base origin/main HEAD)" catalog/ledger | tar -x -C /tmp/base
go run ./ledger/cmd/ledger generate-aliases -source inferencex -reason "..." \
  -out catalog/ledger/aliases/inferencex-1.yaml /tmp/base/catalog/ledger catalog/ledger
```

`ledger check-base` requires every added alias table to equal the generated one. It refuses a
change that would give one new unit or role key two old role keys, which could carry two
roles; such a change needs a new partition version: the next number, with a new seed record
(`ledger check-base` accepts it; no one approves it in the pull request, and the seed itself is a
ledger owner's release-time step below). A change of
source revision and a change of identities go in separate pull requests, so the pairing is
always made on one revision.

## Creating the seed (a ledger owner's task)

No agent creates the seed or the environment. A ledger owner does it once per partition
version, in this repository and, with a seed of its own, in accuracy-dashboard (D7).

1. In the repository settings, create the environment `ledger`:
   - Deployment branches: selected branches only, with the single rule `main`.
   - No required reviewers. A reviewer rule would make every seed-holding job wait for a person.
     The branch policy keeps pull-request code out of the environment.
   - No other secret in it.
2. On a trusted machine, generate the seed and its hash. Do not echo the seed, keep it in
   shell history, or write it to a file:

   ```sh
   seed=$(openssl rand -hex 32)
   printf '%s' "$seed" | xxd -r -p | sha256sum      # the SHA-256 of the 32 bytes
   gh secret set LEDGER_SEED --env ledger --body "$seed"
   unset seed
   ```

3. In a pull request, record the hash and the facts in `ledger.yaml`: `seed.sha256`,
   `seed.generated_by` (the ledger owner's GitHub login), `seed.generated_on` (YYYY-MM-DD)
   and `seed.rotation_owner`. The review question "Was the new seed generated as the ledger
   records?" (D7) is guidance for the review agent. The seed-holding jobs refuse a secret whose
   SHA-256 is not the recorded one, so a record without its secret draws no role.
4. Run `.github/workflows/ledger.yml` with `job: release` on main. The job checks the seed
   against the recorded hash before it draws anything. Its artifact `published` holds
   `fit.yaml`, `anchor.yaml` and `report.yaml`. The release bot commits them under
   `published/` in the release pull request. When that happens, the change also adds the
   pattern `ledger/published/*.yaml` to `CatalogPatterns` and `catalog/ledger/published/*.yaml`
   to the `//go:embed` line of `data.go`: `go:embed` refuses a pattern that matches no file, so
   the pattern cannot come first. `TestPublishedListsAreEmbeddedWhenPresent` fails a tree that
   has the lists without the pattern, and `ledger check -release` fails a tree without them.
5. Check the result before tagging:

   ```sh
   go run ./ledger/cmd/ledger check -release catalog/ledger     # exit 0; fit and anchor nonempty
   go test -count=1 -run 'TestPublishedListsAreEmbeddedWhenPresent|TestEmbedPatterns' .
   ```

When a partition version is retired, publish its seed in the pull request that retires it,
so that anyone can check the split against the committed hash.

## The jobs

`.github/workflows/ledger-candidate.yml` holds the candidate job. It runs a pull request's
code with no secret, no environment and a read-only token. It predicts every unit not on the
published lists with no network, and uploads `predictions.json`, nothing else.

`.github/workflows/ledger.yml` holds the seed-holding jobs: release, select, error, gate and
gate-nightly. Each runs only from main, in the environment `ledger`. Each checks out main and
reads the predictions as untrusted input with a strict loader (`ledger.ReadPredictions`). Each
reduces its own role's units to aggregates over at least 10 units, and checks that its report
names no unit (`ledger.CheckReport`). Units on a legacy side, legacy or new, are reduced
apart and reported as separate figures marked `not_blind`, because the old labels and the
side rules are public. `ledger lint-workflows` refuses any change to that shape.

What the reports protect. A report names no unit and covers at least 10 units, and the units
it covers are fixed by the seed and the measured data, not by the candidate: the loader refuses
predictions that omit a handed unit, and an absent prediction scores as a full error. Select
and error reports exist only for main-branch candidates, so differencing across a pull
request's pushes is not possible; a candidate on main is a merged change, so the number of
evaluations is the number of merges. Held stays pre-registered and counted.

What is not in place yet:

- The operator-level predictor. The candidate job records every unit as not predicted, so
  the seed-holding jobs report nothing.
- The pre-registration check. Anyone with write access starts the gate job by dispatch on main,
  for a change whose pre-registration the review agent has checked, and records the evaluation
  count on the pull request (design §10.2).
- The consumers' records of units read. The nightly join (D2) fails until at least one
  consumer uploads an artifact named `unit-reads-<consumer>` in the format of `ledger.Reads`. The join reads only artifacts uploaded by a run on main of
  this repository; an artifact from a pull request run or a fork is ignored.

## The AISimulate end-to-end artifact

`kernel/scripts/extract_aisimulate_e2e.py` turns the `e2e-accuracy-web` artifact of NVIDIA's
`E2E Accuracy Matrix` workflow into a per-point corpus of relative TTFT and TPOT. The ledger pins
the artifact under source `aisimulate`, `artifacts[e2e-accuracy-web]` (D3):

| Field | Value |
|---|---|
| Run | `ai-dynamo/aisimulate` actions run 36849388864, attempt 1 |
| Evaluated commit | `cab77b9c7888b1c40df8cd3228f5e03c355ef23c` (branch `main`) |
| Artifact | `e2e-accuracy-web-0d6e4079e36703eb`, zip SHA-256 `a3cdf7b4937223a76e32ca028c5442ba1aab523e914d71d2244e5753140d9f5c` |
| File read | `summary.json`, SHA-256 `e1a380ad7b06ae7f27edc95357e24219f4003e3ed73a3f93ff0dcd192578a3a2` |
| Measured values | InferenceX release `db-dump/2026-09-28` (the artifact's `measurement_source_url`) |
| Kept by GitHub until | 2026-12-30; a copy is authentic when its SHA-256 matches |

The extractor reads the pin from `kernel/scripts/aisimulate_e2e.pin.json`, and
`TestExtractorPinEqualsTheLedgerEntry` requires that file to equal the ledger entry. It refuses a
`summary.json` whose SHA-256 differs, whose snapshot states another run id or attempt, another
evaluated commit (or "latest", or no commit record) or another release tag, whose campaign
status is not `complete`, that yields no point, or whose sweep and point counts differ from the
pin's. To reproduce:

```sh
gh api repos/ai-dynamo/aisimulate/actions/artifacts/11159236747/zip > a.zip
shasum -a 256 a.zip                                  # a3cdf7b4...d9f5c
mkdir x && (cd x && unzip ../a.zip)
python3 -I kernel/scripts/extract_aisimulate_e2e.py x/summary.json -o corpus.json
# 204 sweeps, 1066 points, 88 distinct deployments -> corpus.json
# (54 topologies skipped: fewer than three successful points)
shasum -a 256 corpus.json                            # 4128d4e8...fe97 (Python 3.14)
```

The run took 0.3 s and 28 MB on a laptop. The corpus is shape only: every value is
normalised to its sweep's lowest concurrency, so it says nothing about absolute latency. It is an
evaluation set, never fitted to. The predictions are NVIDIA's; the ground truth is InferenceX's,
and the run's overall conclusion is `failure` only because two release-branch jobs failed (the
job for `main` succeeded and the summary's campaign status is `complete`). To move to a newer
artifact, change the ledger entry and the pin file in one pull request. The artifact has no unit
list yet: it holds relatives, not InferenceX units, so no role is drawn for it.

## Other sources

AISimulate (`ai-dynamo/aisimulate` at `2f0f4cd8`) and the forward-pass dataset
(`nvidia/aisimulate-fpm-dataset` at `dbfcdd9e`) are listed in `sources.yaml` with their
licences, but they have no unit list yet. Both are reachable from CI, and their ingest tools
are not yet written. The forward-pass dataset's README states that it has no approved
redistribution licence, so its records are never committed.
