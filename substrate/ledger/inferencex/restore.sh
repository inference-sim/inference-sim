#!/bin/bash
# restore.sh <database>: restore the three tables the InferenceX ingest reads (benchmark_results,
# configs, workflow_runs) from the pinned dump into a new PostgreSQL database.
#
# The dump is 17 parts of about 2 GB (34 GB in all). Each part is downloaded, checked against
# the SHA-256 the release's SHA256SUMS states (copied below and in ../../sources.yaml), and
# streamed into pg_restore, so at most one part is on disk at a time. The 6 GB of server logs
# are skipped. The dump is untrusted data: it is only restored into a database the caller
# owns, never executed.
#
# Needs curl, zstd, sha256sum (or shasum), createdb and pg_restore (PostgreSQL 17).
set -euo pipefail
db=${1:?usage: restore.sh <database>}
base=https://github.com/SemiAnalysisAI/InferenceX-app/releases/download/db-dump/2026-09-28
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
sum() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
fetch() {
  while read -r want name; do
    curl -fsSL --retry 3 -o "$work/part" "$base/$name"
    got=$(sum "$work/part")
    if [ "$got" != "$want" ]; then
      echo "restore.sh: $name has sha256 $got, want $want" >&2
      exit 1
    fi
    cat "$work/part"
    rm -f "$work/part"
  done <<'SUMS'
391a8eb04272f582a619d56b3e7d54f37bbc518b1588271ac79c48b9167a5218 inferencex-2026-09-28.dump.zst.part00
e19591b304b0a4ed7dc2a2e50ff2de669f7ef406433f2d6a25076d33e48dc56e inferencex-2026-09-28.dump.zst.part01
69bbf33c3e69883daacfff91185e7ab1948ae493a124d6d11ad4cb80a1f6bc50 inferencex-2026-09-28.dump.zst.part02
0fc485570acfbf6d33d8ce116723e14d8a333c0a86749e9ba97a88ec6fd0dd68 inferencex-2026-09-28.dump.zst.part03
1d011f90d6c7350bd0efcce0a621bc78933ed4383275149f69eb852331002942 inferencex-2026-09-28.dump.zst.part04
241dedad17e03857ea17319d43a27c02631fbeecb8a95fbdebca3d3710d9356e inferencex-2026-09-28.dump.zst.part05
36e8ab27a6789a4cfdfc1d4d7e8e36d812316820b45ed6241c9b11c6bb39144a inferencex-2026-09-28.dump.zst.part06
d5fafc5847800c11e4483768fcc01cca8a7f5551e09fe1a2e0f0ab2aa114fb08 inferencex-2026-09-28.dump.zst.part07
a13173e2c159af0b8caed5f369f2a66b1ffe4695da87d51d810087c66c930c80 inferencex-2026-09-28.dump.zst.part08
1baf3b1a0b91191524ebc18ca27a0167f9f1d6d69eba154267300cab961c842b inferencex-2026-09-28.dump.zst.part09
9204d96b8c33fbf4f2b0ef2feb474ea89d74866027273579875318d4e213fd56 inferencex-2026-09-28.dump.zst.part10
98de047e9bc7cdb89d5ab800a614f0b4b8720b640518b61db30facf47d572a54 inferencex-2026-09-28.dump.zst.part11
3d87bab8fdbf6df8bef808c5595c3e74abacac644947d362612b5784e24fc651 inferencex-2026-09-28.dump.zst.part12
ed3dbe7969ae4b4744f65d478ef282a6c7f10af0c47828bf5ae5e644638972c5 inferencex-2026-09-28.dump.zst.part13
5295bb87107352b98307e7a4055274a1efcaaada1f39643d66d646a2b8003e65 inferencex-2026-09-28.dump.zst.part14
d206b2d3c9678e985e6fc1a3f80067eedc421accce72937da753e2e54e833883 inferencex-2026-09-28.dump.zst.part15
c560093df925f66245bbb22c2e6833782f9c7083ccb7bfd2150da50791158882 inferencex-2026-09-28.dump.zst.part16
SUMS
}
createdb "$db"
# Without a seekable archive pg_restore reads the parts in order; the three tables' data is
# restored as their blocks pass. --no-owner and --no-privileges keep the dump's roles out.
fetch | zstd -dc | pg_restore --no-owner --no-privileges -d "$db" \
  -t benchmark_results -t configs -t workflow_runs
