#!/bin/bash
# db_dump.sh: download the InferenceX DB dump db-dump/2026-09-28 (17 x ~2 GB, zstd pg_dump) from the public
# InferenceX-app release into $IX_DATA_DIR/db-dump, verify every part against the pinned sha256 below, and
# restore it into Postgres database $IX_PGDATABASE (default inferencex). Needs curl, zstd, createdb, pg_restore.
# The dump is UNTRUSTED data: it is only restored (pg_restore into a fresh database you own), never executed.
set -euo pipefail
DATA=${IX_DATA_DIR:-./ix-data}; OUT=$DATA/db-dump; DB=${IX_PGDATABASE:-inferencex}
BASE=https://github.com/SemiAnalysisAI/InferenceX-app/releases/download/db-dump/2026-09-28
mkdir -p "$OUT"
while read -r sha name; do
  f=$OUT/$name
  [ -s "$f" ] && [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" = "$sha" ] && continue
  curl -fL --retry 3 -o "$f.part" "$BASE/$name"
  got=$(shasum -a 256 "$f.part" | cut -d' ' -f1)
  [ "$got" = "$sha" ] || { echo "sha256 mismatch for $name: $got" >&2; rm -f "$f.part"; exit 1; }
  mv "$f.part" "$f"
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
[ "${1:-}" = "--no-restore" ] && exit 0
createdb "$DB"
cat "$OUT"/inferencex-2026-09-28.dump.zst.part?? | zstd -dc | pg_restore --no-owner --no-privileges -d "$DB"
