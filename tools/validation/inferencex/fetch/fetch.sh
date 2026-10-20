#!/bin/bash
# fetch.sh <out_dir> <name> <path_and_query>: gentle cached GET of the public InferenceX Data API.
# Atomic: body lands as <name>.body only on HTTP 200 with a complete transfer; skipped when present.
set -u
out=$1; name=$2; q=$3; base=https://inferencex.semianalysis.com
mkdir -p "$out"
[ -s "$out/$name.body" ] && exit 0
tmp="$out/.$name.part"
code=$(curl -sS --compressed --max-time ${IXAPI_MAXTIME:-900} -A "${IX_USER_AGENT:-blis-inferencex-harness (research; cached)}" \
  -D "$out/$name.headers" -o "$tmp" -w '%{http_code}' "$base$q"); rc=$?
if [ $rc -eq 0 ] && [ "$code" = 200 ]; then mv "$tmp" "$out/$name.body"; sha=$(shasum -a 256 "$out/$name.body" | cut -d' ' -f1)
else rm -f "$tmp"; sha="-"; echo "FAIL rc=$rc HTTP $code $q" >&2; fi
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$code" "$name" "$base$q" "$sha" >> "$out/FETCHLOG.tsv"
[ "$code" = 429 ] && sleep 60
sleep ${IXAPI_SLEEP:-1.5}
