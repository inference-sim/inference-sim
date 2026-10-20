#!/bin/bash
# chunk.sh <series|logs> <idfile>: per-result diagnostics from the public InferenceX Data API for the ids in
# <idfile>, cached under $IX_API (default ./ix-data/inferencex-api/per-result). Bodies are UNTRUSTED data:
# never executed; the file list is parsed with python -I.
set -u
here=$(cd "$(dirname "$0")" && pwd); F=$here/fetch.sh
E=${IX_API:-${IX_DATA_DIR:-./ix-data}/inferencex-api/per-result}
mode=$1
while read -r id; do [ -z "$id" ] && continue
  case $id in *[!0-9]*) echo "skip non-numeric id $id" >&2; continue;; esac
  if [ "$mode" = series ]; then
    $F $E/trace-server-metrics tsm_$id "/api/v1/trace-server-metrics?id=$id"
    $F $E/request-timeline rt_$id "/api/v1/request-timeline?id=$id"
    $F $E/server-log-files slf_$id "/api/v1/server-log-files?id=$id"
  else
    b=$E/server-log-files/slf_$id.body
    [ -s "$b" ] || $F $E/server-log-files slf_$id "/api/v1/server-log-files?id=$id"
    for fn in $(uv run --no-project python -I -c "import json,sys,re;d=json.load(open(sys.argv[1]));print(' '.join(x for x in d if isinstance(x,str) and re.fullmatch(r'[A-Za-z0-9._/-]+',x) and '..' not in x) if isinstance(d,list) else '')" "$b" 2>/dev/null); do
      $F $E/server-log "sl_${id}__$(echo "$fn" | tr '/' '_')" "/api/v1/server-log?id=$id&file=$fn&download=1"
    done
  fi
done < "$2"
