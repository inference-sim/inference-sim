#!/bin/bash
# db_export.sh: export from the restored dump ($IX_PGDATABASE) the four inputs build_inventory.py and
# reconstruct.py read, into $IX_DATA_DIR/db: rows.tsv (benchmark rows x configs x workflow runs),
# logs.tsv and logs2.tsv (fields parsed from server logs; logs2 batches of 300 ids, retried one by one on
# the dump's oversized-log errors) and heads.tsv (github_run_id, head_sha). These are the exact queries
# behind inventory sha256 678f244a.
set -uo pipefail
DATA=${IX_DATA_DIR:-./ix-data}; O=$DATA/db; DB=${IX_PGDATABASE:-inferencex}
mkdir -p "$O"; O=$(cd "$O" && pwd)
psql -d "$DB" -q -c "\\copy (select br.id, br.workflow_run_id, wr.github_run_id, wr.run_attempt, wr.head_sha, wr.html_url, c.id, c.model, c.hardware, c.framework, c.precision, c.spec_method, c.disagg, c.is_multinode, c.prefill_tp, c.prefill_ep, c.prefill_dp_attention, c.prefill_num_workers, c.decode_tp, c.decode_ep, c.decode_dp_attention, c.decode_num_workers, c.num_prefill_gpu, c.num_decode_gpu, br.benchmark_type, br.date, br.isl, br.osl, br.conc, br.image, (br.error is not null), br.offload_mode, br.server_log_id, br.recipe_fingerprint, br.metrics->>'kv_offload_backend', br.metrics->>'mean_tpot', br.metrics->>'mean_ttft', (br.metrics ? 'kv_cache_pool_tokens'), br.trace_replay_id from benchmark_results br join configs c on c.id=br.config_id join workflow_runs wr on wr.id=br.workflow_run_id) to '$O/rows.tsv' with (format text)"
psql -d "$DB" -q -c "\\copy (select sl.id, substring(sl.server_log from 'version ([0-9][0-9A-Za-z.+_-]*)'), (position('Launching Rust frontend' in sl.server_log)>0), substring(sl.server_log from 'snapshots/([0-9a-f]{40})'), substring(sl.server_log from 'speculative_config.: (\{[^}]*\})'), substring(sl.server_log from 'kv_connector=.([A-Za-z]+)'), substring(sl.server_log from 'non-default args: (\{[^\n]*)') from server_logs sl where sl.id in (select distinct server_log_id from benchmark_results where server_log_id is not null)) to '$O/logs.tsv' with (format text)" 2>>"$O/logs.err"
psql -d "$DB" -At -c "select distinct server_log_id from benchmark_results where server_log_id is not null order by 1" > "$O/all_ids.txt"
: > "$O/logs2.tsv"
T=$(mktemp -d); split -l 300 "$O/all_ids.txt" "$T/ids_"
for f in "$T"/ids_*; do
  ids=$(paste -sd, "$f")
  q="\\copy (select sl.id, substring(sl.server_log from 'vLLM[^\n]{0,60}version ([0-9][0-9A-Za-z.+_-]*)'), substring(sl.server_log from 'sglang[^\n]{0,40}version[^0-9]{0,5}([0-9][0-9A-Za-z.+_-]*)'), (position('Launching Rust frontend' in sl.server_log)>0), substring(sl.server_log from 'snapshots/([0-9a-f]{40})'), substring(sl.server_log from 'speculative_config.: (\{[^}]*\})'), substring(sl.server_log from 'kv_connector=.([A-Za-z]+)'), substring(sl.server_log from 'VLLM_USE_V2_MODEL_RUNNER[^\n]{0,10}'), substring(sl.server_log from 'non-default args: (\{[^\n]*)') from server_logs sl where sl.id in ($ids)) to stdout with (format text)"
  psql -d "$DB" -q -c "$q" >> "$O/logs2.tsv" 2>>"$O/logs2.err" || for id in $(cat "$f"); do psql -d "$DB" -q -c "${q/in ($ids)/in ($id)}" >> "$O/logs2.tsv" 2>>"$O/logs2.err"; done
done
rm -rf "$T"
psql -d "$DB" -At -F $'\t' -c "select distinct github_run_id, head_sha from workflow_runs order by 1" > "$O/heads.tsv"
wc -l "$O"/rows.tsv "$O"/logs.tsv "$O"/logs2.tsv "$O"/heads.tsv
