-- One JSON object per InferenceX benchmark row, in row-id order: the inputs the ingest tool
-- reads (package ledger/inferencex). Run it against a database restored from the pinned dump by
-- restore.sh. github_run_id, config_id and run_attempt are read to rebuild the point
-- identifiers of the legacy partition; run_attempt and run_conclusion are also recorded on
-- each record. jsonb prints each metric's number as the dump stores it, and the
-- measured-value hash uses that text, so it does not depend on a float formatter.
select json_build_object(
  'id', br.id,
  'github_run_id', wr.github_run_id,
  'run_url', wr.html_url,
  'run_attempt', wr.run_attempt,
  'run_conclusion', wr.conclusion,
  'config_id', br.config_id,
  'model', c.model,
  'hardware', c.hardware,
  'precision', c.precision,
  'framework', c.framework,
  'benchmark_type', br.benchmark_type,
  'conc', br.conc,
  'error', br.error,
  'date', br.date,
  'recipe_fingerprint', br.recipe_fingerprint,
  'image', br.image,
  'isl', br.isl,
  'osl', br.osl,
  'offload_mode', br.offload_mode,
  'spec_method', c.spec_method,
  'disagg', c.disagg,
  'is_multinode', c.is_multinode,
  'prefill_tp', c.prefill_tp,
  'prefill_ep', c.prefill_ep,
  'decode_tp', c.decode_tp,
  'decode_ep', c.decode_ep,
  'prefill_dp_attention', c.prefill_dp_attention,
  'decode_dp_attention', c.decode_dp_attention,
  'prefill_num_workers', c.prefill_num_workers,
  'decode_num_workers', c.decode_num_workers,
  'num_prefill_gpu', c.num_prefill_gpu,
  'num_decode_gpu', c.num_decode_gpu,
  'metrics', br.metrics
)::text
from benchmark_results br
join configs c on c.id = br.config_id
join workflow_runs wr on wr.id = br.workflow_run_id
order by br.id;
