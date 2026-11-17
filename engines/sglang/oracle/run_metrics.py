# Original to blis-sglang.
"""Tier-6 metrics runner (S15): SGLang v0.5.23's own Scheduler on CPU with --enable-metrics, and after
every loop iteration every `sglang:*` sample of the process's Prometheus registry, plus the KV-event
batches the scheduler published, as msgpack bytes.

It is oracle/run_sched.py with three additions to each step of its trace:

- "metrics": [[name, {labels}, value], ...] for every sample of the registry whose name starts with
  "sglang:", sorted, as read when the next iteration starts to receive (so it is what the step's
  reporting left). Series of a wall-clock instant (`_created`), the process's real CPU time and the label that
  names this machine's model path are left out;
- "kv_events": the hex of each msgpack payload the KV-event publisher was asked to send in the step.
  A scenario with `"kv_events_config": "{\\"publisher\\": \\"recording\\"}"` among its server_args
  publishes through a publisher that keeps the bytes in place of the ZMQ socket (oracle/mocks.json);
  its batches carry timestamp 0.0.

A scenario's `server_args` may also name `enable_metrics`, `decode_log_interval` and
`kv_events_config` (EXTRA_SERVER_ARGS). Nothing else differs from run_sched.py.

Usage: python -I oracle/run_metrics.py <sglang checkout> <scenario.json> [out.json]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_sched  # noqa: E402

EXTRA_SERVER_ARGS = ("enable_metrics", "decode_log_interval", "kv_events_config")
DROPPED_SUFFIXES = ("_created",)
DROPPED_LABELS = ("model_name",)
# Real process CPU time (a psutil reading): it differs between two runs of one scenario.
DROPPED_SERIES = ("sglang:scheduler_process_cpu_seconds_total",)
PUBLISHED = []  # the payloads (hex) published since the last step began


def registry_samples() -> list:
    from prometheus_client import REGISTRY

    out = []
    for fam in REGISTRY.collect():
        for s in fam.samples:
            if s.name.startswith("sglang:") and not s.name.endswith(DROPPED_SUFFIXES) and s.name not in DROPPED_SERIES:
                labels = {k: v for k, v in sorted(s.labels.items()) if k not in DROPPED_LABELS}
                out.append([s.name, labels, None if s.value != s.value else s.value])  # NaN is not JSON
    out.sort(key=lambda r: (r[0], sorted(r[1].items())))
    return out


def setup(scenario: dict) -> None:
    """Register the recording publisher (in the scheduler's process, before it starts)."""
    import msgspec
    from sglang.srt.disaggregation.kv_events import EventPublisher, EventPublisherFactory

    class RecordingPublisher(EventPublisher):
        def __init__(self, attn_dp_rank: int, **_unused) -> None:
            self._dp_rank = attn_dp_rank
            self._pack = msgspec.msgpack.Encoder()

        def publish(self, events) -> None:
            events.ts = 0.0
            if events.attn_dp_rank is None:
                events.attn_dp_rank = self._dp_rank
            PUBLISHED.append(self._pack.encode(events).hex())

        def shutdown(self) -> None:
            return None

    EventPublisherFactory.register_publisher("recording", RecordingPublisher)


def config(s, cfg: dict) -> dict:
    from sglang.srt.runtime_context import get_observability

    cfg["context_len"] = int(s.model_config.context_len)
    from sglang.srt.disaggregation.kv_events import uses_dynamo_format

    cfg["enable_kv_events"] = bool(s.kv_events_publisher.enable_kv_cache_events)
    cfg["kv_events_dynamo"] = bool(uses_dynamo_format(s.server_args.kv_events_config))
    cfg["enable_metrics"] = bool(get_observability().enable_metrics)
    cfg["decode_log_interval"] = int(s.metrics_reporter.decode_log_interval)
    return cfg


def extend(Base):
    class MetricsRun(Base):
        def begin_step(self):
            # What the step that just ended left, read before the next step's record is opened.
            if self.steps and self.step >= 0:
                if "enable_metrics" in self.scenario.get("server_args", {}):
                    self.steps[-1]["metrics"] = registry_samples()
                if PUBLISHED:
                    self.steps[-1]["kv_events"] = list(PUBLISHED)
                    PUBLISHED.clear()
            return super().begin_step()

    return MetricsRun


run_sched.HOOKS.update(server_args=EXTRA_SERVER_ARGS, inputs=("oracle/run_metrics.py",), run=extend, config=config, setup=setup)

if __name__ == "__main__":
    sys.exit(run_sched.main())
