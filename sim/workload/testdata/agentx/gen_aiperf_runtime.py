"""Run aiperf 0.12.0's own AgentX timing stack in virtual time and dump what it sends.

Differential-conformance oracle for BLIS's agentic replay (#1937). Everything that
decides WHAT is sent and WHEN is aiperf's real code, configured by aiperf's real config
path for `aiperf profile --scenario inferencex-agentx-mvp`:

  * CLIConfig -> resolve_config -> build_benchmark_plan -> BenchmarkRun -> apply_scenario
    -> TimingConfig.from_run (the scenario lock: AGENTIC_REPLAY, ratios [0, 1], 10 s
    system idle cap, first_turn_prefix cache bust, synthesized warmup, 30 s grace);
  * WekaTraceLoader plan building + _reconstruct_serial + _install_replay_dependencies
    (chain splitting on), and ConversationReconstructor for every prompt;
  * PhaseOrchestrator -> PhaseRunner -> AgenticReplayStrategy, BranchOrchestrator,
    SessionTreeRegistry, ReplayBarrierCoordinator, CreditIssuer, CreditCallbackHandler,
    LoopScheduler, TrajectorySource (lanes, t*).

Only three seams are stubbed: (1) the credit router / transport, which returns each
credit after a fixed, deterministic service time instead of calling a server; (2) the
event loop clock, replaced by a virtual clock that jumps to the next timer (time.monotonic
and perf_counter are patched to it, so aiperf's own idle-cap arithmetic runs on it); (3)
token content: a block of hash id h is the tokens h*64+i and every synthetic tail token is
negative, so each dumped prompt reads back as its exact hash-block / tail layout. The
service time is service_s(in, out) = 0.5 + in/200000 + out/100 (in seconds), which
gen_aiperf_runtime's Go counterpart computes identically.

    uv run --no-project --python 3.12 --with aiperf-0.12.0-py3-none-any.whl \
        python -I gen_aiperf_runtime.py CORPUS.jsonl CONCURRENCY DURATION_S > out.json
"""

import asyncio
import hashlib
import itertools
import json
import selectors
import sys
import time
import types

import orjson

_PHASES = {}
_DEBUG = None

# ---------------------------------------------------------------- virtual clock
_VT = [1000.0]


def _vt():
    return _VT[0]


time.monotonic = _vt
time.perf_counter = _vt
time.monotonic_ns = lambda: int(_VT[0] * 1e9)
time.perf_counter_ns = lambda: int(_VT[0] * 1e9)
time.time = _vt
time.time_ns = lambda: int(_VT[0] * 1e9)


class _VirtualSelector(selectors.DefaultSelector):
    def select(self, timeout=None):
        ready = super().select(0)
        if ready:
            return ready
        if timeout is None:
            raise RuntimeError("virtual loop would block forever")
        _VT[0] += timeout
        return []


class VirtualLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__(_VirtualSelector())

    def time(self):
        return _VT[0]


from aiperf.cli_runner import _make_benchmark_run  # noqa: E402
from aiperf.common.models import DatasetMetadata  # noqa: E402
from aiperf.common.scenario import apply_scenario  # noqa: E402
from aiperf.config.flags import CLIConfig  # noqa: E402
from aiperf.config.flags.resolver import resolve_config  # noqa: E402
from aiperf.config.loader import build_benchmark_plan  # noqa: E402
from aiperf.config.resolution.resolvers import DatasetResolver  # noqa: E402
from aiperf.credit.messages import CreditReturn  # noqa: E402
from aiperf.dataset.loader import weka_synth_buf  # noqa: E402
from aiperf.dataset.loader import weka_trace as wt  # noqa: E402
from aiperf.dataset.loader._delay_cap import DelayCapTracker  # noqa: E402
from aiperf.dataset.loader.weka_trace_models import WekaTrace  # noqa: E402
from aiperf.plugin.enums import DatasetSamplingStrategy  # noqa: E402
from aiperf.timing.config import TimingConfig  # noqa: E402
from aiperf.timing.phase_orchestrator import PhaseOrchestrator  # noqa: E402

BS = 64
_tail = itertools.count(1)


def service_s(n_in, n_out):
    """Service time in seconds, rounded to the microsecond as BLIS's clock is."""
    return round((0.5 + n_in / 200000.0 + n_out / 100.0) * 1e6) / 1e6


def canonical_messages(segs):
    """One string per prompt: each message's role and content, block tokens grouped into
    h<hash>.<first offset>-<last offset+1> stretches and synthetic tail tokens into t<n>."""
    out = []
    for role, toks in segs:
        items = []
        for v in toks:
            if v >= 0:
                h, i = divmod(v, BS)
                if items and items[-1][0] == "h" and items[-1][1] == h and items[-1][3] == i:
                    items[-1][3] = i + 1
                else:
                    items.append(["h", h, i, i + 1])
            elif items and items[-1][0] == "t":
                items[-1][1] += 1
            else:
                items.append(["t", 1])
        body = ",".join(f"h{it[1]}.{it[2]}-{it[3]}" if it[0] == "h" else f"t{it[1]}" for it in items)
        out.append(f"{role[0]}:{body}")
    return "|".join(out)


def build_conversations(path):
    data = {}
    for line in open(path, "rb"):
        if line.strip():
            tr = WekaTrace.model_validate(orjson.loads(line))
            data[tr.id] = [tr]
    ld = wt.WekaTraceLoader.__new__(wt.WekaTraceLoader)
    ld._max_osl = ld._max_isl = None
    ld._fixed_schedule_start_offset = ld._fixed_schedule_end_offset = None
    ld._user_block_size_override = None
    ld._block_size = BS
    ld._tool_shaped_messages = False
    ld._delay_cap_tracker = DelayCapTracker(cap_seconds=None)
    ld.prompt_generator = types.SimpleNamespace(
        _cache={}, _bpe_stable_terminator_tokens=[],
        _hash_id_corpus_rng=types.SimpleNamespace(set_trace_id=lambda _: None))
    ld._decode_block_tokens = lambda hids: [h * BS + i for h in hids for i in range(BS)]
    ld.sample_partial_tail_tokens = lambda n, seed: [-next(_tail) for _ in range(n)]
    ld._decode_tokens_to_text = lambda toks: ""
    prompts = []
    orig = weka_synth_buf.ConversationReconstructor.turn_delta

    def turn_delta(self):
        prompts.append([(s.role, list(s.tokens)) for s in self._segments])
        return orig(self)

    weka_synth_buf.ConversationReconstructor.turn_delta = turn_delta
    plans = ld._build_reconstruction_plans(data)
    mv = ld._build_shared_metric_values(plans.parent_plans, plans.child_plans, plans.flat_plans)
    convs = ld._reconstruct_serial(
        parent_plans=plans.parent_plans, child_plans=plans.child_plans, data=data,
        dropped_per_trace={}, ignore_delays=False, think_time_only=False, cap_seconds=None,
        t_start=0.0, model_map_per_trace={}, trace_idle_timing_by_trace={},
        metric_values_by_trace=mv, flat_plans=plans.flat_plans)
    wt._install_replay_dependencies(convs)
    layout = {}
    it = iter(prompts)
    for c in convs:
        for k, _ in enumerate(c.turns):
            segs = next(it)
            canon = canonical_messages(segs)
            if _DEBUG and c.session_id.startswith(_DEBUG[0]) and k == _DEBUG[1]:
                print(canon[:150] + " ... " + canon[-150:], file=sys.stderr)
            layout[(c.session_id, k)] = (sum(len(t) for _, t in segs), len(segs),
                                         hashlib.sha256(canon.encode()).hexdigest()[:16])
    return convs, layout


class StubRouter:
    """The transport seam: returns each credit after service_s, cancels on request."""

    def __init__(self, layout, meta, log):
        self.layout, self.meta, self.log = layout, meta, log
        self.inflight = {}
        self.ret = None

    def set_return_callback(self, cb):
        self.ret = cb

    def set_first_token_callback(self, cb):
        pass

    async def send_credit(self, credit):
        n_in, n_msgs, msgs_sha = self.layout[(credit.conversation_id, credit.turn_index)]
        n_out = credit.max_tokens_override or self.meta[credit.conversation_id].turns[credit.turn_index].max_tokens
        rec = {"t": _VT[0], "phase": str(credit.phase), "conv": credit.conversation_id,
               "turn": credit.turn_index, "depth": credit.agent_depth,
               "root": credit.root_correlation_id or credit.x_correlation_id,
               "in": n_in, "out": n_out, "msgs": n_msgs, "msgs_sha": msgs_sha, "done": None, "cancelled": False}
        self.log.append(rec)
        loop = asyncio.get_running_loop()
        h = loop.call_later(service_s(n_in, n_out), self._finish, credit, rec, False)
        self.inflight[credit.id, str(credit.phase)] = (h, credit, rec)

    def _finish(self, credit, rec, cancelled):
        self.inflight.pop((credit.id, str(credit.phase)), None)
        rec["done"] = _VT[0]
        rec["cancelled"] = cancelled
        asyncio.ensure_future(self.ret("w0", CreditReturn(credit=credit, first_token_sent=not cancelled, cancelled=cancelled)))

    async def cancel_all_credits(self):
        for h, credit, rec in list(self.inflight.values()):
            h.cancel()
            self._finish(credit, rec, True)

    def mark_credits_complete(self):
        pass

    async def wait_for_workers(self, *args, **kwargs):
        return None


class StubPublisher:
    """Records each phase's start and completion (lifecycle started_at / requests_end)."""

    async def publish_phase_complete(self, phase_stats, *a, **k):
        if "PROFILING" in str(phase_stats.phase).upper():
            _PHASES["profiling_start_ns"] = phase_stats.start_ns
            _PHASES["profiling_requests_end_ns"] = phase_stats.requests_end_ns

    def __getattr__(self, name):
        async def _noop(*a, **k):
            return None
        return _noop


async def run(path, conc, dur):
    convs, layout = build_conversations(path)
    cli = CLIConfig(model_names=["m"], urls=["localhost:8000"], endpoint_type="chat", streaming=True,
                    scenario="inferencex-agentx-mvp", public_dataset="semianalysis_cc_traces_weka_062126_256k",
                    concurrency=conc, random_seed=42, benchmark_duration=dur, use_server_token_count=True)
    br = _make_benchmark_run(build_benchmark_plan(resolve_config(cli, None)).configs[0], random_seed=42)
    DatasetResolver().resolve(br)
    apply_scenario(br)
    tc = TimingConfig.from_run(br)
    meta = DatasetMetadata(conversations=[c.to_metadata() for c in convs],
                           sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL)
    log = []
    router = StubRouter(layout, {c.session_id: c for c in convs}, log)
    orch = PhaseOrchestrator(config=tc, phase_publisher=StubPublisher(), credit_router=router,
                             dataset_metadata=meta, run=br)
    t0 = _VT[0]
    src = orch.conversation_source
    lanes = [{"trace": tr.conversation_id, "t_star_ms": tr.snapshot.t_star_ms.hex()} for tr in src.trajectories]
    tree_of = {}
    for i, tr in enumerate(src.trajectories):
        tree_of[tr.snapshot.states[0].root_correlation_id] = i
    await orch._execute_phases()
    sends = []
    for r in log:
        if r["root"] not in tree_of:
            tree_of[r["root"]] = len(tree_of)  # recycled trees, in the order they first send
        sends.append({
            "t": r["t"] - t0, "phase": "warmup" if "warmup" in r["phase"] else "profile",
            "tree": tree_of[r["root"]],
            "conv": r["conv"], "turn": r["turn"], "depth": r["depth"], "in": r["in"], "out": r["out"],
            "msgs": r["msgs"], "msgs_sha": r["msgs_sha"],
            "done": None if r["done"] is None or r["cancelled"] else r["done"] - t0,
        })
    return {"concurrency": conc, "duration_s": dur, "grace_s": 30.0, "lanes": lanes, "sends": sends,
            "t0_ns": int(t0 * 1e9)}


def main():
    path, conc, dur = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    loop = VirtualLoop()
    asyncio.set_event_loop(loop)
    out = loop.run_until_complete(run(path, conc, dur))
    t0 = out.pop("t0_ns")
    for k, v in _PHASES.items():
        out[k.replace("_ns", "_s")] = None if v is None else (v - t0) / 1e9
    json.dump(out, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
