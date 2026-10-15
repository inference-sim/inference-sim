"""Dump aiperf 0.12.0's own reconstruction of a Weka v7 corpus, for BLIS's AgentX tests.

Runs aiperf's real loader and trajectory code (not a re-implementation):
WekaTraceLoader._build_reconstruction_plans / _reconstruct_serial /
_install_replay_dependencies, then TrajectorySource with the inferencex-agentx-mvp
ratios [0, 1] and a SequentialSampler over the roots. Token content is stubbed out: it
does not affect any field dumped here (timestamps, delays, api times, max_tokens,
branches, joins, replay predecessors, lane snapshots).

    uv run --no-project --python 3.12 --with aiperf-0.12.0-py3-none-any.whl \
        python -I gen_aiperf_reference.py CORPUS.jsonl MAX_CTX SEED C1[,C2...] > out.json
"""

import json
import sys
import types

import orjson

from aiperf.common.models import DatasetMetadata
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.dataset.loader import weka_trace as wt
from aiperf.dataset.loader._delay_cap import DelayCapTracker
from aiperf.dataset.loader.weka_trace_models import WekaTrace
from aiperf.timing.trajectory_source import TrajectorySource
from aiperf.plugin.enums import DatasetSamplingStrategy


def make_loader():
    ld = wt.WekaTraceLoader.__new__(wt.WekaTraceLoader)
    ld._max_osl = None
    ld._max_isl = None
    ld._fixed_schedule_start_offset = None
    ld._fixed_schedule_end_offset = None
    ld._user_block_size_override = None
    ld._block_size = 64
    ld._tool_shaped_messages = False
    ld._delay_cap_tracker = DelayCapTracker(cap_seconds=None)
    pg = types.SimpleNamespace(_cache={}, _bpe_stable_terminator_tokens=[], _hash_id_corpus_rng=types.SimpleNamespace(set_trace_id=lambda _: None))
    ld.prompt_generator = pg
    ld._decode_block_tokens = lambda hids: [1] * (len(hids) * ld._block_size)
    ld.sample_partial_tail_tokens = lambda n, seed: [2] * n
    ld._decode_tokens_to_text = lambda toks: ""
    return ld


def main():
    path, max_ctx, seed, cs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    data = {}
    for line in open(path, "rb"):
        if not line.strip():
            continue
        tr = WekaTrace.model_validate(orjson.loads(line))
        if wt._trace_peak_context_length(tr) > max_ctx:
            continue
        data[tr.id] = [tr]
    ld = make_loader()
    plans = ld._build_reconstruction_plans(data)
    mvals = ld._build_shared_metric_values(plans.parent_plans, plans.child_plans, plans.flat_plans)
    convs = ld._reconstruct_serial(
        parent_plans=plans.parent_plans, child_plans=plans.child_plans, data=data,
        dropped_per_trace={}, ignore_delays=False, think_time_only=False, cap_seconds=None,
        t_start=0.0, model_map_per_trace={}, trace_idle_timing_by_trace={},
        metric_values_by_trace=mvals, flat_plans=plans.flat_plans)
    wt._install_replay_dependencies(convs)

    out = {"pool": [c.session_id for c in convs if c.is_root], "conversations": []}
    for c in convs:
        out["conversations"].append({
            "session_id": c.session_id,
            "is_root": c.is_root,
            "scope": c.replay_scope_id,
            "branches": [{"id": b.branch_id, "children": list(b.child_conversation_ids),
                          "background": b.is_background, "start_ms": b.start_timestamp_ms}
                         for b in c.branches],
            "turns": [{"ts": t.timestamp, "delay": t.delay, "api": t.api_time_ms,
                       "max_tokens": t.max_tokens, "branch_ids": list(t.branch_ids),
                       "prereqs": [p.branch_id for p in t.prerequisites],
                       "preds": [[p.conversation_id, p.turn_index] for p in t.replay_predecessors]}
                      for t in c.turns],
        })

    meta = DatasetMetadata(conversations=[c.to_metadata() for c in convs],
                           sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL)
    roots = [c.session_id for c in convs if c.is_root]
    out["lanes"] = {}
    for conc in [int(x) for x in cs.split(",")]:
        src = TrajectorySource(dataset_metadata=meta, dataset_sampler=SequentialSampler(roots),
                               concurrency=conc, random_seed=seed, start_min_ratio=0.0,
                               start_max_ratio=1.0, cache_bust_enabled=True)
        lanes = []
        for tr in src.trajectories:
            snap = tr.snapshot
            lanes.append({
                "trace": tr.conversation_id,
                "t_star_ms": snap.t_star_ms.hex(),
                "states": [{"conv": s.conversation_id, "next": s.next_turn_index,
                            "offset_ms": s.next_dispatch_offset_ms.hex(),
                            "waiting": s.waiting_on_children, "join": s.join_target_turn_index,
                            "branch": s.branch_id, "depth": s.agent_depth} for s in snap.states],
                "boundaries": [[b.conversation_id, b.next_turn_index] for b in snap.replay_resume_boundaries],
            })
        out["lanes"][str(conc)] = {"lanes": lanes,
                                   "next_recycle": src.next_recycle_conversation_id()}
    json.dump(out, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
