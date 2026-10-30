# Original to blis-vllm.
"""Tier-2 oracle: drive vLLM's real Scheduler through a scenario and record every step.

The scenario (JSON, schema in oracle/README.md) fixes the scheduler configuration, a fixed KV
capacity and the requests with their prompts, arrival steps and assigned outputs. This runs
vLLM's own Scheduler (built by vLLM's own test helper, tests/v1/core/utils.py create_scheduler)
and replaces only the GPU: each step's ModelRunnerOutput is built from the scenario's assigned
output tokens, the "shared GPU mock". It patches no scheduler, KV-cache or connector code;
oracle/mocks.json lists exactly what is replaced, and check_mocks.py fails CI if that list
names decision code.

Output (JSON on stdout): provenance, then one record per step with what the Go side must
reproduce: tokens scheduled per request, preempted and finished ids, the free block count,
every running request's blocks by content (block hash and ref count, never block ids), prefix
hits, and the waiting/running counts.

Usage: python -I oracle/run_scheduler.py <vllm checkout> <scenario.json>
"""

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys


def layer_specs(cfg: dict) -> dict:
    """A model's KV cache layers as its attention modules build them, from the scenario's
    "kv_layers": runs of {"count", "spec", ...}. "full" is Attention's FullAttentionSpec and
    "sliding_window" its SlidingWindowSpec (vllm/model_executor/layers/attention/attention.py:
    606-677@5df63a72); "indexer" is a key-only MLAAttentionSpec with the INDEXER role, as
    MiniMax-M3's lightning indexer builds it (vllm/models/minimax_m3/common/indexer.py:145-154).
    Heads and head size are each run's own, so page sizes are the model's own ratios.

    DeepSeek V4's caches are the spec classes its modules build, with each run's own block size,
    tokens per state, state bytes and alignment: "mla" is the compressed MLAAttentionSpec of its
    attention and indexer key caches (vllm/models/deepseek_v4/attention.py:856-925,
    vllm/models/deepseek_v41/attention.py:1138-1240), "sliding_window_mla" SlidingWindowMLASpec
    (vllm/v1/attention/backends/mla/sparse_swa.py:116-140, vllm/models/deepseek_v4/compressor.py:
    179-193) and "circular_buffer" CircularBufferSpec (vllm/models/deepseek_v41/compressor.py:
    159-167). Their uint8 caches are uint8 here; a float32 cache states "elem_bytes": 4. The
    attention specs' caches are float32 unless a run states "elem_bytes" (2 is bfloat16)."""
    import torch  # noqa: E402
    from vllm.v1.kv_cache_interface import (  # noqa: E402
        CircularBufferSpec,
        FullAttentionSpec,
        MLAAttentionSpec,
        SlidingWindowMLASpec,
        SlidingWindowSpec,
        SparseCacheRole,
    )

    dtypes = {1: torch.uint8, 2: torch.bfloat16, 4: torch.float32}
    out = {}
    for r, run in enumerate(cfg["kv_layers"]):
        for i in range(run["count"]):
            common = {"block_size": cfg["block_size"], "num_kv_heads": run.get("num_kv_heads", 1),
                      "head_size": run.get("head_size", 1), "dtype": dtypes[run.get("elem_bytes", 4)]}
            kind = run["spec"]
            deepseek = {"block_size": run.get("block_size", cfg["block_size"]), "num_kv_heads": 1,
                        "head_size": run["head_size"] if "head_size" in run else 1,
                        "dtype": dtypes[run.get("elem_bytes", 1)]}
            if run.get("alignment"):
                deepseek["alignment"] = run["alignment"]
            if run.get("state_bytes"):
                deepseek["state_content_bytes"] = run["state_bytes"]
            if kind == "full":
                spec = FullAttentionSpec(**common)
            elif kind == "sliding_window":
                spec = SlidingWindowSpec(sliding_window=run["sliding_window"], **common)
            elif kind == "indexer":
                spec = MLAAttentionSpec(cache_role=SparseCacheRole.INDEXER, **common)
            elif kind == "mla":
                spec = MLAAttentionSpec(tokens_per_state=run.get("tokens_per_state", 1), **deepseek)
            elif kind == "sliding_window_mla":
                spec = SlidingWindowMLASpec(sliding_window=run["sliding_window"],
                                            bounded_replay=bool(run.get("bounded_replay")), **deepseek)
            elif kind == "circular_buffer":
                deepseek.pop("alignment", None)
                spec = CircularBufferSpec(head_size_v=0, **deepseek)
            else:
                raise ValueError(f"unknown KV layer spec {kind!r}")
            out[f"run{r}.layer{i}"] = spec
    return out


def layer_groups(cfg: dict, vllm_config) -> list:
    """vLLM's own grouping of the layers (get_kv_cache_groups, vllm/v1/core/kv_cache_utils.py:
    2159-2253@5df63a72), with the spec registry the engine registers before it groups
    (register_all_kvcache_specs, vllm/v1/core/single_type_kv_cache_manager.py:2633). The
    layers are named in run order. vLLM buckets layers by spec in order of first appearance and
    splits each bucket by stride, so only which spec comes first matters: a run order that
    starts with the model's first layer's spec (gpt-oss's layer 0 is windowed) groups as the
    model does."""
    import dataclasses  # noqa: E402

    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups  # noqa: E402
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs  # noqa: E402
    from vllm.v1.kv_cache_interface import SlidingWindowSpec  # noqa: E402

    from vllm.v1.attention.backends.utils import (  # noqa: E402
        _DEFAULT_LAYOUT_PREFERENCE,
        resolve_kv_cache_layout,
    )
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config  # noqa: E402
    from vllm.v1.kv_cache_interface import KVCacheConfig  # noqa: E402

    register_all_kvcache_specs(vllm_config)
    layers = layer_specs(cfg)
    extra_retained = max(0, vllm_config.num_prefill_lookahead_tokens - 1)
    for name, layer in layers.items():
        if isinstance(layer, SlidingWindowSpec):
            layers[name] = dataclasses.replace(layer, extra_retained_tokens=extra_retained)
    # The engine core resolves the KV cache layout from the backends' supported layouts before
    # it groups (vllm/v1/engine/core.py:292-301@5df63a72), and resolve_kv_cache_layout records it
    # on the cache config (vllm/v1/attention/backends/utils.py:348). These layers' backends state
    # no preference, so the workers report vLLM's default preference; DeepSeek V4's indexer
    # backend states block-outermost layouts (DeepseekV4IndexerBackend.supported_kv_cache_layouts,
    # vllm/v1/attention/backends/mla/indexer.py:347-351), which its workers report.
    supported = [m.name for m in _DEFAULT_LAYOUT_PREFERENCE]
    if any(run["spec"] in ("mla", "sliding_window_mla", "circular_buffer") for run in cfg["kv_layers"]):
        from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend  # noqa: E402
        from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts  # noqa: E402

        supported = [m.name for m in get_supported_kv_cache_layouts([DeepseekV4IndexerBackend])]
    resolve_kv_cache_layout(vllm_config, [supported], layers.values())
    groups = get_kv_cache_groups(vllm_config, layers)
    if cfg.get("kv_available_bytes"):
        # The pool vLLM sizes from that much KV memory over these groups: its own
        # get_kv_cache_config_from_groups (vllm/v1/core/kv_cache_utils.py:1588-1628@5df63a72) on the
        # groups it just built, as the engine core calls it per worker (get_kv_cache_configs,
        # :2687-2695). Recorded beside the trace; the scenario's num_blocks still runs the
        # scheduler, so no decision moves.
        from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups  # noqa: E402

        sized = get_kv_cache_config_from_groups(vllm_config, groups, cfg["kv_available_bytes"])
        pool = {"available_bytes": cfg["kv_available_bytes"], "num_blocks": sized.num_blocks,
                "kv_cache_layout": vllm_config.cache_config.kv_cache_layout}
        if cfg.get("kv_startup_check"):
            # The engine core's start-up check on the same layers and memory: vLLM's own
            # get_kv_cache_configs (vllm/v1/core/kv_cache_utils.py:2531-2690@5df63a72), whose
            # _check_enough_kv_cache_memory (:2636-2646) refuses a worker whose KV cache cannot hold
            # one max_model_len request across every group. The verdict, vLLM's message and the
            # max_concurrent_batches its sliding windows' admission caps were counted at
            # (max_in_flight_tokens, vllm/config/vllm.py:577-598) are recorded; the scenario's
            # num_blocks still runs the scheduler.
            from vllm.v1.core.kv_cache_utils import get_kv_cache_configs  # noqa: E402

            try:
                get_kv_cache_configs(vllm_config, [layer_specs(cfg)], [cfg["kv_available_bytes"]])
                pool["startup"] = {"refused": False}
            except ValueError as e:
                pool["startup"] = {"refused": True, "message": str(e)}
            pool["startup"]["max_concurrent_batches"] = vllm_config.max_concurrent_batches
        pool_sizes.append(pool)
    # The scheduler's view: a uniform-type group is represented by one of its layers' specs
    # (generate_scheduler_kv_cache_config, vllm/v1/core/kv_cache_utils.py:2276-2297@5df63a72).
    projected = generate_scheduler_kv_cache_config(
        [KVCacheConfig(num_blocks=cfg["num_blocks"], kv_cache_tensors=[], kv_cache_groups=groups)])
    return projected.kv_cache_groups


# What layer_groups sized for a scenario stating kv_available_bytes, for run_one's trace.
pool_sizes: list = []


def main() -> None:
    vllm_root, scenario_path = sys.argv[1], sys.argv[2]
    if os.path.isdir(scenario_path):
        # Batch mode: every *.json scenario in the directory, each trace written beside it as
        # <name>.trace.json. One interpreter for many scenarios, which a fuzz corpus needs.
        out_dir = sys.argv[3] if len(sys.argv) > 3 else scenario_path
        # Golden traces are never overwritten without review (docs/testing.md).
        if "testdata/golden" in os.path.abspath(out_dir).replace(os.sep, "/") and \
                os.environ.get("BLIS_VLLM_ALLOW_GOLDEN_UPDATE") != "1":
            sys.exit("refusing to write golden traces: set BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1, and add "
                     "a golden-reviewed: <reviewer> trailer to the commit")
        names = sorted(f for f in os.listdir(scenario_path)
                       if f.endswith(".json") and not f.endswith(".trace.json"))
        for name in names:
            path = os.path.join(scenario_path, name)
            trace = run_one(vllm_root, path)
            with open(os.path.join(out_dir, name[:-5] + ".trace.json"), "w") as f:
                json.dump(trace, f, indent=1, sort_keys=True)
        return
    print(json.dumps(run_one(vllm_root, scenario_path), indent=1, sort_keys=True))


def run_one(vllm_root: str, scenario_path: str) -> dict:
    # vLLM's test helpers are imported from the pinned checkout itself, never copied; vLLM itself
    # from the installed distribution (the release wheel in CI, the editable build of this same
    # checkout locally). The checkout is appended, not prepended, so its vllm/ source tree (no
    # compiled ops, no generated version) cannot shadow the installed package.
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    # ModelConfig reads a model's config.json even though no weights are loaded. ORACLE_MODEL
    # names a local snapshot of the helper's default model at a pinned revision (see
    # oracle/README.md); its configuration does not reach any scheduling decision, which the
    # scenario's fixed num_blocks and block_size govern.
    model = os.environ.get("ORACLE_MODEL", "facebook/opt-125m")

    import tests.v1.core.utils as helpers  # noqa: E402
    from tests.v1.core.utils import EOS_TOKEN_ID, create_scheduler  # noqa: E402

    if not os.path.abspath(helpers.__file__).startswith(os.path.abspath(vllm_root) + os.sep):
        sys.exit("tests.v1.core.utils did not resolve into the vLLM checkout")
    from vllm.sampling_params import SamplingParams  # noqa: E402
    from vllm.utils.hashing import sha256  # noqa: E402
    from vllm.v1.core.kv_cache_utils import (  # noqa: E402
        get_request_block_hasher,
        init_none_hash,
    )
    from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
    from vllm.v1.request import Request  # noqa: E402

    with open(scenario_path) as f:
        sc = json.load(f)
    cfg = sc["config"]
    block_size = cfg["block_size"]

    scheduler = build_scheduler(helpers, create_scheduler, model, cfg)
    spec = cfg.get("speculative")
    # Draft tokens the mock proposes: K placeholder ids (any non-negative value works; only the
    # count reaches a scheduling decision, and acceptance is the scenario's input).
    num_spec = scheduler.num_spec_tokens

    init_none_hash(sha256)
    # Request block hashes are computed at the scheduler's hash_block_size, as the engine core
    # builds the hasher (vllm/v1/engine/core.py:233-240@5df63a72): the block size, or for groups
    # of several block sizes the one vLLM resolved for them.
    hasher = get_request_block_hasher(scheduler.hash_block_size, sha256)
    pending = sorted(sc["requests"], key=lambda r: (r["arrival_step"], r["id"]))
    # LoRA adapter ids, numbered from 1 in name order (vllm/lora/request.py:37-39 refuses 0).
    lora_ids = {n: i + 1 for i, n in enumerate(sorted({r["lora"] for r in sc["requests"] if r.get("lora")}))}
    accepted_plan = {r["id"]: list(r.get("accepted", [])) for r in sc["requests"]}
    verify_steps = {r["id"]: 0 for r in sc["requests"]}
    by_id = {r["id"]: r for r in sc["requests"]}
    sampled_so_far = {r["id"]: 0 for r in sc["requests"]}
    finished = {}
    steps = []
    step = 0
    max_steps = sc.get("max_steps", 10000)

    def add_arrivals(upto: int) -> None:
        while pending and pending[0]["arrival_step"] <= upto:
            r = pending.pop(0)
            # The assigned output ends where the model samples EOS (the mock below); a
            # client budget, when the scenario states one, is max_tokens.
            params = SamplingParams(max_tokens=r.get("max_tokens", 1 << 20))
            params.update_from_generation_config({}, EOS_TOKEN_ID)
            lora = None
            if r.get("lora"):
                from vllm.lora.request import LoRARequest  # noqa: E402

                lora = LoRARequest(lora_name=r["lora"], lora_int_id=lora_ids[r["lora"]],
                                   lora_path="/adapters/" + r["lora"])
            req = Request(
                request_id=r["id"],
                prompt_token_ids=list(r["prompt"]),
                sampling_params=params,
                pooling_params=None,
                arrival_time=float(r["arrival_step"]),
                priority=r.get("priority", 0),
                lora_request=lora,
                cache_salt=r.get("cache_salt"),
                block_hasher=hasher,
            )
            scheduler.add_request(req)

    kv_events = KVEventTap(scheduler) if cfg.get("kv_events") else None
    while step < max_steps:
        add_arrivals(step)
        if not scheduler.has_unfinished_requests() and not pending:
            break
        out = scheduler.schedule()
        record = {
            "step": step,
            "scheduled": dict(sorted(out.num_scheduled_tokens.items())),
            "preempted": sorted(out.preempted_req_ids or []),
        }
        # The shared GPU mock: a request whose step completes its known tokens samples its
        # next assigned output token. Whether it does is read from the scheduler's own state
        # (is_prefill_chunk, set in _update_after_schedule after schedule() advanced
        # num_computed_tokens), never decided here.
        req_ids, sampled = [], []
        record["spec_scheduled"] = {}
        for rid in out.num_scheduled_tokens:
            req = scheduler.requests[rid]
            req_ids.append(rid)
            if not req.is_prefill_chunk:
                drafts = out.scheduled_spec_decode_tokens.get(rid, [])
                n = 1
                if drafts:
                    record["spec_scheduled"][rid] = len(drafts)
                    plan = accepted_plan[rid]
                    i = verify_steps[rid]
                    verify_steps[rid] = i + 1
                    # A placeholder draft (-1, first-decode padding) is always rejected
                    # (vllm/v1/sample/rejection_sampler.py:762-763).
                    real = len([d for d in drafts if d >= 0])
                    n = 1 + min(plan[i] if i < len(plan) else 0, real)
                toks = []
                outp = by_id[rid]["output"]
                for _ in range(n):
                    k = sampled_so_far[rid]
                    if k >= len(outp):
                        break
                    # The last assigned token is sampled as EOS: that is how the assigned
                    # output length reaches the engine, which stops on it as on a real EOS.
                    toks.append(EOS_TOKEN_ID if k == len(outp) - 1 else outp[k])
                    sampled_so_far[rid] = k + 1
                sampled.append(toks)
            else:
                sampled.append([])
        if not record["spec_scheduled"]:
            del record["spec_scheduled"]
        mro = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        engine_outputs = scheduler.update_from_output(out, mro)
        if spec and out.total_num_scheduled_tokens > 0:
            # The drafter proposes for every request the step ran (the input batch); the
            # scheduler ignores proposals for prefill chunks (scheduler.py:2560-2564).
            from vllm.v1.outputs import DraftTokenIds  # noqa: E402

            ids = [rid for rid in out.num_scheduled_tokens if rid in scheduler.requests]
            scheduler.update_draft_token_ids(DraftTokenIds(ids, [[0] * num_spec for _ in ids]))
        done = []
        for eco in engine_outputs.values():
            for o in eco.outputs:
                if o.finish_reason is not None:
                    done.append(o.request_id)
                    finished[o.request_id] = {
                        "reason": str(o.finish_reason),
                        "output_tokens": sampled_so_far[o.request_id],
                        "step": step,
                    }
        record["finished"] = sorted(done)
        record.update(kv_state(scheduler))
        if kv_events is not None:
            record["kv_events"] = kv_events.drain()
        steps.append(record)
        step += 1

    out = {
        "provenance": provenance(vllm_root, scenario_path),
        "steps": steps,
        "finished": dict(sorted(finished.items())),
    }
    if pool_sizes:
        out["kv_pool"] = pool_sizes[-1]
        pool_sizes.clear()
    return out


def build_scheduler(helpers, create_scheduler, model: str, cfg: dict, extra: dict | None = None):
    """vLLM's real Scheduler for the scenario's configuration.

    vLLM's CPU test helper (tests/v1/core/utils.py:54 create_scheduler) builds the Scheduler;
    the configuration objects it does not expose (a speculative config other than ngram, a
    LoRA config, and the scheduler knobs it fixes or omits: watermark, max_num_active_seqs,
    max_num_scheduled_tokens, long_prefill_token_threshold_adaptive, KV events) are built from
    vLLM's own config classes and handed to the same constructors the helper calls. That is
    environment set-up, listed in oracle/mocks.json; no scheduler, KV-cache or config logic is
    replaced or edited, and every derived value (draft slots, lookahead, prefill lookahead,
    EAGLE block drop, the scheduled-token budget) is computed by vLLM itself.

    extra are create_scheduler's arguments a caller adds, such as the tier-3 oracle's
    async_scheduling, pipeline_parallel_size, use_v2_model_runner and scheduler_cls
    (oracle/run_enginecore.py); they override what this function would pass.
    """
    from vllm.config import KVEventsConfig, LoRAConfig, SpeculativeConfig  # noqa: E402

    spec = cfg.get("speculative")
    sched_extra = {k: cfg[k] for k in ("watermark", "max_num_active_seqs", "max_num_scheduled_tokens",
                                       "long_prefill_token_threshold_adaptive") if k in cfg}
    real_vllm_config, real_scheduler_config = helpers.VllmConfig, helpers.SchedulerConfig

    built = {}

    def vllm_config(**kw):
        if spec:
            args = {"method": spec["method"], "num_speculative_tokens": spec["num_speculative_tokens"]}
            if spec.get("draft"):
                # A configuration-only draft checkpoint (oracle/drafts/<name>/config.json): it
                # names the drafter's architecture, never weights.
                args["model"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drafts", spec["draft"])
            if spec.get("parallel_drafting"):
                args["parallel_drafting"] = True
            kw["speculative_config"] = SpeculativeConfig(
                target_model_config=kw["model_config"], target_parallel_config=kw["parallel_config"], **args)
        if cfg.get("max_loras"):
            kw["lora_config"] = LoRAConfig(max_loras=cfg["max_loras"], max_cpu_loras=cfg.get("max_cpu_loras"))
        if cfg.get("kv_events"):
            kw["kv_events_config"] = KVEventsConfig(enable_kv_cache_events=True, publisher="null")
        built["config"] = real_vllm_config(**kw)
        return built["config"]

    def scheduler_config(**kw):
        kw.update(sched_extra)
        return real_scheduler_config(**kw)

    real_kv_cache_config = helpers.KVCacheConfig
    groups = cfg.get("kv_groups")

    def kv_cache_config(num_blocks, kv_cache_tensors, kv_cache_groups):
        # A hybrid model's KV cache groups, from vLLM's own spec classes, one 1-head, 1-dim
        # float32 layer each, as the helper builds its single full-attention group; vLLM's
        # KVCacheManager picks the coordinator and per-group managers from them.
        if cfg.get("kv_layers"):
            # A dashboard model's attention layers, each with its own spec as the model builds it
            # (layer_specs), grouped by vLLM's own get_kv_cache_groups.
            return real_kv_cache_config(num_blocks=num_blocks, kv_cache_tensors=kv_cache_tensors,
                                        kv_cache_groups=layer_groups(cfg, built["config"]))
        if not groups:
            return real_kv_cache_config(num_blocks=num_blocks, kv_cache_tensors=kv_cache_tensors,
                                        kv_cache_groups=kv_cache_groups)
        import torch  # noqa: E402
        from vllm.v1.kv_cache_interface import (  # noqa: E402
            FullAttentionSpec,
            KVCacheGroupSpec,
            SlidingWindowSpec,
        )

        layers = {}
        for i, g in enumerate(groups):
            common = {"block_size": cfg["block_size"], "num_kv_heads": 1, "head_size": 1, "dtype": torch.float32}
            if g["kind"] == "full":
                layers[f"layer{i}"] = FullAttentionSpec(**common)
            elif g["kind"] == "sliding_window":
                layers[f"layer{i}"] = SlidingWindowSpec(sliding_window=g["sliding_window"], **common)
            else:
                raise ValueError(f"unknown KV cache group kind {g['kind']!r}")
        # A launch tags every sliding-window spec with extra_retained_tokens, the tokens a
        # multi-module MTP drafter may re-prefill (get_kv_cache_configs,
        # vllm/v1/core/kv_cache_utils.py:2582-2589@5df63a72), before it groups them; the count is
        # vLLM's own num_prefill_lookahead_tokens on the config the helper just built.
        extra_retained = max(0, built["config"].num_prefill_lookahead_tokens - 1)
        for name, layer in layers.items():
            if isinstance(layer, SlidingWindowSpec):
                layers[name] = dataclasses.replace(layer, extra_retained_tokens=extra_retained)
        if cfg.get("disable_hybrid_kv_cache_manager"):
            # disable_hybrid_kv_cache_manager (vllm/config/scheduler.py:182@5df63a72): vLLM
            # promotes every sliding-window layer to full attention before grouping
            # (get_kv_cache_groups, kv_cache_utils.py:2173-2174, unify_hybrid_kv_cache_specs,
            # :1799-1822), so all layers form one uniform group (:2185-2189). Both calls are
            # vLLM's own.
            from vllm.v1.core.kv_cache_utils import (  # noqa: E402
                _get_kv_cache_groups_uniform_spec,
                is_kv_cache_spec_uniform,
                unify_hybrid_kv_cache_specs,
            )

            unify_hybrid_kv_cache_specs(layers)
            assert is_kv_cache_spec_uniform(layers)
            specs = _get_kv_cache_groups_uniform_spec(layers)
        else:
            from vllm.v1.core.kv_cache_utils import (  # noqa: E402
                _get_kv_cache_groups_uniform_spec,
                is_kv_cache_spec_uniform,
            )

            # Layers that share one spec form one group, as get_kv_cache_groups gives them
            # (kv_cache_utils.py:2185-2189@5df63a72); otherwise each layer is its own group.
            if is_kv_cache_spec_uniform(layers):
                specs = _get_kv_cache_groups_uniform_spec(layers)
            else:
                specs = [KVCacheGroupSpec([name], spec) for name, spec in layers.items()]
        return real_kv_cache_config(num_blocks=num_blocks, kv_cache_tensors=kv_cache_tensors, kv_cache_groups=specs)

    v2 = (bool(spec) and spec["method"] == "dspark") or bool((extra or {}).get("use_v2_model_runner"))
    restore_env = v2_runner_env(v2)
    helpers.VllmConfig, helpers.SchedulerConfig = vllm_config, scheduler_config
    helpers.KVCacheConfig = kv_cache_config
    real_scheduler_cls = (extra or {}).get("scheduler_cls")
    if cfg.get("kv_layers") and any(r["spec"] in ("mla", "sliding_window_mla", "circular_buffer")
                                    for r in cfg["kv_layers"]):
        # Groups of several block sizes: the engine core hands the Scheduler the block sizes
        # vLLM's own resolve_kv_cache_block_sizes resolves from the groups, the scheduler block
        # size (their LCM) as block_size and hash_block_size beside it (vllm/v1/engine/core.py:
        # 170-182, 361-374@5df63a72), where the helper passes its one block_size. The class is
        # the one the helper would build; only those two arguments are the engine core's.
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes  # noqa: E402
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
        from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402

        base = real_scheduler_cls or (AsyncScheduler if (extra or {}).get("async_scheduling") else Scheduler)

        def engine_core_scheduler(*, vllm_config, kv_cache_config, block_size, **kw):
            sched_bs, hash_bs = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
            return base(vllm_config=vllm_config, kv_cache_config=kv_cache_config, block_size=sched_bs,
                        hash_block_size=hash_bs, **kw)

        extra = {**(extra or {}), "scheduler_cls": engine_core_scheduler}
    try:
        return create_scheduler(
            model=model,
            max_num_seqs=cfg["max_num_seqs"],
            max_num_batched_tokens=cfg["max_num_batched_tokens"],
            long_prefill_token_threshold=cfg.get("long_prefill_token_threshold", 0),
            enable_prefix_caching=cfg.get("enable_prefix_caching", False),
            num_blocks=cfg["num_blocks"],
            block_size=cfg["block_size"],
            max_model_len=cfg.get("max_model_len"),
            scheduling_policy=cfg.get("scheduling_policy", "fcfs"),
            **{"use_v2_model_runner": v2 or None, **(extra or {})},
        )
    finally:
        helpers.VllmConfig, helpers.SchedulerConfig = real_vllm_config, real_scheduler_config
        helpers.KVCacheConfig = real_kv_cache_config
        restore_env()


def v2_runner_env(v2: bool):
    """The environment of a GPU host that selects vLLM's V2 model runner, which DSpark needs
    (vllm/config/vllm.py:2976-2979@5df63a72) and which needs Triton on the host (:724, 3189): with
    v2, HAS_TRITON is set True and VLLM_USE_V2_MODEL_RUNNER=1 (oracle/mocks.json). The scheduler
    is the same class either way; selecting the V2 runner is the environment, not a decision
    (docs/testing.md). Returns the function that restores both. oracle/run_connector.py's
    drafter scenarios use it too."""
    import vllm.config.vllm as vllm_config_module  # noqa: E402

    saved_triton, saved_env = vllm_config_module.HAS_TRITON, os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    if v2:
        vllm_config_module.HAS_TRITON = True
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

    def restore():
        vllm_config_module.HAS_TRITON = saved_triton
        if saved_env is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = saved_env

    return restore


class KVEventTap:
    """The KV events vLLM's scheduler publishes each step, by content.

    The scheduler hands each step's events to its publisher (scheduler.py:2371-2385@5df63a72);
    with the "null" publisher configured they are dropped there. The tap wraps that publisher
    object's publish method to keep a copy (a transport, listed in oracle/mocks.json), so the
    events are exactly what vLLM would put on the wire.
    """

    def __init__(self, scheduler):
        self.batches = []
        pub = scheduler.kv_event_publisher
        original = pub.publish

        def publish(batch):
            self.batches.append(batch)
            return original(batch)

        pub.publish = publish

    def drain(self) -> list:
        """The events since the last drain, in publish order. A block hash is the integer
        vLLM puts on the wire by default (the low 64 bits of the hash, big-endian:
        VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES, vllm/envs.py:2033-2035, applied in
        kv_cache_utils.py:96-99), written as a decimal string so JSON keeps all 64 bits."""
        def h(v):
            return str(v) if isinstance(v, int) else v.hex()

        out = []
        for batch in self.batches:
            for ev in batch.events:
                name = type(ev).__name__
                if name == "BlockStored":
                    out.append(["stored", [h(x) for x in ev.block_hashes],
                                h(ev.parent_block_hash) if ev.parent_block_hash is not None else None,
                                list(ev.token_ids), ev.block_size, ev.lora_name])
                elif name == "BlockRemoved":
                    out.append(["removed", [h(x) for x in ev.block_hashes]])
                else:
                    out.append(["cleared"])
        self.batches = []
        return out


def kv_state(scheduler) -> dict:
    """The KV state compared by content: free blocks, and per running request the hash and
    reference count of each of its blocks, never the block id (block ids are an allocator
    detail two correct implementations may number differently)."""
    mgr = scheduler.kv_cache_manager
    pool = mgr.block_pool
    per_request = {}
    managers = mgr.coordinator.single_type_managers
    for rid in sorted(r.request_id for r in scheduler.running):
        rows = []
        for gi, m in enumerate(managers):
            blocks = m.req_to_blocks.get(rid, [])
            row = [[b.block_hash.hex() if b.block_hash is not None else None, b.ref_cnt] for b in blocks]
            if len(managers) == 1:
                rows = row
            else:
                # One list per group; the null block (a sliding window's freed slot) as "null".
                rows.append([["null", 0] if b.is_null else r for b, r in zip(blocks, row)])
        per_request[rid] = rows
    num_running, num_waiting = scheduler.get_request_counts()
    return {
        "free_blocks": pool.get_num_free_blocks(),
        "running": sorted(r.request_id for r in scheduler.running),
        "num_running": num_running,
        "num_waiting": num_waiting,
        "kv_usage": scheduler.get_kv_cache_usage(),
        "blocks": per_request,
    }


def blis_vllm_commit() -> str:
    """The blis-vllm commit a run checks: CI's GITHUB_SHA, else this checkout's HEAD (with
    "+dirty" when it has uncommitted changes), else "unknown"."""
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", here, "status", "--porcelain"], text=True,
                                        stderr=subprocess.DEVNULL).strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def committed_lock_sha256() -> str:
    """The sha256 of the committed oracle/requirements.lock: the environment CI installs and
    check_lock.py proves was installed exactly, so a trace names a lock in the repository."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(here, "requirements.lock"), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "unknown"


def scenario_seed(scenario_path: str):
    """The seed a generated (fuzz) scenario was made from, or None for a hand-written one."""
    with open(scenario_path) as f:
        return json.load(f).get("seed")


def provenance(vllm_root: str, scenario_path: str) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", vllm_root, *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    import vllm  # noqa: E402

    with open(scenario_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "mocks.json"), "rb") as f:
        mocks = json.load(f)
    # ORACLE_LOCKFILE names the frozen package list of the Python environment that ran vLLM
    # (CI writes it with pip freeze); its hash pins every package the run imported.
    lockfile = os.environ.get("ORACLE_LOCKFILE")
    lock_digest = "unrecorded"
    if lockfile:
        with open(lockfile, "rb") as f:
            lock_digest = hashlib.sha256(f.read()).hexdigest()
    return {
        "tier": 2,
        "vllm_version": vllm.__version__,
        # The installed distribution's version, e.g. "0.32.0+cpu" for the release CPU wheel.
        "vllm_distribution": importlib.metadata.version("vllm"),
        "vllm_commit": git("rev-parse", "HEAD"),
        "scenario_sha256": digest,
        "mock_version": mocks["version"],
        # The model's identity, never the local path ORACLE_MODEL points at.
        "model": "facebook/opt-125m",
        "model_revision": os.environ.get("ORACLE_MODEL_REVISION", "unpinned"),
        "mocks": [m["symbol"] for m in mocks["mocks"]],
        "python": sys.version.split()[0],
        "platform": platform.system().lower() + "-" + platform.machine(),
        "lockfile_sha256": lock_digest,
        "requirements_lock_sha256": committed_lock_sha256(),
        "blis_vllm_commit": blis_vllm_commit(),
        "seed": scenario_seed(scenario_path),
    }


if __name__ == "__main__":
    main()
