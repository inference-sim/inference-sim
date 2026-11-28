# Original to blis-sglang.
"""Scheduler runner for a hybrid (linear-attention) model: SGLang's own Scheduler on CPU over a hybrid
model's configuration, with the recurrent-state bookkeeping of each step recorded
(inference-sim/inference-sim#2366, #2442).

It is oracle/run_sched.py with a scenario that names its model by "model_dir" (a directory of
oracle/models/ holding the model's configuration, no weights; ORACLE_MODEL is not read) and may set
`max_mamba_cache_size`, `mamba_full_memory_ratio`, `mamba_radix_cache_strategy` and
`hybrid_cache_checkpoints_level` among its server_args. The model runner is the one oracle/mocks.json
lists (no weights, no kernel), so the scheduler builds its own hybrid pools and radix cache with the
state component and runs its event loop; what each forward pass would compute is not run.

Added to each forward pass of the trace: the batch's tracking arrays (the mask, positions and
lengths of the states it will leave, and the slots they go to). Added to each step: the state slot
allocator's free count, the tree's state ledgers, and per request of the running batch, the waiting
queue and the chunked request, its state slot, tracking slots and tracked lengths.

Usage: python -I oracle/run_sched_mamba.py <sglang checkout> <scenario.json> [out.json]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_sched  # noqa: E402

EXTRA_SERVER_ARGS = ("max_mamba_cache_size", "mamba_full_memory_ratio", "mamba_radix_cache_strategy",
                     "hybrid_cache_checkpoints_level", "mamba_track_interval")
KV_FIELDS = ("mamba_pool_idx", "mamba_ping_pong_track_buffer", "mamba_ping_pong_empty", "mamba_next_track_idx",
             "mamba_last_track_idx", "mamba_last_track_seqlen", "mamba_prev_track_seqlen", "mamba_cow_src_index",
             "mamba_needs_clear")
BATCH_FIELDS = ("mamba_track_mask_cpu", "mamba_prefill_track_mask_cpu", "mamba_track_seqlens_cpu",
                "mamba_track_buffer_indices", "mamba_track_indices", "mamba_track_mask", "mamba_track_seqlens",
                "mamba_track_mask_next_cpu", "mamba_cow_dst_indices")


def enc(x):
    import torch

    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    return str(x)


def setup(scenario: dict) -> None:
    """The model's configuration is the scenario's own: a directory of oracle/models/."""
    os.environ["ORACLE_MODEL"] = os.path.join(os.path.dirname(HERE), "oracle", "models", scenario["model_dir"])
    # extra_buffer needs a platform that can snapshot a recurrent state (CUDA, ROCm, XPU, MUSA, NPU:
    # arg_groups/mamba_hook.py:107-117). The base platform of the oracle has none; the snapshot is a
    # kernel the mocked model runner does not run, so the platform's answer is the CUDA image's.
    from sglang.srt.arg_groups import mamba_hook

    mamba_hook._platform_supports_mamba_cache_extra_buffer = lambda: True
    # The batch pins its tracking mask for an asynchronous host-to-device copy (schedule_batch.py:3373):
    # page-locked memory needs an accelerator, and the base platform has none. A pinned tensor has the
    # values of the tensor it pins, so the replacement returns it; the copy that follows is the
    # mocked device's.
    import torch

    torch.Tensor.pin_memory = lambda self, device=None: self


def req_state(r) -> dict:
    return {"rid": r.rid, **{k: enc(getattr(r.kv, k, None)) for k in KV_FIELDS},
            "branching": enc(getattr(r, "mamba_branching_seqlen", None))}


def mamba_state(s) -> dict:
    cache = s.tree_cache
    alloc = s.req_to_token_pool.mamba_allocator
    chunked = [s.chunked_req] if s.chunked_req is not None else []
    return {"mamba_available": int(alloc.available_size()),
            "tree_mamba_evictable": int(cache.tree_core.mamba_evictable_size()),
            "tree_mamba_protected": int(cache.tree_core.mamba_protected_size()),
            "running_mamba": [req_state(r) for r in s.running_batch.reqs],
            "waiting_mamba": [req_state(r) for r in s.waiting_queue],
            "chunked_mamba": [req_state(r) for r in chunked]}


def config(s, cfg: dict) -> dict:
    from sglang.srt.managers.schedule_policy import mamba_prefill_chunk_grid
    from sglang.srt.runtime_context import get_exec

    cfg["mamba_chunk_grid"] = int(mamba_prefill_chunk_grid(s.tree_cache))
    m = get_exec().mamba
    cfg["mamba"] = {"extra_buffer": bool(m.enable_mamba_extra_buffer), "lazy": bool(m.enable_mamba_extra_buffer_lazy),
                    "path_state_cap": int(m.mamba_path_state_cap), "checkpoints_level": int(m.hybrid_cache_checkpoints_level),
                    "mamba_size": int(s.req_to_token_pool.mamba_pool.size),
                    "ping_pong_size": int(s.req_to_token_pool.mamba_ping_pong_track_buffer_size),
                    "initial_tracking_slots": int(s.req_to_token_pool.mamba_initial_tracking_slots),
                    "track_interval": int(m.mamba_track_interval),
                    "state_chunk_size": int(getattr(s.model_config.hf_text_config, "mamba_chunk_size", 64)),
                    "tree_slot_interval": int(s.tree_cache.components[s.tree_cache.slot_id("state")].mamba_cache_chunk_size)}
    return cfg


def extend(Base):
    class MambaRun(Base):
        def begin_step(self):
            sched = run_sched.event_loop_scheduler()
            if sched is not None and self.step >= 0:
                self.mamba_after = mamba_state(sched)
            reqs = super().begin_step()
            if self.steps and len(self.steps) >= 2 and getattr(self, "mamba_after", None) is not None:
                self.steps[-2].setdefault("after", {})["mamba"] = self.mamba_after
            return reqs

        def record_forward(self, rec):
            batch = run_sched.frame_local("run_batch", "batch")
            if batch is not None:
                rec["mamba_batch"] = {k: enc(getattr(batch, k, None)) for k in BATCH_FIELDS if getattr(batch, k, None) is not None}
            super().record_forward(rec)

    return MambaRun


run_sched.HOOKS.update(server_args=EXTRA_SERVER_ARGS, inputs=("oracle/run_sched_mamba.py",), run=extend, config=config, setup=setup)

if __name__ == "__main__":
    sys.exit(run_sched.main())
