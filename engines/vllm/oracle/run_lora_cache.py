# Original to blis-vllm.
"""Tier-2 oracle for the worker's LoRA adapter cache (max_cpu_loras, blis-vllm#5).

vLLM's scheduler never reads max_cpu_loras. The worker does: LRUCacheWorkerLoRAManager keeps an LRU of
loaded adapters of that capacity (vllm/lora/model_manager.py LRUCacheLoRAModelManager and
vllm/lora/worker_manager.py LRUCacheWorkerLoRAManager at v0.33.0), and an adapter missing from it is
loaded before its batch runs. This runner builds vLLM's real manager over the dummy LoRA model that
vLLM's own test suite uses (tests/lora/conftest.py DummyLoRAModel, tests/lora/utils.py create_peft_lora,
as tests/lora/test_lora_manager.py::test_lru_cache_worker_adapter_manager does), on CPU, and calls
set_active_adapters once per scenario step, as the model runner does once per scheduled batch
(vllm/v1/worker/lora_model_runner_mixin.py set_active_loras).

Nothing is patched. The runner subclasses the manager classes only to record what they do: it
overrides _load_adapter, remove_oldest_adapter, activate_adapter and _deactivate_adapter to note the
adapter and call super unchanged.

One input is the runner's, not the scenario's: vLLM hands set_active_adapters a Python set of
LoRARequest (make_lora_inputs, vllm/v1/worker/gpu_input_batch.py), whose iteration order follows
hash(lora_name), so it varies with PYTHONHASHSEED. The runner passes the requests as a list in the
scenario's order, as the unit test does, and the trace records that order; a Go engine replays it.

Usage: python -I oracle/run_lora_cache.py <vllm checkout> <scenario.json | dir> [out dir]
"""
import json
import os
import sys
import tempfile
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
_distributed = []  # vLLM's process group is initialised once per process


def run_one(vllm_root: str, scenario_path: str) -> dict:
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch  # noqa: E402
    import run_scheduler  # noqa: E402
    from tests.lora.conftest import DummyLoRAModel  # noqa: E402
    from tests.lora.utils import create_peft_lora  # noqa: E402
    from tests.utils import ensure_current_vllm_config  # noqa: E402
    from unittest.mock import MagicMock  # noqa: E402
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
    from vllm.config.lora import LoRAConfig  # noqa: E402
    from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402
    from vllm.lora.model_manager import LRUCacheLoRAModelManager  # noqa: E402
    from vllm.lora.request import LoRARequest  # noqa: E402
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager  # noqa: E402
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear  # noqa: E402
    from vllm.model_executor.layers.logits_processor import LogitsProcessor  # noqa: E402
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead  # noqa: E402
    from vllm.lora.layers import LoRAMapping  # noqa: E402

    with open(scenario_path) as f:
        sc = json.load(f)
    cfg = sc["config"]
    max_loras, max_cpu_loras = cfg["max_loras"], cfg.get("max_cpu_loras")

    events = []  # what the manager did during the current step, in order

    class Recording(LRUCacheLoRAModelManager):
        def remove_oldest_adapter(self):
            oldest = next(iter(self._registered_adapters.order), None)
            removed = super().remove_oldest_adapter()
            if removed:
                events.append(["evict", oldest])
            return removed

        def activate_adapter(self, lora_id):
            activated = super().activate_adapter(lora_id)
            if activated:
                events.append(["activate", lora_id])
            return activated

        def _deactivate_adapter(self, lora_id):
            if lora_id in self.lora_index_to_id:
                events.append(["deactivate", lora_id])
            return super()._deactivate_adapter(lora_id)

    class Worker(LRUCacheWorkerLoRAManager):
        _manager_cls = Recording

        def _load_adapter(self, lora_request):
            events.append(["load", lora_request.lora_int_id])
            return super()._load_adapter(lora_request)

    ids = {}  # adapter name -> lora_int_id, in order of first use, as blis-vllm numbers them

    def lora_id(name):
        return ids.setdefault(name, len(ids) + 1)

    with tempfile.TemporaryDirectory() as tmp, tempfile.NamedTemporaryFile() as dist_file:
        model_config = ModelConfig(model=os.environ.get("ORACLE_MODEL", "facebook/opt-125m"), max_model_len=16)
        lora_config = LoRAConfig(max_lora_rank=8, max_cpu_loras=max_cpu_loras, max_loras=max_loras,
                                 lora_dtype=torch.get_default_dtype())
        vllm_config = VllmConfig(model_config=model_config, lora_config=lora_config)
        with set_current_vllm_config(VllmConfig()), ensure_current_vllm_config():
            if not _distributed:
                init_distributed_environment(world_size=1, rank=0,
                                             distributed_init_method=f"file://{dist_file.name}",
                                             local_rank=0, backend="gloo")
                initialize_model_parallel(1, 1)
                _distributed.append(True)
            class Model(DummyLoRAModel):
                # what vLLM's dummy_model fixture sets after building it
                config = MagicMock()
                unpadded_vocab_size = 32064

            model = Model(OrderedDict([
                ("dense1", ColumnParallelLinear(764, 100)),
                ("dense2", RowParallelLinear(100, 50)),
                ("layer1", torch.nn.Sequential(OrderedDict([
                    ("dense1", ColumnParallelLinear(100, 10)), ("dense2", RowParallelLinear(10, 50))]))),
                ("act2", torch.nn.ReLU()),
                ("output", ColumnParallelLinear(50, 10)),
                ("outact", torch.nn.Sigmoid()),
                ("lm_head", ParallelLMHead(32064, 10)),
                ("logits_processor", LogitsProcessor(32064)),
            ]))
            files = os.path.join(tmp, "adapter")
            os.makedirs(files)
            create_peft_lora(model, save_dir=files, target_modules=["layer1.dense1", "dense2"],
                             lora_dtype=torch.get_default_dtype())
            worker = Worker(vllm_config, "cpu", {"embed_tokens": "input_embeddings", "lm_head": "output_embeddings"})
            worker.create_lora_manager(model, vllm_config)
            mgr = worker._adapter_manager

            steps = []
            for step in sc["steps"]:
                requested = []
                for n in step["adapters"]:
                    if lora_id(n) not in requested:
                        requested.append(lora_id(n))
                del events[:]
                err = None
                try:
                    worker.set_active_adapters([LoRARequest(str(i), i, files) for i in requested], LoRAMapping([], []))
                except RuntimeError as e:
                    err = str(e)
                steps.append({
                    "requested": requested,
                    "events": [list(e) for e in events],
                    "error": err,
                    # oldest first, the LRU's own order
                    "registered": list(mgr._registered_adapters.order),
                    "active": list(mgr._active_adapters.order),
                    "slots": list(mgr.lora_index_to_id),
                })
    trace = {"config": {"max_loras": max_loras, "max_cpu_loras": lora_config.max_cpu_loras},
             "adapter_ids": ids, "steps": steps}
    trace["provenance"] = run_scheduler.provenance(vllm_root, scenario_path)
    return trace


def main() -> None:
    vllm_root, scenario_path = sys.argv[1], sys.argv[2]
    if os.path.isdir(scenario_path):
        out_dir = sys.argv[3] if len(sys.argv) > 3 else scenario_path
        if "testdata/golden" in os.path.abspath(out_dir).replace(os.sep, "/") and \
                os.environ.get("BLIS_VLLM_ALLOW_GOLDEN_UPDATE") != "1":
            sys.exit("refusing to write golden traces: set BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1, and add "
                     "a golden-reviewed: <trace>: <reason> trailer for each trace the commit changes")
        for name in sorted(f for f in os.listdir(scenario_path)
                           if f.endswith(".json") and not f.endswith(".trace.json")):
            trace = run_one(vllm_root, os.path.join(scenario_path, name))
            with open(os.path.join(out_dir, name[:-5] + ".trace.json"), "w") as f:
                json.dump(trace, f, indent=1, sort_keys=True)
        return
    print(json.dumps(run_one(vllm_root, scenario_path), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
