# Original to blis-vllm.
"""Tests for gen_surface.py over a small synthetic vLLM tree and repository.

Run: python3 -I tools/surface/test_gen_surface.py
"""

import ast
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_surface  # noqa: E402

KC = "vllm/distributed/kv_transfer/kv_connector"
LOADER = """
        import importlib
        class {owner}:
            _registry = {{}}
            @classmethod
            def {method}(cls, name, module_path, class_name):
                def loader():
                    module = importlib.import_module(module_path)
                    return getattr(module, class_name)
                cls._registry[name] = loader
"""

FILES = {
    KC + "/factory.py": LOADER.format(owner="KVConnectorFactory", method="register_connector") + """
        def create_connector(kv_transfer_config):
            connector_module_path = kv_transfer_config.kv_connector_module_path
            if connector_module_path:
                connector_module = importlib.import_module(connector_module_path)
        KVConnectorFactory.register_connector("PullConnector", "vllm.conn.pull", "PullConnector")
        KVConnectorFactory.register_connector("OldConnector", "vllm.conn.pull", "OldConnector")
        KVConnectorFactory.register_connector("StoreConnector", "vllm.conn.store", "StoreConnector")
    """,
    "vllm/conn/pull/__init__.py": """
        from vllm.conn.pull.impl import PullConnector, OldConnector
    """,
    "vllm/conn/pull/impl.py": """
        class PullConnector:
            pass
        OldConnector = PullConnector
    """,
    "vllm/conn/store.py": """
        class StoreConnector:
            pass
    """,
    "vllm/v1/kv_offload/factory.py": LOADER.format(owner="OffloadingSpecFactory", method="register_spec") + """
        def get_spec_cls(extra_config):
            spec_module_path = extra_config.get("spec_module_path")
            spec_module = importlib.import_module(spec_module_path)
        OffloadingSpecFactory.register_spec("CPUSpec", "vllm.conn.store", "StoreConnector")
    """,
    "vllm/v1/kv_offload/tiering/factory.py": """
        import importlib
        def _import_class(name):
            if "." in name:
                module_path, _, class_name = name.rpartition(".")
            module = importlib.import_module(module_path)
    """ + LOADER.format(owner="SecondaryTierFactory", method="register_tier") + """
            @classmethod
            def get_tier_class(cls, tier_config):
                module_path = tier_config.get("module_path")
                module = importlib.import_module(module_path)
        SecondaryTierFactory.register_tier("fs", "vllm.conn.store", "StoreConnector")
    """,
    "vllm/v1/kv_offload/cpu/policies/factory.py": LOADER.format(owner="CachePolicyFactory", method="register_cache_policy") + """
            @classmethod
            def get_cache_policy_cls(cls, name, module_path=None):
                if module_path is None:
                    raise ValueError(name)
                module = importlib.import_module(module_path)
        CachePolicyFactory.register_cache_policy("lru", "vllm.conn.store", "StoreConnector")
    """,
    "vllm/v1/kv_offload/tiering/backpressure.py": """
        from abc import ABC, abstractmethod
        class BackpressurePolicy(ABC):
            @abstractmethod
            def drop(self): ...
        class DropStorePolicy(BackpressurePolicy):
            pass
    """,
    "vllm/v1/kv_offload/tiering/kvcr/manager.py": """
        def _resolve_policy(name):
            return resolve_obj_by_qualname(name)
    """,
    "vllm/v1/kv_offload/base.py": """
        from abc import ABC, abstractmethod
        class Medium(Enum):
            CPU = "CPU"
        class OffloadingManager(ABC):
            @abstractmethod
            def lookup(self): ...
        class Helper(ABC):
            def nothing_abstract(self): ...
    """,
    "vllm/v1/kv_offload/cpu/policies/base.py": "",
    "vllm/v1/kv_offload/tiering/base.py": "",
    KC + "/v1/base.py": """
        from abc import ABC, abstractmethod
        class KVConnectorBase_V1(ABC):
            # ==============================
            # Worker-side methods
            # ==============================
            @abstractmethod
            def start_load_kv(self): ...
            # ==============================
            # Scheduler-side methods
            # ==============================
            @abstractmethod
            def build_connector_meta(self): ...
    """,
    "vllm/config/cache.py": """
        KVOffloadingBackend = Literal["native"]
    """,
    KC + "/v1/simple_cpu_offload_connector.py": """
        VALID_KV_OFFLOAD_BACKENDS = ("cpu",)
    """,
    "vllm/distributed/kv_events.py": """
        class KVCacheEvent:
            pass
        class BlockStored(KVCacheEvent):
            pass
        class KVConnectorKVEvents(ABC):
            pass
        class EventPublisherFactory:
            _registry: dict = {
                "null": NullEventPublisher,
            }
            @classmethod
            def register_publisher(cls, name, ctor):
                cls._registry[name] = ctor
    """,
    "vllm/conn/events.py": """
        class StoreKVEvents(KVConnectorKVEvents):
            pass
    """,
    "vllm/v1/core/single_type_kv_cache_manager.py": """
        def register_all_kvcache_specs(vllm_config):
            KVCacheSpecRegistry.register(FullAttentionSpec, FullAttentionManager)
            KVCacheSpecRegistry.register_role_manager(KVCacheGroupRole.SOURCE, SourceManager)
            current_platform.register_custom_kv_cache_specs(vllm_config)
    """,
    "vllm/platforms/cuda.py": """
        class CudaPlatform:
            pass
    """,
    "vllm/config/speculative.py": """
        MTPModelTypes = Literal["deepseek_mtp", "mtp"]
        SpeculativeMethod = Literal["ngram", MTPModelTypes]
    """,
    "vllm/config/scheduler.py": """
        SchedulerPolicy = Literal["fcfs"]
        class SchedulerConfig:
            def get_scheduler_cls(self):
                return resolve_obj_by_qualname(self.scheduler_cls)
    """,
    "vllm/model_executor/models/registry.py": """
        _TEXT_GENERATION_MODELS = {
            "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
            "OlmoForCausalLM": ("olmo", "OlmoForCausalLM"),
        }
        _EMBEDDING_MODELS = {
            "BertModel": ("bert", "BertEmbeddingModel"),
            **{
                k: (mod, arch)
                for k, (mod, arch) in _TEXT_GENERATION_MODELS.items()
                if arch == "LlamaForCausalLM"
            },
        }
        _TRANSFORMERS_SUPPORTED_MODELS = {
            "GlmForCausalLM": ("transformers", "TransformersForCausalLM"),
            "EmuForConditionalGeneration": (
                "transformers",
                "TransformersMultiModalForCausalLM",
            ),
        }
        _TRANSFORMERS_BACKEND_MODELS: dict = {
            "TransformersForCausalLM": ("transformers", "TransformersForCausalLM"),
            "TransformersMultiModalForCausalLM": ("transformers", "TransformersMultiModalForCausalLM"),
        }
        _PREVIOUSLY_SUPPORTED_MODELS = {"GlmForCausalLM": {"max_transformers_version": "4.0"}}
        _VLLM_MODELS = {
            **_TEXT_GENERATION_MODELS,
            **_EMBEDDING_MODELS,
            **_TRANSFORMERS_SUPPORTED_MODELS,
            **_TRANSFORMERS_BACKEND_MODELS,
        }
        class _ModelRegistry:
            def _try_resolve_transformers(self, architecture, model_config):
                pass
    """,
    "vllm/v1/spec_decode/custom_class_proposer.py": """
        import importlib
        def load(module_path):
            module = importlib.import_module(module_path)
    """,
    "vllm/v1/worker/gpu/spec_decode/__init__.py": """
        def init_speculator(qualname):
            speculator_cls = resolve_obj_by_qualname(qualname)
    """,
}

def config_class(name, fields, classvars=""):
    body = "".join("    %s: object = %s\n" % (f, d) for f, d in fields)
    return "class %s:\n%s%s" % (name, classvars, body)


CACHE_FIELDS = [("block_size", "None"), ("gpu_memory_utilization", "0.9"), ("cache_dtype", '"auto"'),
                ("enable_prefix_caching", "True"), ("prefix_caching_hash_algo", '"sha256"'),
                ("num_gpu_blocks_override", "None"), ("kv_cache_memory_bytes", "None"), ("mamba_block_size", "None")]
SCHED_FIELDS = [("max_num_batched_tokens", "DEFAULT_MAX_NUM_BATCHED_TOKENS"), ("max_num_seqs", "128"),
                ("enable_chunked_prefill", "True"), ("policy", '"fcfs"'), ("long_prefill_token_threshold", "0"),
                ("prefill_schedule_interval", "1"), ("async_scheduling", "None"), ("stream_interval", "1")]
KVEV_FIELDS = [("enable_kv_cache_events", "False"), ("publisher", "None"), ("endpoint", '"tcp://*:5557"'),
                ("replay_endpoint", "None"), ("buffer_steps", "10000"), ("hwm", "100000"),
                ("max_queue_size", "100000"), ("topic", '""')]

ROW4_FILES = {
    "vllm/model_executor/layers/fused_moe/fused_moe.py": """
        def get_config_file_name(E, N, dtype):
            device_name = current_platform.get_device_name().replace(" ", "_")
            dtype_selector = "" if not dtype else f",dtype={dtype}"
            if "H200" in device_name.split("_"):
                device_name = "NVIDIA_H200"
            return f"E={E},N={N},device_name={device_name}{dtype_selector}.json"
        def get_moe_configs(E, N, dtype, M, down_moe=False):
            if envs.VLLM_BATCH_INVARIANT:
                return None
            json_file_name = get_config_file_name(E, N, dtype)
            if down_moe:
                json_file_name = json_file_name.removesuffix(".json") + "_down.json"
            user_defined_config_folder = envs.VLLM_TUNED_CONFIG_FOLDER
            default_config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)
            return {k: v for k, v in configs.items()}
        def try_get_optimal_moe_config(w1_shape, M, override_config=None):
            override_config = get_config()
            nearest = min(configs.keys(), key=lambda x: abs(x - M))
            return get_default_config(M, E)
    """,
    "vllm/utils/platform_utils.py": """
        def get_device_name_as_file_name(device_id=0):
            name = current_platform.get_device_name(device_id)
            name = re.sub(r"[\\s/]+", "_", name)
            return name
    """,
    "vllm/model_executor/layers/fused_moe/configs/E=1,N=2,device_name=NVIDIA_H200.json": "{}",
    "vllm/model_executor/layers/fused_moe/configs/E=1,N=2,device_name=NVIDIA_H200,dtype=fp8_w8a8_down.json": "{}",
    "vllm/model_executor/layers/fused_moe/configs/E=1,N=2,device_name=AMD_MI350X,dtype=int4,backend=flydsl.json": "{}",
    "vllm/config/cache.py": "KVOffloadingBackend = Literal['native']\n" + config_class(
        "CacheConfig", CACHE_FIELDS, "    DEFAULT_BLOCK_SIZE: ClassVar[int] = 16\n"),
    "vllm/config/scheduler.py": "SchedulerPolicy = Literal['fcfs']\n" + config_class(
        "SchedulerConfig", SCHED_FIELDS, "    DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048\n") + """
    def get_scheduler_cls(self):
        return resolve_obj_by_qualname(self.scheduler_cls)
""",
    "vllm/config/kv_events.py": config_class("KVEventsConfig", KVEV_FIELDS),
    "vllm/config/vllm.py": """
        class VllmConfig:
            def max_concurrent_batches(self):
                pp_size = self.parallel_config.pipeline_parallel_size
                if self.scheduler_config.async_scheduling:
                    return pp_size + 1
                return pp_size
            def _set_defaults(self):
                if self.scheduler_config.async_scheduling is None:
                    if self.model_config is not None and self.model_config.runner_type == "pooling":
                        self.scheduler_config.async_scheduling = False
                    elif not executor_supports_async_sched:
                        self.scheduler_config.async_scheduling = False
                    else:
                        self.scheduler_config.async_scheduling = True
    """,
    "vllm/v1/engine/__init__.py": """
        class EngineCoreRequestType:
            ADD = b"\\x00"
            ABORT = b"\\x01"
            WAKEUP = b"\\x02"
    """,
    "vllm/v1/engine/core.py": """
        _INPUTS_HANDLED_DURING_STEP = (EngineCoreRequestType.ADD, EngineCoreRequestType.ABORT)
        class EngineCore:
            def __init__(self):
                if self.batch_queue_size > 1:
                    self.batch_queue = deque()
                self.step_fn = self.step if self.batch_queue is None else self.step_with_batch_queue
            def step_with_batch_queue(self):
                batch_queue = self.batch_queue
                if self.deferred_step is None and self.scheduler.has_requests():
                    scheduler_output = self.scheduler.schedule()
                    future = self.model_executor.execute_model(scheduler_output)
                    if not scheduler_output.pending_structured_output_tokens:
                        self.model_executor.sample_tokens(grammar_output)
                    batch_queue.appendleft(future)
                batch_queue.pop()
                self._wait_for_output()
                self._process_aborts_queue()
                self.scheduler.update_from_output()
        class EngineCoreProc(EngineCore):
            def _process_input_queue(self):
                req = self.input_queue.get()
    """,
    "vllm/v1/core/block_pool.py": """
        class BlockPool:
            def cache_full_blocks(self, request, num_cached_blocks, num_full_blocks):
                if num_cached_blocks >= num_full_blocks:
                    return
                if self.enable_kv_cache_events:
                    self.kv_event_queue.append(BlockStored(hashes=[]))
            def _emit_block_removed_events(self, hashes):
                if not self.enable_kv_cache_events:
                    return
                self.kv_event_queue.append(BlockRemoved(hashes=hashes))
    """,
    "vllm/v1/core/kv_cache_manager.py": """
        class KVCacheManager:
            def get_computed_blocks(self, request, num_new_computed_tokens):
                if num_new_computed_tokens > 0 and self.enable_kv_cache_events and request.kv_cache_report_mode == "full":
                    self.coordinator.emit_cached_block_events(request)
            def allocate_slots(self, request):
                self.block_pool.cache_full_blocks(request, 0, 1)
                self.block_pool._emit_block_removed_events([])
    """,
    "vllm/v1/request.py": """
        class Request:
            def __init__(self, params):
                self.kv_cache_report_mode = "incremental"
    """,
    "vllm/v1/core/sched/async_scheduler.py": """
        class AsyncScheduler:
            def _update_after_schedule(self, scheduler_output):
                scheduler_output.pending_structured_output_tokens = x
                request.num_output_placeholders += 1
    """,
    "vllm/platforms/cuda.py": """
        def _get_backend_priorities(use_mla, device_capability):
            if use_mla:
                return [AttentionBackendEnum.FLASHMLA]
            return [AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TRITON_ATTN]
        class CudaPlatformBase:
            @classmethod
            def get_valid_backends(cls, device_capability):
                priorities = _get_backend_priorities(False, device_capability)
                return priorities, {}
    """,
    "vllm/platforms/interface.py": """
        class Platform:
            @classmethod
            def _preferred_block_size_for_backends(cls, backend_classes, default_block_size, vllm_config):
                if len(backend_classes) == 1:
                    return backend_classes[0].get_preferred_block_size(default_block_size)
                return default_block_size
    """,
}
FILES.update(ROW4_FILES)
BLOCKSIZE_ORACLE_TEXT = "platform.get_valid_backends(c)\nplatform._preferred_block_size_for_backends(b, 16, cfg)\n"

ROW4_IDS = [
    "moe_tuned_rule:config_file_name", "moe_tuned_rule:device_alias", "moe_tuned_rule:batch_invariant_bypass",
    "moe_tuned_rule:down_projection_suffix", "moe_tuned_rule:user_folder", "moe_tuned_rule:default_folder",
    "moe_tuned_rule:closest_batch", "moe_tuned_rule:override", "moe_tuned_rule:default_heuristic",
    "moe_tuned_rule:device_name_source", "moe_tuned_device:AMD_MI350X", "moe_tuned_device:NVIDIA_H200",
    "kv_event_emitter:cache_full_blocks", "kv_event_emitter:_emit_block_removed_events",
    "kv_report_mode:incremental", "kv_report_mode:full",
    "step_input_kind:ADD", "step_input_kind:ABORT", "step_input_kind:WAKEUP",
    "step_input_read:EngineCoreProc._process_input_queue:get#1",
    "async_depth:max_concurrent_batches", "async_depth:batch_queue_enabled", "async_depth:step_function",
    "async_default:async_scheduling=False if self.model_config is not None and self.model_config.runner_type == 'pooling'",
    "async_default:async_scheduling=False if not executor_supports_async_sched",
    "async_default:async_scheduling=True otherwise",
    "async_placeholder:scheduler_output.pending_structured_output_tokens",
    "async_placeholder:request.num_output_placeholders",
    "step_order:self.scheduler.schedule#1", "step_order:self.model_executor.execute_model#1",
    "step_order:self.model_executor.sample_tokens#1", "step_order:batch_queue.appendleft#1",
    "step_order:batch_queue.pop#1", "step_order:self._wait_for_output#1",
    "step_order:self._process_aborts_queue#1", "step_order:self.scheduler.update_from_output#1",
    "platform_preference:_get_backend_priorities", "platform_preference:CudaPlatformBase.get_valid_backends",
    "platform_preference:Platform._preferred_block_size_for_backends",
] + ["engine_default:%s.%s" % (c, f) for c, fs in (("CacheConfig", CACHE_FIELDS), ("SchedulerConfig", SCHED_FIELDS),
                                                   ("KVEventsConfig", KVEV_FIELDS)) for f, _ in fs] + [
    "engine_default:CacheConfig.DEFAULT_BLOCK_SIZE", "engine_default:SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS",
]

MANIFEST = {"generator": "test", "vllm": "0.32.0", "families": [
    {"name": "vllm:num_requests_running", "defined": "vllm/v1/metrics/loggers.py:10", "class": "exact"}]}

ISSUE = "https://github.com/inference-sim/blis-vllm/issues/1"
PROVENANCE = """package blisvllm

var Upstream = UpstreamRelease{
	Version: "0.32.0",
	Tag:     "v0.32.0",
	Commit:  "5df63a7203cdb9328d824a27b682e77c7b9928be",
}
"""
PROVENANCE_INDIRECT = """package blisvllm

import (
	"github.com/inference-sim/blis-vllm/internal/upstream"
)

var Upstream = UpstreamRelease{
	Version: upstream.Version,
	Tag:     upstream.Tag,
	Commit:  upstream.Commit,
}
"""
UPSTREAM_GO = """package upstream

const (
	Version = "0.33.0"
	Tag = "v0.33.0"
	Commit = "4ba787ec9959e566d3b587a796dff07673324da5"
)
"""
GOLDEN = "oracle/scenarios/basic.json"


def refused():
    return {"status": "refused", "issue": ISSUE}


def good_status():
    s = {
        "connector:PullConnector": {"status": "modelled", "evidence": [GOLDEN]},
        "connector:OldConnector": {"status": "modelled", "evidence": [GOLDEN]},
        "connector:StoreConnector": refused(),
        "offload_spec:CPUSpec": refused(),
        "secondary_tier:fs": refused(),
        "cache_policy:lru": refused(),
        "backpressure_class:DropStorePolicy": refused(),
        "offload_medium:CPU": refused(),
        "offload_backend:kv_offloading_backend=native": refused(),
        "offload_backend:SimpleCPUOffloadConnector kv_offload_backend=cpu": refused(),
        "kv_event_publisher:null": refused(),
        "kv_event_type:BlockStored": refused(),
        "kv_event_source:StoreKVEvents": refused(),
        "kv_cache_spec:FullAttentionSpec": refused(),
        "kv_cache_spec:role SOURCE": refused(),
        "speculative_method:ngram": refused(),
        "speculative_method:mtp": refused(),
        "speculative_method:deepseek_mtp": {"status": "out_of_scope", "reason": "deprecated",
                                            "citation": "vllm/config/speculative.py:2@5df63a72"},
        "scheduler_policy:fcfs": refused(),
        "framework:OffloadingManager": refused(),
        "framework:BackpressurePolicy": refused(),
        "framework:KVConnectorBase_V1 worker side": refused(),
        "framework:KVConnectorBase_V1 scheduler side": refused(),
        "metric_family:vllm:num_requests_running": refused(),
        "transformers_fallback:GlmForCausalLM": refused(),
        "transformers_fallback:EmuForConditionalGeneration": refused(),
        "transformers_fallback:TransformersForCausalLM": refused(),
        "transformers_fallback:TransformersMultiModalForCausalLM": refused(),
        "transformers_fallback:auto": refused(),
    }
    for name, *_ in gen_surface.PATH_LOADERS:
        s["path_loader:" + name] = refused()
    for iid in ROW4_IDS:
        s[iid] = refused()
    return {"items": s}


class Tree:
    def __init__(self, files=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "vllm-src")
        self.repo = os.path.join(self.tmp.name, "repo")
        for rel, body in (files or FILES).items():
            self.write(self.src, rel, textwrap.dedent(body))
        self.write(self.repo, "oracle/metrics/manifest.json", json.dumps(MANIFEST))
        self.write(self.repo, GOLDEN, "{}")
        self.write(self.repo, "oracle/run_x.py", "")
        self.write(self.repo, "oracle/blocksize/oracle.py", BLOCKSIZE_ORACLE_TEXT)
        self.write(self.repo, "provenance.go", PROVENANCE)

    @staticmethod
    def write(root, rel, text):
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def append(self, rel, text):
        with open(os.path.join(self.src, rel), "a", encoding="utf-8") as f:
            f.write(textwrap.dedent(text))

    def close(self):
        self.tmp.cleanup()


def run(tree, status, *extra):
    path = os.path.join(tree.tmp.name, "status.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status, f)
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = gen_surface.main([tree.src, path, "--repo", tree.repo, *extra])
    return code, out.getvalue(), err.getvalue()


class GenSurfaceTest(unittest.TestCase):
    def setUp(self):
        self.tree = Tree()

    def tearDown(self):
        self.tree.close()

    def items(self, status=None):
        code, out, err = run(self.tree, status or good_status())
        self.assertEqual(code, 0, err)
        return {i["id"]: i for i in json.loads(out)["items"]}

    def refusal(self, status):
        code, _, err = run(self.tree, status)
        self.assertEqual(code, 1, "the generator accepted it")
        return err

    def test_every_registered_item_is_listed_with_its_status(self):
        got = self.items()
        self.assertEqual(set(got), set(good_status()["items"]))
        self.assertEqual(got["connector:StoreConnector"]["status"], "refused")
        self.assertEqual(got["connector:StoreConnector"]["issue"], ISSUE)

    def test_the_transformers_fallback_paths_are_listed_by_route(self):
        got = self.items()
        reg = "vllm/model_executor/models/registry.py"
        self.assertEqual(got["transformers_fallback:GlmForCausalLM"]["route"], "architecture")
        self.assertEqual(got["transformers_fallback:GlmForCausalLM"]["backend"], "TransformersForCausalLM")
        self.assertEqual(got["transformers_fallback:GlmForCausalLM"]["site"], reg + ":15")
        self.assertEqual(got["transformers_fallback:TransformersForCausalLM"]["route"], "backend_class")
        self.assertEqual(got["transformers_fallback:auto"]["route"], "unregistered_architecture")
        self.assertNotIn("transformers_fallback:LlamaForCausalLM", got)
        self.assertNotIn("transformers_fallback:BertModel", got)

    def test_model_architectures_say_which_route_vllm_resolves_each_by(self):
        self.assertEqual(gen_surface.model_architectures(self.tree.src), {
            "LlamaForCausalLM": "native", "OlmoForCausalLM": "native", "BertModel": "native",
            "GlmForCausalLM": "transformers", "EmuForConditionalGeneration": "transformers",
            "TransformersForCausalLM": "transformers", "TransformersMultiModalForCausalLM": "transformers"})

    def test_only_literal_tables_and_items_of_one_are_evaluated(self):
        def pure(src):
            return gen_surface.pure_table(ast.parse(src, mode="eval").body)
        self.assertTrue(pure("{'A': ('m', 'C')}"))
        self.assertTrue(pure("{k: v for k, v in table.items()}"))
        self.assertFalse(pure("{'A': mod.CLS}"))
        self.assertFalse(pure("table.pop('A')"))
        self.assertFalse(pure("build()"))

    def test_a_registry_table_that_is_not_literal_fails(self):
        self.tree.append("vllm/model_executor/models/registry.py", """
            _VLLM_MODELS = {**build_models()}
        """)
        self.assertIn("_VLLM_MODELS is not a table of literals", self.refusal(good_status()))

    def test_a_name_bound_to_another_names_class_is_an_alias(self):
        got = self.items()
        self.assertEqual(got["connector:OldConnector"].get("alias_of"), "PullConnector")
        self.assertNotIn("alias_of", got["connector:PullConnector"])
        self.assertEqual(got["speculative_method:deepseek_mtp"].get("alias_of"), "mtp")
        self.assertNotIn("alias_of", got["speculative_method:mtp"])

    def test_the_connector_base_is_listed_as_two_parts(self):
        got = self.items()
        self.assertIn("framework:KVConnectorBase_V1 worker side", got)
        self.assertIn("framework:KVConnectorBase_V1 scheduler side", got)
        self.assertNotIn("framework:Helper", got, "a base class with no abstract method is not a framework")

    def test_a_newly_registered_connector_without_a_status_fails(self):
        self.tree.append(KC + "/factory.py",
                         'KVConnectorFactory.register_connector("NewConnector", "vllm.conn.store", "StoreConnector")\n')
        err = self.refusal(good_status())
        self.assertIn("connector:NewConnector: no status", err)

    def test_a_newly_registered_tier_publisher_and_spec_type_without_a_status_fail(self):
        self.tree.append("vllm/v1/kv_offload/tiering/factory.py",
                         'SecondaryTierFactory.register_tier("obj", "vllm.conn.store", "StoreConnector")\n')
        self.tree.write(self.tree.src, "vllm/v1/core/single_type_kv_cache_manager.py", textwrap.dedent("""
            def register_all_kvcache_specs(vllm_config):
                KVCacheSpecRegistry.register(FullAttentionSpec, FullAttentionManager)
                KVCacheSpecRegistry.register(NewSpec, NewManager)
                KVCacheSpecRegistry.register_role_manager(KVCacheGroupRole.SOURCE, SourceManager)
        """))
        err = self.refusal(good_status())
        self.assertIn("secondary_tier:obj: no status", err)
        self.assertIn("kv_cache_spec:NewSpec: no status", err)

    def test_a_relative_import_resolves_against_the_importing_package(self):
        self.tree.write(self.tree.src, "vllm/conn/pull/__init__.py",
                        "from .impl import PullConnector, OldConnector\n")
        got = self.items()
        self.assertEqual(set(got), set(good_status()["items"]))
        self.assertEqual(got["connector:OldConnector"]["status"], "modelled")

    def test_without_a_spec_registry_the_kv_cache_managers_are_listed(self):
        self.tree.write(self.tree.src, "vllm/v1/core/single_type_kv_cache_manager.py", textwrap.dedent("""
            class SingleTypeKVCacheManager:
                pass
            class _InternalBase(SingleTypeKVCacheManager):
                pass
            class FullAttentionManager(SingleTypeKVCacheManager):
                pass
            class MambaManager(_InternalBase):
                pass
            class NotAManager:
                pass
        """))
        s = good_status()
        for k in [k for k in s["items"] if k.startswith("kv_cache_spec:")]:
            del s["items"][k]
        s["items"]["kv_cache_manager:FullAttentionManager"] = refused()
        s["items"]["kv_cache_manager:MambaManager"] = refused()
        got = self.items(s)
        managers = {k: v for k, v in got.items() if k.startswith("kv_cache_manager:")}
        self.assertEqual(set(managers), {"kv_cache_manager:FullAttentionManager", "kv_cache_manager:MambaManager"})
        self.assertEqual(managers["kv_cache_manager:MambaManager"]["extends"], "_InternalBase")

    def test_a_path_loader_whose_file_a_commit_lacks_is_left_out(self):
        os.remove(os.path.join(self.tree.src, "vllm/v1/spec_decode/custom_class_proposer.py"))
        s = good_status()
        del s["items"]["path_loader:speculative_config.method custom_class (speculative_config.model as a module path)"]
        got = self.items(s)
        self.assertFalse([k for k in got if "custom_class" in k])
        self.assertIn("path_loader:speculative_config.speculator_config.speculator_class (qualified name)", got)

    def test_an_optional_path_loader_whose_call_a_file_lacks_is_left_out(self):
        self.tree.write(self.tree.src, "vllm/v1/worker/gpu/spec_decode/__init__.py", "def init_speculator(config):\n    pass\n")
        s = good_status()
        del s["items"]["path_loader:speculative_config.speculator_config.speculator_class (qualified name)"]
        self.assertFalse([k for k in self.items(s) if "speculator_class" in k])

    def test_a_new_path_loaded_component_must_be_accounted_for(self):
        self.tree.append("vllm/v1/kv_offload/base.py", """
            def load_plugin(path):
                return importlib.import_module(path)
        """)
        err = self.refusal(good_status())
        self.assertIn("vllm/v1/kv_offload/base.py", err)
        self.assertIn("no path_loader", err)

    def test_a_status_for_an_item_vllm_does_not_register_fails(self):
        s = good_status()
        s["items"]["connector:GoneConnector"] = refused()
        self.assertIn("connector:GoneConnector: has a status but vLLM", self.refusal(s))

    def test_an_entry_without_a_status_fails(self):
        s = good_status()
        s["items"]["connector:StoreConnector"] = {"issue": ISSUE}
        self.assertIn("connector:StoreConnector: status None", self.refusal(s))

    def test_modelled_needs_evidence_that_names_an_oracle_scenario(self):
        s = good_status()
        s["items"]["connector:PullConnector"] = {"status": "modelled", "evidence": ["oracle/run_x.py"]}
        self.assertIn("names no oracle scenario", self.refusal(s))
        s["items"]["connector:PullConnector"] = {"status": "modelled", "evidence": ["oracle/scenarios/missing.json"]}
        self.assertIn("is not a path in this repository", self.refusal(s))
        s["items"]["connector:PullConnector"] = {"status": "modelled"}
        self.assertIn("non-empty list", self.refusal(s))

    def test_refused_needs_a_blis_vllm_issue(self):
        s = good_status()
        s["items"]["connector:StoreConnector"] = {"status": "refused", "issue": "https://example.com/1"}
        self.assertIn("is not a blis-vllm or inference-sim issue URL", self.refusal(s))
        s["items"]["connector:StoreConnector"] = {"status": "refused", "issue": ISSUE, "refusal": "made_up"}
        self.assertIn("refusal 'made_up'", self.refusal(s))

    def test_out_of_scope_needs_one_of_three_reasons_and_a_citation(self):
        s = good_status()
        s["items"]["speculative_method:deepseek_mtp"] = {"status": "out_of_scope", "reason": "rarely used",
                                                         "citation": "vllm/config/speculative.py:2@5df63a72"}
        self.assertIn("reason 'rarely used'", self.refusal(s))
        s["items"]["speculative_method:deepseek_mtp"] = {"status": "out_of_scope", "reason": "deprecated",
                                                         "citation": "speculative.py line 2"}
        self.assertIn("citation", self.refusal(s))

    def test_a_framework_is_modelled_only_with_recorded_and_direct_traces(self):
        s = good_status()
        s["items"]["framework:OffloadingManager"] = {"status": "modelled", "evidence": [GOLDEN]}
        err = self.refusal(s)
        self.assertIn("only with recorded traces", err)
        self.assertIn("only with direct traces", err)
        s["items"]["framework:OffloadingManager"] = {"status": "modelled", "evidence": [GOLDEN],
                                                     "recorded": [GOLDEN], "direct": [GOLDEN]}
        self.assertEqual(self.items(s)["framework:OffloadingManager"]["status"], "modelled")

    def test_traces_belong_only_to_a_framework(self):
        s = good_status()
        s["items"]["connector:PullConnector"] = {"status": "modelled", "evidence": [GOLDEN], "recorded": [GOLDEN]}
        self.assertIn("traces are a framework's", self.refusal(s))

    def test_a_key_that_does_not_belong_with_the_status_fails(self):
        s = good_status()
        s["items"]["connector:StoreConnector"] = {"status": "refused", "issue": ISSUE, "evidence": [GOLDEN]}
        self.assertIn("do not belong with status refused", self.refusal(s))

    def test_the_release_is_provenance_gos_unless_named(self):
        code, out, err = run(self.tree, good_status())
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["upstream"], {"name": "vllm", "tag": "v0.32.0",
                                                       "commit": "5df63a7203cdb9328d824a27b682e77c7b9928be"})
        # Another release: its own commit in every out-of-scope citation, and its own manifest.
        other = ("v0.33.0", "4ba787ec995" + "0" * 29)
        code, _, err = run(self.tree, good_status(), "--upstream-tag", other[0], "--upstream-commit", other[1])
        self.assertEqual(code, 1)
        self.assertIn("oracle/metrics/manifest.json is for vLLM 0.32.0, not v0.33.0", err)
        MANIFEST_33 = dict(MANIFEST, vllm="0.33.0")
        self.tree.write(self.tree.repo, "oracle/metrics/manifest.json", json.dumps(MANIFEST_33))
        code, _, err = run(self.tree, good_status(), "--upstream-tag", other[0], "--upstream-commit", other[1])
        self.assertEqual(code, 1)
        self.assertIn("is not vllm/<path>:<lines>@4ba787ec", err)
        s = good_status()
        s["items"]["speculative_method:deepseek_mtp"]["citation"] = "vllm/config/speculative.py:2@4ba787ec"
        code, out, err = run(self.tree, s, "--upstream-tag", other[0], "--upstream-commit", other[1])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["upstream"]["tag"], "v0.33.0")

    def git_checkout(self):
        """Makes the synthetic vLLM tree a git checkout of its own and returns its HEAD."""
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
        for args in (["init", "-q"], ["add", "-A"], ["-c", "commit.gpgsign=false", "commit", "-q", "-m", "x"]):
            subprocess.run(["git", "-C", self.tree.src, *args], check=True, env=env, capture_output=True)
        return subprocess.run(["git", "-C", self.tree.src, "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()

    def test_a_checkout_at_another_commit_is_generated_for_that_commit(self):
        head = self.git_checkout()
        manifest = os.path.join(self.tree.tmp.name, "other-manifest.json")
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(dict(MANIFEST, vllm="9.9.9"), f)
        s = good_status()
        s["items"]["speculative_method:deepseek_mtp"]["citation"] = "vllm/config/speculative.py:2@" + head[:8]
        code, out, err = run(self.tree, s, "--manifest", manifest)
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["upstream"], {"name": "vllm", "tag": head[:12], "commit": head})
        code, out, err = run(self.tree, s, "--manifest", manifest, "--upstream-tag", "v9.9.9")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["upstream"], {"name": "vllm", "tag": "v9.9.9", "commit": head})

    def test_a_tag_alone_names_the_checkouts_commit(self):
        head = self.git_checkout()
        code, _, err = run(self.tree, good_status(), "--upstream-tag", "v9.9.9",
                           "--manifest", os.path.join(self.tree.repo, "oracle/metrics/manifest.json"))
        self.assertEqual(code, 1)
        self.assertIn("is not vllm/<path>:<lines>@" + head[:8], err)

    def test_a_commit_the_checkout_is_not_at_fails(self):
        self.git_checkout()
        code, _, err = run(self.tree, good_status(), "--upstream-commit", "4ba787ec995" + "0" * 29)
        self.assertEqual(code, 1)
        self.assertIn("the vLLM checkout is at", err)

    def test_an_unnamed_commit_with_the_pinned_manifest_fails(self):
        # The repository's manifest is for the pinned release; it cannot stand for another commit.
        head = self.git_checkout()
        s = good_status()
        s["items"]["speculative_method:deepseek_mtp"]["citation"] = "vllm/config/speculative.py:2@" + head[:8]
        code, _, err = run(self.tree, s)
        self.assertEqual(code, 1)
        self.assertIn("oracle/metrics/manifest.json is for vLLM 0.32.0", err)

    def test_the_release_is_read_through_provenance_gos_references(self):
        # v0.33.0's provenance.go names the release once, in internal/upstream (X23).
        self.tree.write(self.tree.repo, "provenance.go", PROVENANCE_INDIRECT)
        self.tree.write(self.tree.repo, "internal/upstream/upstream.go", UPSTREAM_GO)
        code, out, err = run(self.tree, good_status(), "--upstream-tag", "v0.32.0",
                             "--upstream-commit", "5df63a7203cdb9328d824a27b682e77c7b9928be")
        self.assertEqual(code, 0, err)
        self.assertEqual(gen_surface.provenance_upstream(self.tree.repo),
                         ("v0.33.0", "4ba787ec9959e566d3b587a796dff07673324da5"))

    def test_the_in_tree_module_path_is_read_too(self):
        in_tree = "github.com/inference-sim/inference-sim/engines/vllm/"
        self.tree.write(self.tree.repo, "provenance.go",
                        PROVENANCE_INDIRECT.replace("github.com/inference-sim/blis-vllm/", in_tree))
        self.tree.write(self.tree.repo, "internal/upstream/upstream.go", UPSTREAM_GO)
        self.assertEqual(gen_surface.provenance_upstream(self.tree.repo),
                         ("v0.33.0", "4ba787ec9959e566d3b587a796dff07673324da5"))

    def test_an_unresolvable_reference_fails(self):
        self.tree.write(self.tree.repo, "provenance.go", PROVENANCE_INDIRECT)
        # internal/upstream is missing, so Tag and Commit name nothing.
        with self.assertRaisesRegex(gen_surface.GenError, "no Upstream record with a Tag and a 40-hex Commit"):
            gen_surface.provenance_upstream(self.tree.repo)
        self.tree.write(self.tree.repo, "internal/upstream/upstream.go", UPSTREAM_GO.replace("4ba787ec", "4ba787"))
        with self.assertRaisesRegex(gen_surface.GenError, "40-hex"):
            gen_surface.provenance_upstream(self.tree.repo)

    def edit(self, rel, old, new):
        path = os.path.join(self.tree.src, rel)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.assertIn(old, text)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new))

    def test_the_tuned_moe_selection_is_listed_by_rule_and_device(self):
        got = self.items()
        self.assertEqual(got["moe_tuned_rule:device_alias"]["value"]["sets"], ["device_name = 'NVIDIA_H200'"])
        self.assertEqual(got["moe_tuned_rule:batch_invariant_bypass"]["value"]["then"], ["return None"])
        self.assertEqual(got["moe_tuned_rule:closest_batch"]["value"], "min(configs.keys(), key=lambda x: abs(x - M))")
        self.assertEqual(got["moe_tuned_device:NVIDIA_H200"]["value"], {"files": 2, "down_files": 1})

    def test_a_backend_selector_in_a_tuned_file_name_is_not_part_of_the_device(self):
        got = self.items()
        self.assertEqual(got["moe_tuned_device:AMD_MI350X"]["value"], {"files": 1, "down_files": 0})
        self.assertFalse([k for k in got if k.startswith("moe_tuned_device:") and "backend=" in k])

    def test_the_tuned_moe_rules_fail_when_a_rule_is_dropped(self):
        moe = "vllm/model_executor/layers/fused_moe/fused_moe.py"
        self.tearDown()
        for old, new, msg in (
                ('if "H200" in device_name.split("_"):', "if False:", "device-name aliases"),
                ("envs.VLLM_BATCH_INVARIANT", "envs.OTHER", "batch-invariant bypasses"),
                ("default_config_file_path =", "other_path =", "assigns default_config_file_path 0 times"),
                ("return get_default_config(M, E)", "return None", "calls get_default_config 0 times"),
                ("min(configs.keys()", "max(configs.keys()", "closest-batch lookups")):
            with self.subTest(old=old):
                self.setUp()
                try:
                    self.edit(moe, old, new)
                    self.assertIn(msg, self.refusal(good_status()))
                finally:
                    self.tearDown()
        self.setUp()

    def test_a_commit_before_the_down_projection_suffix_lists_no_such_rule(self):
        self.edit("vllm/model_executor/layers/fused_moe/fused_moe.py", "if down_moe:", "if False:")
        s = good_status()
        del s["items"]["moe_tuned_rule:down_projection_suffix"]
        self.assertNotIn("moe_tuned_rule:down_projection_suffix", self.items(s))

    def test_a_tuned_file_with_an_unparsable_name_fails(self):
        self.tree.write(self.tree.src, "vllm/model_executor/layers/fused_moe/configs/other.json", "{}")
        self.assertIn("not a tuned-config file name", self.refusal(good_status()))

    def test_a_new_tuned_device_needs_a_status(self):
        self.tree.write(self.tree.src, "vllm/model_executor/layers/fused_moe/configs/E=1,N=2,device_name=NEW_GPU.json", "{}")
        self.assertIn("moe_tuned_device:NEW_GPU: no status", self.refusal(good_status()))

    def test_a_setting_a_commit_added_later_is_listed_only_where_it_exists(self):
        self.assertNotIn("engine_default:SchedulerConfig.async_scheduling_depth", self.items())
        self.edit("vllm/config/scheduler.py", "    stream_interval: object = 1\n",
                  "    stream_interval: object = 1\n    async_scheduling_depth: object = 2\n")
        s = good_status()
        s["items"]["engine_default:SchedulerConfig.async_scheduling_depth"] = refused()
        self.assertEqual(self.items(s)["engine_default:SchedulerConfig.async_scheduling_depth"]["value"], 2)

    def test_a_default_is_the_value_the_source_gives_and_follows_a_constant(self):
        got = self.items()
        self.assertEqual(got["engine_default:CacheConfig.DEFAULT_BLOCK_SIZE"]["value"], 16)
        self.assertEqual(got["engine_default:CacheConfig.gpu_memory_utilization"]["value"], 0.9)
        self.assertEqual(got["engine_default:CacheConfig.block_size"]["value"], None)
        self.assertEqual(got["engine_default:SchedulerConfig.max_num_batched_tokens"]["value"], 2048)

    def test_a_default_that_changes_is_a_different_item_value(self):
        before = self.items()["engine_default:CacheConfig.DEFAULT_BLOCK_SIZE"]
        self.edit("vllm/config/cache.py", "DEFAULT_BLOCK_SIZE: ClassVar[int] = 16", "DEFAULT_BLOCK_SIZE: ClassVar[int] = 64")
        after = self.items()["engine_default:CacheConfig.DEFAULT_BLOCK_SIZE"]
        self.assertNotEqual(before["value"], after["value"])

    def test_a_missing_default_setting_fails_unless_a_commit_added_it_later(self):
        self.edit("vllm/config/cache.py", "    cache_dtype: object = \"auto\"\n", "")
        self.assertIn("no field CacheConfig.cache_dtype", self.refusal(good_status()))

    def test_the_kv_event_emitters_are_listed_with_their_guards_and_callers(self):
        got = self.items()
        full = got["kv_event_emitter:cache_full_blocks"]["value"]
        self.assertEqual(full["event"], "BlockStored")
        self.assertEqual(full["when"], ["self.enable_kv_cache_events", "not (num_cached_blocks >= num_full_blocks)"])
        self.assertEqual(full["called_from"], ["vllm/v1/core/kv_cache_manager.py::KVCacheManager.allocate_slots"])
        self.assertEqual(got["kv_event_emitter:_emit_block_removed_events"]["value"]["event"], "BlockRemoved")

    def test_a_new_event_emission_site_needs_a_status(self):
        self.edit("vllm/v1/core/block_pool.py", "    def _emit_block_removed_events",
                  "    def cache_partial_block(self, block):\n"
                  "        if self.enable_kv_cache_events:\n"
                  "            self.kv_event_queue.append(BlockStored(hashes=[]))\n"
                  "    def _emit_block_removed_events")
        self.assertIn("kv_event_emitter:cache_partial_block: no status", self.refusal(good_status()))

    def test_a_pool_with_no_event_emission_fails(self):
        self.tree.write(self.tree.src, "vllm/v1/core/block_pool.py", "class BlockPool:\n    def f(self):\n        pass\n")
        self.assertIn("appends no event to kv_event_queue", self.refusal(good_status()))

    def test_the_report_modes_are_the_default_and_the_cache_hit_branch(self):
        got = self.items()
        self.assertEqual(got["kv_report_mode:incremental"]["value"], {"default": True, "on_cache_hit": []})
        self.assertEqual(got["kv_report_mode:full"]["value"]["on_cache_hit"], ["self.coordinator.emit_cached_block_events"])
        self.assertIn("kv_cache_report_mode", got["kv_report_mode:full"]["value"]["when"].replace("request.", "kv_cache_report_mode"))

    def test_a_report_mode_without_a_default_fails(self):
        self.edit("vllm/v1/request.py", 'self.kv_cache_report_mode = "incremental"', "self.kv_cache_report_mode = mode")
        self.assertIn("kv_cache_report_mode is not set from a string default", self.refusal(good_status()))

    def test_a_report_mode_with_two_defaults_fails(self):
        self.edit("vllm/v1/request.py", 'self.kv_cache_report_mode = "incremental"',
                  'self.kv_cache_report_mode = "incremental"\n        self.kv_cache_report_mode = "full"')
        self.assertIn("has defaults", self.refusal(good_status()))

    def test_step_inputs_are_split_by_whether_vllm_handles_them_during_a_step(self):
        got = self.items()
        self.assertEqual(got["step_input_kind:ADD"]["value"], "during_step")
        self.assertEqual(got["step_input_kind:WAKEUP"]["value"], "after_step")
        self.assertFalse(got["step_input_read:EngineCoreProc._process_input_queue:get#1"]["value"]["filtered_by_step_kind"])

    def test_a_new_step_input_kind_needs_a_status(self):
        self.edit("vllm/v1/engine/__init__.py", "    WAKEUP =", "    NEW_INPUT = b'x'\n    WAKEUP =")
        self.assertIn("step_input_kind:NEW_INPUT: no status", self.refusal(good_status()))

    def test_a_during_step_kind_the_types_lack_fails(self):
        self.edit("vllm/v1/engine/core.py", "EngineCoreRequestType.ABORT)", "EngineCoreRequestType.GONE)")
        self.assertIn("names ['GONE']", self.refusal(good_status()))

    def test_the_step_n_plus_2_order_is_recorded_with_its_conditions(self):
        got = self.items()
        order = sorted((v["value"]["position"], k) for k, v in got.items() if k.startswith("step_order:"))
        self.assertEqual([k for _, k in order], [
            "step_order:self.scheduler.schedule#1", "step_order:self.model_executor.execute_model#1",
            "step_order:self.model_executor.sample_tokens#1", "step_order:batch_queue.appendleft#1",
            "step_order:batch_queue.pop#1", "step_order:self._wait_for_output#1",
            "step_order:self._process_aborts_queue#1", "step_order:self.scheduler.update_from_output#1"])
        self.assertEqual(got["step_order:batch_queue.appendleft#1"]["value"]["when"],
                         ["self.deferred_step is None and self.scheduler.has_requests()"])

    def test_a_step_that_stops_calling_a_stage_fails(self):
        self.edit("vllm/v1/engine/core.py", "self._process_aborts_queue()", "pass")
        self.assertIn("no longer calls ['self._process_aborts_queue']", self.refusal(good_status()))

    def test_the_async_defaults_and_placeholders_are_listed(self):
        got = self.items()
        self.assertEqual(got["async_default:async_scheduling=True otherwise"]["value"]["when"], [
            "not (self.model_config is not None and self.model_config.runner_type == 'pooling')",
            "not (not executor_supports_async_sched)"])
        self.assertEqual(got["async_placeholder:request.num_output_placeholders"]["value"], {"op": "+=", "expr": "1"})
        self.assertEqual(got["async_depth:max_concurrent_batches"]["value"][0],
                         {"when": ["self.scheduler_config.async_scheduling"], "returns": "pp_size + 1"})

    def test_a_new_async_default_branch_needs_a_status(self):
        self.edit("vllm/config/vllm.py", "            else:\n                self.scheduler_config.async_scheduling = True",
                  "            elif new_condition:\n                self.scheduler_config.async_scheduling = False\n"
                  "            else:\n                self.scheduler_config.async_scheduling = True")
        self.assertIn("async_default:async_scheduling=False if new_condition: no status", self.refusal(good_status()))

    def test_the_batch_queue_selection_is_read_not_assumed(self):
        self.edit("vllm/v1/engine/core.py", "if self.batch_queue_size > 1:", "if True:")
        self.assertIn("EngineCore.__init__ has 0 batch-queue tests", self.refusal(good_status()))

    def test_the_platform_preferences_are_the_lists_the_faked_platform_returns(self):
        got = self.items()
        self.assertEqual(got["platform_preference:_get_backend_priorities"]["value"], [
            {"when": ["use_mla"], "returns": "[AttentionBackendEnum.FLASHMLA]"},
            {"when": [], "returns": "[AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TRITON_ATTN]"}])

    def test_a_platform_function_the_oracle_does_not_run_fails(self):
        self.tree.write(self.tree.repo, "oracle/blocksize/oracle.py", "platform.get_valid_backends(c)\n")
        self.assertIn("no longer calls _preferred_block_size_for_backends", self.refusal(good_status()))

    def test_a_changed_priority_is_a_different_item_value(self):
        before = self.items()["platform_preference:_get_backend_priorities"]["value"]
        self.edit("vllm/platforms/cuda.py", "FLASH_ATTN, AttentionBackendEnum.TRITON_ATTN", "TRITON_ATTN, AttentionBackendEnum.FLASH_ATTN")
        self.assertNotEqual(before, self.items()["platform_preference:_get_backend_priorities"]["value"])

    def test_check_fails_on_drift_and_passes_when_current(self):
        code, out, err = run(self.tree, good_status())
        self.assertEqual(code, 0, err)
        committed = os.path.join(self.tree.tmp.name, "surface.json")
        with open(committed, "w", encoding="utf-8") as f:
            f.write(out)
        self.assertEqual(run(self.tree, good_status(), "--check", committed)[0], 0)
        s = copy.deepcopy(good_status())
        s["items"]["connector:StoreConnector"]["issue"] = ISSUE + "0"
        code, diff, err = run(self.tree, s, "--check", committed)
        self.assertEqual(code, 1)
        self.assertIn("regenerate it", err)
        self.assertIn("issues/10", diff)


if __name__ == "__main__":
    unittest.main()
