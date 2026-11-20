# Original to blis-sglang.
"""Tests for gen_surface.py over a small synthetic SGLang tree and repository.

Run: python3 -I tools/surface/test_gen_surface.py
"""

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

COMMIT = "0123456789abcdef0123456789abcdef01234567"
SRT = "python/sglang/srt/"

FILES = {
    "disaggregation/utils.py": """
        from enum import Enum
        class TransferBackend(Enum):
            MOONCAKE = "mooncake"
            FAKE = "fake"
        def get_kv_class(transfer_backend, class_type): ...
        def get_kv_class(transfer_backend, class_type):
            if transfer_backend == TransferBackend.MOONCAKE:
                class_mapping = {KVClassType.MANAGER: MooncakeKVManager}
            elif transfer_backend == TransferBackend.FAKE:
                class_mapping = {KVClassType.MANAGER: FakeKVManager}
            return class_mapping.get(class_type)
    """,
    "arg_groups/choices.py": """
        DISAGG_TRANSFER_BACKEND_CHOICES = ["mooncake", "fake", "mooncake_tcp"]
        RADIX_EVICTION_POLICY_CHOICES = ["lru", "lfu"]
        add_disagg_transfer_backend_choices = DISAGG_TRANSFER_BACKEND_CHOICES.extend
    """,
    "arg_groups/pd_disaggregation_hook.py": """
        def handle(cfg):
            if cfg.disaggregation_transfer_backend == "mooncake_tcp":
                cfg.disaggregation_transfer_backend = "mooncake"
    """,
    "arg_groups/fields/disagg.py": """
        from sglang.srt.arg_groups.choices import DISAGG_TRANSFER_BACKEND_CHOICES
        class Disagg(msgspec.Struct):
            _NS_PATH = "disagg"
            disaggregation_transfer_backend: A[str, Arg(help="h", choices=DISAGG_TRANSFER_BACKEND_CHOICES)] = "mooncake"
            disaggregation_decode_retraction_backup: A[str, Arg(choices=["cpu_tensor", "none"])] = "none"
    """,
    "arg_groups/fields/memory.py": """
        from sglang.srt.arg_groups.choices import RADIX_EVICTION_POLICY_CHOICES
        class Memory(msgspec.Struct):
            _NS_PATH = "memory"
            radix_eviction_policy: A[str, Arg(choices=RADIX_EVICTION_POLICY_CHOICES)] = "lru"
            hicache_write_policy: A[str, Arg(choices=["write_back", "write_through"])] = "write_through"
            hicache_io_backend: A[str, Arg(choices=["direct", "kernel"])] = "kernel"
            hicache_mem_layout: A[str, Arg(choices=["layer_first"])] = "layer_first"
            hicache_storage_prefetch_policy: A[str, Arg(choices=["timeout"])] = "timeout"
            hicache_host_memory_mode: A[Literal["cache", "buffer_only"], "help"] = "cache"
            hicache_storage_backend: A[Optional[str], Arg(choices=["file", "dynamic"])] = None
            unified_cache_external_linker_backend: A[str, Arg(choices=["mooncake", "dynamic"])] = "mooncake"
    """,
    "arg_groups/fields/schedule.py": """
        class Schedule(msgspec.Struct):
            _NS_PATH = "schedule"
            schedule_policy: A[str, Arg(choices=["fcfs", "lpm", "priority"])] = "fcfs"
    """,
    "arg_groups/fields/exec_.py": """
        class Moe(msgspec.Struct):
            _NS_PATH = "exec.moe"
            moe_a2a_backend: A[Literal["none", "deepep"], Arg(choices=["none", "deepep"])] = "none"
    """,
    "mem_cache/storage/backend_factory.py": """
        class StorageBackendFactory:
            @classmethod
            def create_backend(cls, backend_name):
                if backend_name == "dynamic" and extra is not None:
                    return cls._create_dynamic_backend()
        StorageBackendFactory.register_backend("file", "sglang.srt.mem_cache.hicache_storage", "HiCacheFile")
    """,
    "mem_cache/hicache_storage.py": """
        class HiCacheStorage(ABC):
            pass
        class HiCacheFile(HiCacheStorage):
            pass
    """,
    "mem_cache/base_prefix_cache.py": """
        class BasePrefixCache(ABC):
            pass
    """,
    "mem_cache/radix_cache.py": """
        class RadixCache(BasePrefixCache):
            pass
    """,
    "mem_cache/unified_radix_cache.py": """
        class UnifiedRadixCache(BasePrefixCache):
            pass
    """,
    "mem_cache/pure_swa_radix_cache.py": """
        class PureSWARadixCache(BasePrefixCache):
            pass
    """,
    "mem_cache/storage/flexkv/__init__.py": """
        from sglang.srt.mem_cache.registry import register_radix_cache_backend
        def _flexkv_factory(ctx):
            return FlexKVRadixCache(ctx)
        try:
            register_radix_cache_backend("flexkv", _flexkv_factory)
        except ValueError:
            pass
    """,
    "mem_cache/storage/flexkv/flexkv_radix_cache.py": """
        class FlexKVRadixCache(RadixCache):
            pass
    """,
    "mem_cache/registry.py": """
        def register_radix_cache_backend(name, factory):
            pass
        def default_radix_cache_factory(ctx):
            if ctx.disable_radix_cache:
                return PureSWARadixCache(params=ctx.params)
            if get_memory().enable_flexkv:
                return _flexkv_factory(ctx)
            return create_unified_radix_cache(ctx)
        def create_unified_radix_cache(ctx):
            cache = UnifiedRadixCache(ctx.params)
            if ctx.enable_hierarchical_cache:
                cache.init_hicache(ctx.server_args, ctx.params)
            return cache
    """,
    "mem_cache/unified_cache/tree_core_registry.py": """
        def register_tree_core_backend(name, factory):
            pass
        register_tree_core_backend("python", _python_tree_core_factory)
    """,
    "mem_cache/unified_cache/component_type.py": """
        class ComponentType(int, Enum):
            FULL = 0
            SWA = 1
    """,
    "mem_cache/unified_cache/linker_factory.py": """
        def create_unified_cache_linker(server_args, params, *, backend, config, components):
            if backend == "dynamic":
                return load(config)
            if backend == "mooncake":
                linker_cls = MooncakeDirectLinker
            else:
                raise ValueError(backend)
            return linker_cls()
    """,
    "mem_cache/unified_cache/unified_cache_linker.py": """
        class UnifiedCacheLinker(ABC):
            pass
    """,
    "mem_cache/utils.py": """
        _EVICTION_POLICY_FACTORIES: dict = {
            "lru": LRUStrategy,
            "lfu": LFUStrategy,
            "fifo": FIFOStrategy,
        }
    """,
    "managers/schedule_policy.py": """
        class CacheAwarePolicy(Enum):
            LPM = "lpm"
        class CacheAgnosticPolicy(Enum):
            FCFS = "fcfs"
        class SchedulePolicy:
            def __init__(self):
                self.waiting_queue_radix_tree = RadixCache.create_simulated()
            def _validate_and_adjust_policy(self, policy):
                raise ValueError(f"Unknown schedule_policy: {policy=}")
    """,
    "mem_cache/allocator/base.py": """
        class BaseTokenToKVPoolAllocator(ABC):
            pass
    """,
    "mem_cache/allocator/paged.py": """
        class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
            pass
    """,
    "mem_cache/memory_pool.py": """
        class ReqToTokenPool:
            pass
        class KVCache(ABC):
            pass
        class MambaPool:
            pass
        class MHATokenToKVPool(KVCache):
            pass
        class MLATokenToKVPool(KVCache):
            pass
    """,
    "disaggregation/decode.py": """
        class DecodeReqToTokenPool:
            pass
    """,
    "mem_cache/unified_memory_pool.py": """
        class UnifiedKVPool:
            pass
        def init_unified_swa_pools(cfg):
            return UnifiedKVPool()
    """,
    "mem_cache/allocator/mamba.py": """
        class MambaSlotAllocator:
            pass
    """,
    "mem_cache/unified_memory_pool_slots.py": """
        class UnifiedMambaSlotAllocator:
            pass
    """,
    "mem_cache/kv_cache_configurator.py": """
        class KVCacheConfigurator:
            def _build_req_to_token_pool(self):
                if self.decode:
                    return DecodeReqToTokenPool()
                return ReqToTokenPool()
            def _build_token_to_kv_pool(self):
                if self.use_mla_backend:
                    return MLATokenToKVPool()
                return MHATokenToKVPool()
            def _build_token_to_kv_pool_allocator(self):
                return PagedTokenToKVPoolAllocator()
    """,
    "mem_cache/pool_host/base.py": """
        class HostKVCache(ABC):
            pass
    """,
    "mem_cache/pool_host/mla.py": """
        class MLATokenToKVPoolHost(HostKVCache):
            pass
    """,
    "speculative/spec_info.py": """
        class SpeculativeAlgorithm(Enum):
            EAGLE = auto()
            NONE = auto()
            def create_worker(self, server_args):
                if self.is_eagle():
                    return EAGLEWorkerV2
                raise ValueError("unreachable")
    """,
    "speculative/spec_registry.py": """
        _RESERVED_ALIASES = frozenset({"NEXTN"})
        def register_algorithm(name):
            pass
    """,
    "layers/moe/utils.py": """
        class MoeA2ABackend(Enum):
            NONE = "none"
            DEEPEP = "deepep"
            CUSTOMIZED = "customized"
    """,
    "disaggregation/kv_events.py": """
        class BlockStored(KVCacheEvent):
            pass
        class StorageMedium(str, enum.Enum):
            GPU = "GPU"
            CPU = "CPU_PINNED"
        class EventPublisherFactory:
            _registry: dict = {
                "null": NullEventPublisher,
            }
            @classmethod
            def register_publisher(cls, name, ctor):
                pass
    """,
    "observability/metrics_collector.py": """
        from collections import Counter
        def f(reqs):
            return Counter(r.priority for r in reqs)
        class Collector:
            def __init__(self):
                self.a = Gauge(name="sglang:num_running_reqs", documentation="d")
                self.b = Histogram(name="sglang:ttft_seconds", documentation="d")
    """,
    "observability/ray_wrappers.py": """
        class RayGauge:
            def __init__(self, name):
                self.g = Gauge(name=name)
    """,
    "managers/cache_controller.py": """
        class LayerDoneCounter:
            pass
        class HiCacheController:
            pass
    """,
    "mem_cache/hybrid_cache/hybrid_cache_controller.py": """
        class HybridCacheController(BaseHiCacheController):
            pass
    """,
    "mem_cache/l2_transfer.py": """
        class L2TransferEngine:
            pass
    """,
    "disaggregation/base/conn.py": """
        class BaseKVManager(ABC):
            pass
        class BaseKVSender(ABC):
            pass
        class BaseKVReceiver(ABC):
            pass
        class BaseKVBootstrapServer(ABC):
            pass
    """,
    "platforms/interface.py": """
        class Platform:
            def get_kv_pool_cls(self, *, kind):
                return None
            def get_paged_allocator_cls(self):
                return None
    """,
    "arg_groups/resolution_hooks.py": """
        def register_resolution_hook(name):
            pass
    """,
}

ISSUE = "https://github.com/inference-sim/blis-sglang/issues/3"
PROVENANCE = """// Original to blis-sglang.
package blissglang

var Upstream = harness.Upstream{
\tVersion: "0.5.23",
\tTag:     "v0.5.23",
\tCommit:  "%s",
}
""" % COMMIT


def write_tree(root, files):
    for rel, text in files.items():
        path = os.path.join(root, SRT, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(text).lstrip("\n"))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "sglang")
        self.repo = os.path.join(self.tmp.name, "repo")
        self.files = copy.deepcopy(FILES)
        os.makedirs(os.path.join(self.repo, "oracle", "scenarios"))
        with open(os.path.join(self.repo, "oracle", "scenarios", "tiny.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(self.repo, "provenance.go"), "w") as f:
            f.write(PROVENANCE)
        gen_surface.UPSTREAM["tag"], gen_surface.UPSTREAM["commit"] = "v0.5.23", COMMIT

    def tearDown(self):
        self.tmp.cleanup()

    def derive(self):
        write_tree(self.src, self.files)
        return gen_surface.derive(self.src)

    def ids(self):
        return {i["id"] for i in self.derive()}

    def all_refused(self):
        return {"items": {i: {"status": "refused", "issue": ISSUE} for i in self.ids()}}

    def build(self, status):
        write_tree(self.src, self.files)
        return gen_surface.build(self.src, status, self.repo)

    def fails(self, pattern, status=None):
        with self.assertRaisesRegex(gen_surface.GenError, pattern):
            if status is None:
                self.derive()
            else:
                self.build(status)


class TestItems(Base):
    def test_every_category_is_read_from_its_registration_site(self):
        by_id = {i["id"]: i for i in self.derive()}
        want = {
            "transfer_backend:mooncake": "disaggregation/utils.py:3",
            "transfer_backend:mooncake_tcp": "arg_groups/pd_disaggregation_hook.py:2",
            "hicache_storage:file": "mem_cache/storage/backend_factory.py:6",
            "tree_cache:UnifiedRadixCache": "mem_cache/unified_radix_cache.py:1",
            "tree_cache_backend:flexkv": "mem_cache/storage/flexkv/__init__.py:5",
            "tree_core:python": "mem_cache/unified_cache/tree_core_registry.py:3",
            "eviction_policy:fifo": "mem_cache/utils.py:4",
            "kv_pool:MLATokenToKVPool": "mem_cache/memory_pool.py:9",
            "host_pool:MLATokenToKVPoolHost": "mem_cache/pool_host/mla.py:1",
            "schedule_policy:priority": "arg_groups/fields/schedule.py:3",
            "spec_algo:NEXTN": "speculative/spec_registry.py:1",
            "spec_worker:EAGLEWorkerV2": "speculative/spec_info.py:5",
            "metric_family:sglang:ttft_seconds": "observability/metrics_collector.py:7",
            "extension_point:Platform.get_kv_pool_cls": "platforms/interface.py:2",
        }
        for iid, s in want.items():
            self.assertIn(iid, by_id)
            self.assertEqual(by_id[iid]["site"], s, iid)

    def test_aliases_rejections_and_flags_are_recorded(self):
        by_id = {i["id"]: i for i in self.derive()}
        self.assertEqual(by_id["transfer_backend:mooncake_tcp"]["alias_of"], "mooncake")
        self.assertEqual(by_id["spec_algo:NEXTN"]["alias_of"], "EAGLE")
        self.assertEqual(by_id["schedule_policy:priority"]["rejected"], "managers/schedule_policy.py:9")
        self.assertIs(by_id["eviction_policy:fifo"]["cli"], False)
        self.assertNotIn("cli", by_id["eviction_policy:lru"])
        self.assertIs(by_id["moe_a2a_backend:customized"]["cli"], False)
        self.assertTrue(by_id["hicache_storage:dynamic"]["path_loaded"])
        self.assertEqual(by_id["hicache_storage:file"]["level"], "L3")
        self.assertEqual(by_id["host_pool:MLATokenToKVPoolHost"]["level"], "L2")
        self.assertEqual(by_id["kv_pool:MHATokenToKVPool"]["level"], "L1")
        self.assertEqual(by_id["transfer_backend:fake"]["manager"], "FakeKVManager")
        self.assertEqual(by_id["tree_cache_backend:flexkv"]["builds"], "FlexKVRadixCache")

    def test_the_selection_chain_records_where_it_selects_each_tree_cache(self):
        by_id = {i["id"]: i for i in self.derive()}
        self.assertEqual(by_id["tree_cache:PureSWARadixCache"]["selected"], "mem_cache/registry.py:5")
        self.assertEqual(by_id["tree_cache:FlexKVRadixCache"]["selected"], "mem_cache/registry.py:7")
        self.assertEqual(by_id["tree_cache:UnifiedRadixCache"]["selected"], "mem_cache/registry.py:8")
        self.assertEqual(by_id["tree_cache:RadixCache"]["selected"], "managers/schedule_policy.py:7")
        self.assertNotIn("selected", by_id["tree_cache:BasePrefixCache"])

    def test_a_pool_the_configurator_builds_records_the_line(self):
        by_id = {i["id"]: i for i in self.derive()}
        self.assertEqual(by_id["kv_pool:MLATokenToKVPool"]["built"], "mem_cache/kv_cache_configurator.py:8")
        self.assertEqual(by_id["req_pool:DecodeReqToTokenPool"]["built"], "mem_cache/kv_cache_configurator.py:4")
        self.assertEqual(by_id["kv_pool:UnifiedKVPool"]["built"], "mem_cache/unified_memory_pool.py:4")
        self.assertNotIn("built", by_id["kv_pool:KVCache"])

    def test_a_collections_counter_is_not_a_metric_and_a_wrapper_is_allowed(self):
        names = [i["name"] for i in self.derive() if i["category"] == "metric_family"]
        self.assertEqual(names, ["sglang:num_running_reqs", "sglang:ttft_seconds"])


class TestKnownBad(Base):
    """Each input a new release could bring that the list would otherwise miss fails the generator."""

    def test_an_unlisted_branch_in_the_tree_cache_chain(self):
        self.files["mem_cache/registry.py"] = self.files["mem_cache/registry.py"].replace(
            "            return create_unified_radix_cache(ctx)\n        def create",
            "            if ctx.new_flag:\n                return build_new_cache(ctx)\n"
            "            return create_unified_radix_cache(ctx)\n        def create")
        self.fails(r"build_new_cache.*cannot tell which one tree cache class")

    def test_an_unreadable_return_in_the_tree_cache_chain(self):
        self.files["mem_cache/registry.py"] = self.files["mem_cache/registry.py"].replace(
            "            return create_unified_radix_cache(ctx)\n        def create",
            "            if ctx.x:\n                return ctx.cache\n"
            "            return create_unified_radix_cache(ctx)\n        def create")
        self.fails(r"default_radix_cache_factory returns ctx.cache")

    def test_a_new_tree_cache_class_has_no_status(self):
        self.files["mem_cache/new_cache.py"] = "class NewRadixCache(UnifiedRadixCache):\n    pass\n"
        status = self.all_refused()
        del status["items"]["tree_cache:NewRadixCache"]
        self.fails(r"tree_cache:NewRadixCache: no status", status)

    def test_an_unregistered_transfer_backend_choice(self):
        self.files["arg_groups/choices.py"] = self.files["arg_groups/choices.py"].replace(
            '"mooncake_tcp"]', '"mooncake_tcp", "bogus"]')
        self.fails(r"choice 'bogus' is not a TransferBackend member and no alias rewrites it")

    def test_a_transfer_backend_with_no_class_map_branch(self):
        self.files["disaggregation/utils.py"] = self.files["disaggregation/utils.py"].replace(
            '    FAKE = "fake"', '    FAKE = "fake"\n            NIXL = "nixl"')
        self.fails(r"TransferBackend.NIXL has no get_kv_class branch")

    def test_an_unregistered_storage_choice(self):
        self.files["arg_groups/fields/memory.py"] = self.files["arg_groups/fields/memory.py"].replace(
            '["file", "dynamic"]', '["file", "dynamic", "newstore"]')
        self.fails(r"--hicache-storage-backend choices \['newstore'\]")

    def test_a_storage_class_outside_the_storage_interface(self):
        self.files["mem_cache/hicache_storage.py"] = "class HiCacheStorage(ABC):\n    pass\nclass HiCacheFile:\n    pass\n"
        self.fails(r"registers HiCacheFile, which does not derive from HiCacheStorage")

    def test_a_schedule_policy_choice_that_is_no_policy(self):
        self.files["arg_groups/fields/schedule.py"] = self.files["arg_groups/fields/schedule.py"].replace(
            '"priority"]', '"priority", "edf"]')
        self.fails(r"--schedule-policy choice 'edf' is not a policy")

    def test_an_eviction_choice_with_no_factory(self):
        self.files["arg_groups/choices.py"] = self.files["arg_groups/choices.py"].replace(
            '["lru", "lfu"]', '["lru", "lfu", "arc"]')
        self.fails(r"--radix-eviction-policy choice 'arc' has no eviction factory")

    def test_the_configurator_building_an_unclassified_pool(self):
        self.files["mem_cache/kv_cache_configurator.py"] = self.files["mem_cache/kv_cache_configurator.py"].replace(
            "                return MHATokenToKVPool()",
            "                if self.new:\n                    return NewShapeKVPool()\n                return MHATokenToKVPool()")
        self.files["mem_cache/new_pool.py"] = "class NewShapeKVPool:\n    pass\n"
        self.fails(r"the pool configurator uses NewShapeKVPool \(mem_cache/kv_cache_configurator.py:\d+\), which derive from none")

    def test_a_metric_whose_name_is_not_a_literal(self):
        self.files["observability/other.py"] = "def f(n):\n    return Gauge(name=n)\n"
        self.fails(r"observability/other.py:2: Gauge's metric name is not a literal")

    def test_a_moe_a2a_choice_that_is_not_a_member(self):
        self.files["arg_groups/fields/exec_.py"] = self.files["arg_groups/fields/exec_.py"].replace(
            'choices=["none", "deepep"]', 'choices=["none", "deepep", "uccl"]')
        self.fails(r"--moe-a2a-backend choice 'uccl' is not a MoeA2ABackend member")


class TestStatuses(Base):
    def test_complete_statuses_build_a_list(self):
        doc = self.build(self.all_refused())
        self.assertEqual(doc["upstream"], {"name": "sglang", "tag": "v0.5.23", "commit": COMMIT})
        self.assertEqual(doc["counts"], {"refused": len(doc["items"])})

    def test_a_status_for_an_item_sglang_does_not_offer(self):
        status = self.all_refused()
        status["items"]["hicache_storage:gone"] = {"status": "refused", "issue": ISSUE}
        self.fails(r"hicache_storage:gone: has a status but SGLang v0.5.23 offers no such item", status)

    def test_a_refusal_needs_a_blis_sglang_issue(self):
        status = self.all_refused()
        status["items"]["tree_core:python"] = {"status": "refused", "issue": "https://github.com/inference-sim/blis-vllm/issues/1"}
        self.fails(r"tree_core:python: refused, but issue .* is not a blis-sglang issue URL", status)

    def test_modelled_needs_a_scenario_and_a_framework_needs_its_traces(self):
        status = self.all_refused()
        status["items"]["tree_core:python"] = {"status": "modelled", "evidence": ["provenance.go"]}
        status["items"]["framework:HiCacheController"] = {"status": "modelled", "evidence": ["oracle/scenarios/tiny.json"]}
        with self.assertRaises(gen_surface.GenError) as e:
            self.build(status)
        self.assertIn("tree_core:python: modelled, but its evidence names no oracle scenario", str(e.exception))
        self.assertIn("framework:HiCacheController: a framework is modelled only with recorded traces", str(e.exception))
        self.assertIn("framework:HiCacheController: a framework is modelled only with direct traces", str(e.exception))

    def test_a_framework_with_its_traces_is_modelled(self):
        status = self.all_refused()
        status["items"]["framework:HiCacheController"] = {
            "status": "modelled", "evidence": ["oracle/scenarios/tiny.json"],
            "recorded": ["oracle/scenarios/tiny.json"], "direct": ["oracle/scenarios/tiny.json"]}
        doc = self.build(status)
        self.assertEqual(doc["counts"]["modelled"], 1)

    def test_traces_on_a_feature_are_refused(self):
        status = self.all_refused()
        status["items"]["tree_core:python"] = {"status": "modelled", "evidence": ["oracle/scenarios/tiny.json"],
                                               "recorded": ["oracle/scenarios/tiny.json"]}
        self.fails(r"tree_core:python: recorded traces are a framework's \(FID-5\)", status)

    def test_out_of_scope_needs_a_listed_reason_and_a_pinned_citation(self):
        status = self.all_refused()
        status["items"]["transfer_backend:fake"] = {"status": "out_of_scope", "reason": "rarely used",
                                                    "citation": "python/sglang/srt/disaggregation/utils.py:686@deadbeef"}
        with self.assertRaises(gen_surface.GenError) as e:
            self.build(status)
        self.assertIn("reason 'rarely used' is not one of hardware, no_priced_work, deprecated", str(e.exception))
        self.assertIn("is not python/sglang/<path>:<lines>@01234567", str(e.exception))

    def test_a_key_that_does_not_belong_with_the_status(self):
        status = self.all_refused()
        status["items"]["tree_core:python"] = {"status": "refused", "issue": ISSUE, "evidence": ["x"]}
        self.fails(r"tree_core:python: keys \['evidence'\] do not belong with status refused", status)


class TestCommandLine(Base):
    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = gen_surface.main(list(args))
        return rc, out.getvalue(), err.getvalue()

    def test_check_passes_on_the_written_list_and_fails_on_drift(self):
        write_tree(self.src, self.files)
        status_path = os.path.join(self.tmp.name, "status.json")
        with open(status_path, "w") as f:
            json.dump(self.all_refused(), f)
        rc, out, _ = self.run_main(self.src, status_path, "--repo", self.repo)
        self.assertEqual(rc, 0)
        listed = os.path.join(self.tmp.name, "surface.json")
        with open(listed, "w") as f:
            f.write(out)
        self.assertEqual(self.run_main(self.src, status_path, "--repo", self.repo, "--check", listed)[0], 0)
        with open(listed, "w") as f:
            f.write(out.replace('"refused"', '"modelled"', 1))
        rc, _, err = self.run_main(self.src, status_path, "--repo", self.repo, "--check", listed)
        self.assertEqual(rc, 1)
        self.assertIn("is not what the generator writes", err)

    def git_checkout(self):
        write_tree(self.src, self.files)
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
        for cmd in (["init", "-q"], ["add", "."], ["commit", "-q", "--no-gpg-sign", "-m", "x"]):
            subprocess.run(["git", "-C", self.src] + cmd, check=True, env=env, capture_output=True)
        return subprocess.run(["git", "-C", self.src, "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()

    def upstream_of(self, *extra):
        status_path = os.path.join(self.tmp.name, "status.json")
        with open(status_path, "w") as f:
            json.dump(self.all_refused(), f)
        rc, out, err = self.run_main(self.src, status_path, "--repo", self.repo, *extra)
        self.assertEqual(rc, 0, err)
        return json.loads(out)["upstream"]

    def test_a_checkout_at_another_commit_is_generated_for_that_commit(self):
        head = self.git_checkout()
        self.assertNotEqual(head, COMMIT)
        self.assertEqual(self.upstream_of(), {"name": "sglang", "tag": head[:12], "commit": head})
        self.assertEqual(self.upstream_of("--upstream-tag", "v9.9.9"), {"name": "sglang", "tag": "v9.9.9", "commit": head})
        self.assertEqual(self.upstream_of("--upstream-commit", head)["commit"], head)

    def test_a_commit_the_checkout_is_not_at_is_refused(self):
        self.git_checkout()
        rc, _, err = self.run_main(self.src, "--repo", self.repo, "--items", "--upstream-commit", COMMIT)
        self.assertEqual(rc, 1)
        self.assertIn("the SGLang checkout is at", err)
        self.assertIn("not %s" % COMMIT, err)
        rc, _, err = self.run_main(self.src, "--repo", self.repo, "--items", "--upstream-commit", "abc")
        self.assertEqual(rc, 1)
        self.assertIn("40 hex", err)

    def test_a_directory_that_is_no_checkout_is_the_commit_provenance_names(self):
        write_tree(self.src, self.files)
        self.assertEqual(self.upstream_of(), {"name": "sglang", "tag": "v0.5.23", "commit": COMMIT})


if __name__ == "__main__":
    unittest.main()
