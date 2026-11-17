# Original to blis-vllm.
"""Tests for gen_surface.py over a small synthetic vLLM tree and repository.

Run: python3 -I tools/surface/test_gen_surface.py
"""

import copy
import io
import json
import os
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
    "vllm/v1/spec_decode/custom_class_proposer.py": """
        import importlib
        def load(module_path):
            module = importlib.import_module(module_path)
    """,
}

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
    }
    for name, *_ in gen_surface.PATH_LOADERS:
        s["path_loader:" + name] = refused()
    return {"items": s}


class Tree:
    def __init__(self, files=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "vllm-src")
        self.repo = os.path.join(self.tmp.name, "repo")
        for rel, body in (files or FILES).items():
            self.write(self.src, rel, textwrap.dedent(body))
        self.write(self.repo, "docs/metrics-manifest.json", json.dumps(MANIFEST))
        self.write(self.repo, GOLDEN, "{}")
        self.write(self.repo, "oracle/run_x.py", "")
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
        self.assertIn("is not a blis-vllm issue URL", self.refusal(s))
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
        self.assertIn("docs/metrics-manifest.json is for vLLM 0.32.0, not v0.33.0", err)
        MANIFEST_33 = dict(MANIFEST, vllm="0.33.0")
        self.tree.write(self.tree.repo, "docs/metrics-manifest.json", json.dumps(MANIFEST_33))
        code, _, err = run(self.tree, good_status(), "--upstream-tag", other[0], "--upstream-commit", other[1])
        self.assertEqual(code, 1)
        self.assertIn("is not vllm/<path>:<lines>@4ba787ec", err)
        s = good_status()
        s["items"]["speculative_method:deepseek_mtp"]["citation"] = "vllm/config/speculative.py:2@4ba787ec"
        code, out, err = run(self.tree, s, "--upstream-tag", other[0], "--upstream-commit", other[1])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["upstream"]["tag"], "v0.33.0")

    def test_the_release_is_read_through_provenance_gos_references(self):
        # v0.33.0's provenance.go names the release once, in internal/upstream (X23).
        self.tree.write(self.tree.repo, "provenance.go", PROVENANCE_INDIRECT)
        self.tree.write(self.tree.repo, "internal/upstream/upstream.go", UPSTREAM_GO)
        code, out, err = run(self.tree, good_status(), "--upstream-tag", "v0.32.0",
                             "--upstream-commit", "5df63a7203cdb9328d824a27b682e77c7b9928be")
        self.assertEqual(code, 0, err)
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

    def test_a_half_named_release_fails(self):
        code, _, err = run(self.tree, good_status(), "--upstream-tag", "v0.33.0")
        self.assertEqual(code, 1)
        self.assertIn("go together", err)

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
