# Original to inference-sim.
"""Tests for upgrade_report.py on small synthetic fixtures (no engine source, no network).

Run: python3 -I engines/upgrade/test_upgrade_report.py
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upgrade_report as ur  # noqa: E402


def item(id_, category="connector", status="modelled"):
    return {"id": id_, "category": category, "site": "x.py:1", "status": status}


class TestSurfaceDiff(unittest.TestCase):
    def test_every_added_item_has_a_status(self):
        old = [item("connector:A"), item("connector:B", status="refused")]
        new = [item("connector:A"), item("connector:C"), item("connector:D"), item("transformers_fallback:auto", "transformers_fallback")]
        statuses = {"connector:C": {"status": "refused", "issue": "https://github.com/inference-sim/inference-sim/issues/2286"},
                    "transformers_fallback:auto": {"status": "refused", "issue": "i"}}
        d = ur.surface_diff(old, new, statuses)
        self.assertEqual({r["id"]: r["status"] for r in d["added"]},
                         {"connector:C": "refused", "connector:D": "unstatused", "transformers_fallback:auto": "refused"})
        self.assertEqual(d["added"][0]["issue"], "https://github.com/inference-sim/inference-sim/issues/2286")
        self.assertEqual(d["removed"], [{"id": "connector:B", "category": "connector", "status": "refused"}])
        self.assertEqual(d["distance"], {"added": 3, "removed": 1, "changed": 0, "unstatused": 1,
                                         "added_by_status": {"refused": 2, "unstatused": 1}})
        self.assertEqual(d["transformers_fallback"], ["transformers_fallback:auto"])

    def test_an_item_whose_value_differs_is_changed_and_keeps_no_old_verdict(self):
        old = [dict(item("engine_default:CacheConfig.DEFAULT_BLOCK_SIZE", "engine_default"), value=16),
               dict(item("engine_default:CacheConfig.cache_dtype", "engine_default"), value="auto"),
               item("connector:A")]
        new = [dict(item("engine_default:CacheConfig.DEFAULT_BLOCK_SIZE", "engine_default"), value=64),
               dict(item("engine_default:CacheConfig.cache_dtype", "engine_default"), value="auto"),
               item("connector:A")]
        d = ur.surface_diff(old, new, {"engine_default:CacheConfig.DEFAULT_BLOCK_SIZE": {"status": "modelled"}})
        self.assertEqual(d["changed"], [{"id": "engine_default:CacheConfig.DEFAULT_BLOCK_SIZE", "category": "engine_default",
                                         "status": "modelled", "from": 16, "to": 64}])
        self.assertEqual((d["distance"]["changed"], d["added"], d["removed"]), (1, [], []))

    def test_a_value_compares_by_content_not_by_key_order(self):
        old = [dict(item("kv_report_mode:full", "kv_report_mode"), value={"a": 1, "b": [1, 2]})]
        new = [dict(item("kv_report_mode:full", "kv_report_mode"), value={"b": [1, 2], "a": 1})]
        self.assertEqual(ur.surface_diff(old, new, {})["changed"], [])

    def test_no_change_is_zero_distance(self):
        items = [item("connector:A")]
        d = ur.surface_diff(items, items, {})
        self.assertEqual(d["distance"]["added"] + d["distance"]["removed"], 0)

    def test_a_status_for_an_item_that_exists_does_not_make_it_added(self):
        items = [item("connector:A")]
        self.assertEqual(ur.surface_diff(items, items, {"connector:A": {"status": "refused"}})["added"], [])


def field(name, status="modelled", source="config", **vc):
    f = {"name": name, "source": source, "status": status}
    if vc:
        f["value_classes"] = vc
    return f


class TestFieldDemand(unittest.TestCase):
    OLD = [field("a"), field("b", "not_modelled"), field("gone")]
    NEW = [field("a"), field("b", "modelled"), field("backend", type="enumerated", values=["x", "y", "z"], covered=["x"]),
           field("flag", type="boolean", covered=["true"]), field("n", type="numeric", covered_classes=[0]),
           field("off", "not_modelled")]

    def test_added_removed_and_status_changed(self):
        d = ur.field_demand(self.OLD, self.NEW, {})
        self.assertEqual([f["name"] for f in d["added"]], ["backend", "flag", "n", "off"])
        self.assertEqual(d["removed"], [{"name": "gone", "source": "config"}])
        self.assertEqual(d["status_changed"], [{"name": "b", "source": "config", "from": "not_modelled", "to": "modelled"}])

    def test_demand_is_blocked_by_status_by_value_and_by_absence(self):
        demand = {"backend": {"x": 5, "y": 7}, "flag": {"true": 1, "false": 2}, "off": {"": 11},
                  "missing": {"": 3}, "n": {"4": 9}, "a": {"": 1}}
        rows = {(r["field"], r["value"]): r for r in ur.field_demand(self.OLD, self.NEW, demand)["demand"]}
        self.assertFalse(rows[("backend", "x")]["blocked"])
        self.assertTrue(rows[("backend", "y")]["blocked"])
        self.assertEqual(rows[("backend", "y")]["value_covered"], False)
        self.assertFalse(rows[("flag", "true")]["blocked"])
        self.assertTrue(rows[("flag", "false")]["blocked"])
        self.assertTrue(rows[("off", "")]["blocked"])
        self.assertEqual(rows[("missing", "")]["status"], "absent")
        self.assertIsNone(rows[("n", "4")]["value_covered"])
        self.assertFalse(rows[("n", "4")]["blocked"])
        self.assertFalse(rows[("a", "")]["blocked"])

    def test_blocked_points_sum_and_order(self):
        demand = {"backend": {"x": 5, "y": 7}, "off": {"": 11}}
        d = ur.field_demand(self.OLD, self.NEW, demand)
        self.assertEqual(d["blocked_points"], 18)
        self.assertEqual([r["points"] for r in d["demand"]], [11, 7, 5])


def table(**paths):
    return {"paths": [{"path": p, "code_id": cid, "functions": [{"id": k, "hash": h} for k, h in fns.items()]}
                      for p, (cid, fns) in paths.items()]}


class TestSourceDiff(unittest.TestCase):
    def test_code_path_diff(self):
        old = table(host=("c1", {"f::a": "h1", "f::b": "h2", "f::gone": "h3"}), graphs=("g1", {"g::x": "h"}))
        new = table(host=("c2", {"f::a": "h1", "f::b": "h9", "f::new": "h4"}), graphs=("g1", {"g::x": "h"}))
        d = {p["path"]: p for p in ur.code_path_diff(old, new)}
        self.assertEqual((d["host"]["from"], d["host"]["to"]), ("c1", "c2"))
        self.assertEqual((d["host"]["changed"], d["host"]["added"], d["host"]["removed"]), (["f::b"], ["f::new"], ["f::gone"]))
        self.assertEqual((d["graphs"]["changed"], d["graphs"]["added"], d["graphs"]["removed"]), ([], [], []))

    def test_kernel_impls_diff(self):
        def impl(variant, h, prim="attention.decode"):
            return {"primitive": prim, "selection": "s", "library": "l", "variant": variant, "target_arch": "sm_90a", "code_hash": h}
        old = {"implementations": [impl("a", "1"), impl("b", "2"), impl("c", "3")]}
        new = {"implementations": [impl("a", "1"), impl("b", "9"), impl("d", "4")]}
        d = ur.kernel_impls_diff(old, new)
        self.assertEqual([k[3] for k in d["added"]], ["d"])
        self.assertEqual([k[3] for k in d["removed"]], ["c"])
        self.assertEqual([k[3] for k in d["changed"]], ["b"])
        self.assertEqual(d["hashes"], ["4", "9"])

    def test_derived_files_and_ported_files_changed(self):
        texts = {"a.go": "// Original to x.\n// Derived from vllm/v1/a.py at vLLM v0.33.0 (abc).\n",
                 "b.go": "// Derived from vllm/v1/a.py at vLLM v0.33.0 (abc).\n",
                 "c.go": "// Derived from vllm/v1/c.py at vLLM v0.33.0 (abc).\n", "d.go": "// nothing\n"}
        derived = ur.derived_files(texts)
        self.assertEqual(derived, {"vllm/v1/a.py": ["a.go", "b.go"], "vllm/v1/c.py": ["c.go"]})
        d = ur.source_diff(derived, ["vllm/v1/a.py", "vllm/other.py"], [], None)
        self.assertEqual(d["ported_files_changed"], [{"upstream": "vllm/v1/a.py", "files": ["a.go", "b.go"]}])
        self.assertEqual(ur.source_diff(derived, [], [], None)["ported_files_changed"], [])


class TestModels(unittest.TestCase):
    def test_support_before_and_after_and_transformers_only(self):
        models = {"m1": "LlamaForCausalLM", "m2": "NewForCausalLM", "m3": "TfForCausalLM", "m4": "OddForCausalLM"}
        old = {"LlamaForCausalLM": "native", "NewForCausalLM": "transformers", "TfForCausalLM": "transformers"}
        new = {"LlamaForCausalLM": "native", "NewForCausalLM": "native", "TfForCausalLM": "transformers"}
        r = ur.model_report(models, old, new, {"code_ids": {"host": "c"}})
        rows = {m["model"]: m for m in r["models"]}
        self.assertEqual(rows["m1"]["support"], {"from": "native", "to": "native"})
        self.assertTrue(rows["m2"]["changed"])
        self.assertEqual(rows["m4"]["support"], {"from": "unregistered", "to": "unregistered"})
        self.assertEqual(r["transformers_only"], ["m3", "m4"])
        self.assertEqual(rows["m1"]["missing_coefficients"], {"code_ids": {"host": "c"}})
        self.assertTrue(all(m["catalog_entry"] for m in r["models"]))

    def test_without_an_old_tree_the_change_is_unknown(self):
        row = ur.model_report({"m": "A"}, None, {"A": "native"}, {})["models"][0]
        self.assertEqual((row["support"]["from"], row["changed"]), (None, None))

    def test_uncovered_coefficients(self):
        u = ur.uncovered({"host": "h1", "graphs": "g1"}, ["k1", "k2", "k2"], {"h1"}, {"k1"})
        self.assertEqual(u, {"code_ids": {"graphs": "g1"}, "kernel_code_hashes": ["k2"]})


class TestReadingTheRepository(unittest.TestCase):
    def test_catalog_models_reads_the_derived_graph(self):
        with tempfile.TemporaryDirectory() as d:
            for name, text in (("m1", "global:\n  architecture: LlamaForCausalLM\n"), ("m2", "global:\n  other: 1\n")):
                os.makedirs(os.path.join(d, "models", name))
                with open(os.path.join(d, "models", name, "graph.yaml"), "w") as f:
                    f.write(text)
            os.makedirs(os.path.join(d, "models", "m3"))
            self.assertEqual(ur.catalog_models(d), {"m1": "LlamaForCausalLM"})

    def test_coverage_reads_code_ids_and_kernel_hashes(self):
        h1, h2 = "a" * 64, "b" * 64
        text = ("  - x:\n      code_ids: [%s, %s]\n      kernel_impls: [{library: l, variant: v, code_hash: %s}]\n"
                "      measured_on: [{library: l, variant: v, code_hash: %s}]\n" % (h1, h2, h1, h2))
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "c.yaml"), "w") as f:
                f.write(text)
            with open(os.path.join(d, "ignored.txt"), "w") as f:
                f.write("code_ids: [%s]" % ("c" * 64))
            ids, hashes = ur.coverage(d)
        self.assertEqual(ids, {h1, h2})
        self.assertEqual(hashes, {h1})


if __name__ == "__main__":
    unittest.main()
