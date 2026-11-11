# Original to blis-vllm.
"""Tests for fields/gen_classes.py and for fields/classes.json (VLM-3, #102).

Run: python3 -I fields/test_classes.py
With the pinned vLLM source (CI sets both): VLLM_SOURCE=<checkout> VLLM_SOURCE_REQUIRED=1 python3 -I fields/test_classes.py

  - A synthetic vLLM tree gives the classes the comparisons in it imply (every shape the generator reads).
  - The committed fields/classes.json holds classes for every modelled or proven-inert field of
    fields.json, and for no other, and each class is backed by a comparison of the entry.
  - With the pinned source, each place a class cites holds the comparison the entry states, and the
    generator writes classes.json byte for byte.
  - The checker fails on each known-bad document: a modelled field without classes, a class no
    comparison backs, a cited line the source does not hold, classes that overlap.
"""

import copy
import importlib.util
import json
import os
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gc = load("gen_classes")
read_json, read_text = gc.read_json, gc.read_text

FILES = {
    "vllm/engine/arg_utils.py": """
        class EngineArgs:
            max_num_seqs: int | None = None
            policy_name: SchedulerPolicy = SchedulerConfig.policy
            enable_thing: bool = SchedulerConfig.enable_thing
            speculative_config: dict[str, Any] | None = None
            def add_cli_args(parser):
                sched = get_kwargs(SchedulerConfig)
                parser.add_argument("--max-num-seqs", **sched["max_num_seqs"])
                parser.add_argument("--policy-name")
                parser.add_argument("--enable-thing")
                parser.add_argument("--speculative-config")
        class AsyncEngineArgs(EngineArgs):
            pass
    """,
    "vllm/entrypoints/launchers/cli_args.py": """
        class BaseFrontendArgs:
            port: int = 8000
        class FrontendArgs(BaseFrontendArgs):
            pass
        def make_arg_parser(parser):
            return parser
    """,
    "vllm/config/scheduler.py": """
        from typing import Literal
        from vllm.config.utils import config
        SchedulerPolicy = Literal["fcfs", "priority", "other"]
        @config
        class SchedulerConfig:
            max_num_seqs: int = Field(default=128, ge=1)
            policy: SchedulerPolicy = "fcfs"
            enable_thing: bool = False
            def check(self):
                if self.max_num_seqs > 256:
                    raise ValueError("x")
                if self.max_num_seqs <= 4:
                    return 1
                if self.enable_thing:
                    return 2
    """,
    "vllm/config/speculative.py": """
        from vllm.config.utils import config
        @config
        class SpeculativeConfig:
            method: str | None = None
            num_speculative_tokens: int = Field(default=1, gt=0)
    """,
    "vllm/config/vllm.py": """
        from vllm.config.utils import config
        @config
        class VllmConfig:
            scheduler_config: SchedulerConfig = None
            speculative_config: SpeculativeConfig | None = None
    """,
    "vllm/v1/sched.py": """
        from vllm.v1.q import Policy
        class Scheduler:
            def __init__(self, vllm_config):
                self.scheduler_config = vllm_config.scheduler_config
                sc = vllm_config.scheduler_config
                self.cap = sc.max_num_seqs
                limit = self.scheduler_config.max_num_seqs
                if 8 < limit:
                    pass
                if limit == 64:
                    pass
                if self.scheduler_config.policy == Policy.PRIORITY:
                    pass
                spec = vllm_config.speculative_config
                if spec is not None and spec.method in ("eagle", "ngram"):
                    pass
                if spec.method is None:
                    pass
                if spec.num_speculative_tokens >= 3:
                    pass
            def step(self):
                if self.cap >= 1000:
                    pass
    """,
    "vllm/v1/q.py": """
        from enum import Enum
        class Policy(Enum):
            FCFS = "fcfs"
            PRIORITY = "priority"
    """,
    "vllm/other.py": """
        class NotTheConfig:
            def f(self, x):
                # `max_num_seqs` of some other object: not the field.
                if x.max_num_seqs == 7:
                    pass
    """,
}


def write_tree(root, files):
    for path, body in files.items():
        p = os.path.join(root, path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(textwrap.dedent(body))


def field(name, status="modelled", source="config", traces=()):
    return {"name": name, "source": source, "status": status,
            "evidence": {"traces": list(traces)} if status == "modelled" else {"proof": "x"},
            "estimation": {"rule": "none"}}


class SyntheticTree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        write_tree(self.root, FILES)
        # The repository the traces live in: one golden trace, whose scenario states the values.
        write_tree(self.root, {"oracle/scenarios/t1.json": json.dumps({"config": {
            "max_num_seqs": 100, "policy_name": "priority", "enable_thing": True,
            "speculative.method": "eagle", "speculative.num_speculative_tokens": 3}})})
        gc._PARSED.clear()
        gc.ENUM_MEMBERS.clear()
        t = ["core/x/testdata/golden/t1.json"]
        self.fl = {"kind": "FieldList", "engine": {"kind": "vllm", "version": "0.32.0"},
                   "fields": [field("max_num_seqs", traces=t), field("policy_name", traces=t), field("enable_thing", traces=t),
                              field("speculative_config.method", traces=t),
                              field("speculative_config.num_speculative_tokens", traces=t)]}
        path = os.path.join(self.root, "fields.json")
        with open(path, "w") as f:
            json.dump(self.fl, f)
        self.doc = gc.build(self.root, path, self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def edges(self, name):
        return [(c["lo"], c["hi"], c["lo_closed"], c["hi_closed"]) for c in self.doc["fields"][name]["basis"]["classes"] if "lo" in c]

    def test_a_numeric_field_is_cut_at_every_comparison_with_a_literal(self):
        # >256, <=4, 8 < limit (limit is max_num_seqs), == 64, cap >= 1000 (cap is stored from it): domain [1, inf).
        want = [(1, 5, True, False), (5, 9, True, False), (9, 64, True, False), (64, 65, True, False),
                (65, 257, True, False), (257, 1000, True, False), (1000, None, True, False)]
        self.assertEqual(self.edges("max_num_seqs"), want)

    def test_every_class_of_the_numeric_field_cites_a_comparison(self):
        for c in self.doc["fields"]["max_num_seqs"]["basis"]["classes"]:
            self.assertTrue(c["compared_at"], c)

    def test_a_comparison_on_another_object_is_not_the_field(self):
        sites = [s["at"] for s in self.doc["fields"]["max_num_seqs"]["basis"]["sites"]]
        self.assertFalse([a for a in sites if a.startswith("vllm/other.py")], sites)

    def test_an_enumeration_gives_a_class_per_compared_value_and_one_for_the_rest(self):
        # `field == Policy.PRIORITY` compares with an enum member; "fcfs" and "other" no comparison tells apart.
        cl = self.doc["fields"]["policy_name"]["basis"]["classes"]
        self.assertEqual([c.get("values") for c in cl], [["priority"], ["fcfs", "other"]])
        self.assertTrue(cl[1]["uncompared"])

    def test_a_boolean_has_both_values_as_classes_backed_by_its_truth_test(self):
        cl = self.doc["fields"]["enable_thing"]["basis"]["classes"]
        self.assertEqual([c["values"] for c in cl], [[False], [True]])
        for c in cl:
            self.assertEqual(c["compared_at"], ["vllm/config/scheduler.py:15"])

    def test_a_string_field_has_a_class_per_compared_value_and_an_other_class_and_an_unset_class(self):
        cl = self.doc["fields"]["speculative_config.method"]["basis"]["classes"]
        self.assertEqual([c.get("values") for c in cl[:2]], [["eagle"], ["ngram"]])
        self.assertTrue(cl[2]["other"])
        self.assertTrue(cl[3]["unset"] is True and cl[3]["compared_at"])
        self.assertEqual(len(cl), 4)

    def test_a_domain_bound_that_is_open_on_an_integer_field_moves_to_the_next_integer(self):
        # num_speculative_tokens: gt=0 means 1 and up; >= 3 cuts at 3.
        self.assertEqual(self.edges("speculative_config.num_speculative_tokens"), [(1, 3, True, False), (3, None, True, False)])

    def test_the_substrate_object_is_the_evidence_projected_onto_the_schema(self):
        vc = {n: r["value_classes"] for n, r in self.doc["fields"].items()}
        self.assertEqual(vc["max_num_seqs"], {
            "type": "numeric",
            "thresholds": [{"value": 5, "closed": "above"}, {"value": 9, "closed": "above"}, {"value": 64, "closed": "above"},
                           {"value": 65, "closed": "above"}, {"value": 257, "closed": "above"}, {"value": 1000, "closed": "above"}],
            "covered_classes": [4]})
        self.assertEqual(vc["enable_thing"], {"type": "boolean", "covered": ["true"]})
        self.assertEqual(vc["policy_name"], {"type": "enumerated", "values": ["fcfs", "priority", "other"], "covered": ["priority"]})
        self.assertEqual(vc["speculative_config.method"], {"type": "enumerated", "covered": ["eagle"]})
        self.assertEqual(vc["speculative_config.num_speculative_tokens"],
                         {"type": "numeric", "thresholds": [{"value": 3, "closed": "above"}], "covered_classes": [1]})

    def test_class_of_follows_the_substrates_rule_on_the_line(self):
        above, below = [{"value": 8, "closed": "above"}], [{"value": 8, "closed": "below"}]
        self.assertEqual([gc.class_of(above, x) for x in (7, 8, 9)], [0, 1, 1])
        self.assertEqual([gc.class_of(below, x) for x in (7, 8, 9)], [0, 0, 1])

    def test_a_value_class_the_basis_does_not_give_fails(self):
        doc = copy.deepcopy(self.doc)
        doc["fields"]["max_num_seqs"]["value_classes"]["covered_classes"] = [0]
        self.assertTrue([p for p in gc.check_document(doc, self.fl, None) if "not what its basis gives" in p])

    def test_a_field_whose_evidence_states_no_value_fails(self):
        doc = copy.deepcopy(self.doc)
        b = doc["fields"]["enable_thing"]
        b["basis"]["stated"] = []
        b["value_classes"]["covered"] = []
        self.assertTrue([p for p in gc.check_document(doc, self.fl, None) if "states no value" in p])

    def test_the_document_passes_its_own_check(self):
        self.assertEqual(gc.check_document(self.doc, self.fl, self.root), [])

    def test_a_modelled_field_missing_from_the_document_fails(self):
        doc = copy.deepcopy(self.doc)
        del doc["fields"]["max_num_seqs"]
        self.assertTrue([p for p in gc.check_document(doc, self.fl, None) if "has no value classes" in p])

    def test_an_entry_for_a_field_that_is_not_modelled_fails(self):
        fl = copy.deepcopy(self.fl)
        fl["fields"][0]["status"] = "not_modelled"
        fl["fields"][0]["evidence"] = {"issue": "inference-sim/blis-vllm#1"}
        self.assertTrue([p for p in gc.check_document(self.doc, fl, None) if "neither modelled nor proven inert" in p])

    def test_a_field_with_no_classes_fails(self):
        doc = copy.deepcopy(self.doc)
        doc["fields"]["max_num_seqs"]["basis"]["classes"] = []
        self.assertTrue([p for p in gc.check_document(doc, self.fl, None) if "no classes" in p])

    def test_a_class_no_comparison_backs_fails(self):
        doc = copy.deepcopy(self.doc)
        doc["fields"]["max_num_seqs"]["basis"]["classes"][1]["compared_at"] = []
        self.assertTrue([p for p in gc.check_document(doc, self.fl, None) if "backed by no comparison" in p])

    def test_a_class_citing_a_place_that_is_not_a_comparison_of_the_entry_fails(self):
        doc = copy.deepcopy(self.doc)
        doc["fields"]["max_num_seqs"]["basis"]["classes"][1]["compared_at"] = ["vllm/v1/sched.py:1"]
        self.assertTrue([p for p in gc.check_document(doc, self.fl, None) if "not a comparison of the entry" in p])

    def test_a_class_whose_edge_no_cited_comparison_names_fails(self):
        doc = copy.deepcopy(self.doc)
        cl = doc["fields"]["max_num_seqs"]["basis"]["classes"]
        cl[1]["compared_at"] = cl[0]["compared_at"][:1] if cl[0]["compared_at"] != cl[1]["compared_at"] else cl[3]["compared_at"]
        cl[1]["lo"], cl[1]["hi"] = 6, 7
        self.assertTrue([p for p in gc.check_document(doc, self.fl, None) if "which compares" in p])

    def test_classes_that_overlap_fail(self):
        doc = copy.deepcopy(self.doc)
        doc["fields"]["max_num_seqs"]["basis"]["classes"][1]["lo"] = 3
        self.assertTrue([p for p in gc.check_document(doc, self.fl, None) if "do not touch exactly" in p])

    def test_a_cited_line_the_source_no_longer_holds_fails(self):
        path = os.path.join(self.root, "vllm/config/scheduler.py")
        text = read_text(path)
        with open(path, "w") as f:
            f.write(text.replace("self.max_num_seqs > 256", "self.max_num_seqs > 512"))
        self.assertTrue([p for p in gc.check_document(self.doc, self.fl, self.root) if "no longer holds" in p])

    def test_a_site_whose_text_does_not_mention_its_value_fails(self):
        doc = copy.deepcopy(self.doc)
        for s in doc["fields"]["max_num_seqs"]["basis"]["sites"]:
            if s["expr"] == "self.max_num_seqs > 256":
                s["values"] = [300]
        for c in doc["fields"]["max_num_seqs"]["basis"]["classes"]:
            if c.get("lo") == 257 or c.get("hi") == 257:
                c["lo"] = 301 if c.get("lo") == 257 else c["lo"]
                c["hi"] = 301 if c.get("hi") == 257 else c["hi"]
        problems = gc.check_document(doc, self.fl, self.root)
        self.assertTrue([p for p in problems if "does not mention the value 300" in p], problems)

    def test_every_reviewed_site_and_stated_value_gives_its_reason(self):
        extra = os.path.join(HERE, "classes_extra.json")
        for name, entry in read_json(extra).items():
            for s in entry.get("sites", []) + entry.get("stated", []):
                self.assertTrue(s.get("why"), name)


@unittest.skipUnless(os.environ.get("VLLM_SOURCE"), "VLLM_SOURCE is unset: the pinned source is not checked")
class AgainstThePinnedSource(unittest.TestCase):
    def test_the_committed_classes_are_what_the_pinned_source_gives_and_cite_real_lines(self):
        root = os.environ["VLLM_SOURCE"]
        fl = read_json(os.path.join(REPO, "fields.json"))
        doc = read_json(os.path.join(HERE, "value_classes.json"))
        self.assertEqual(gc.check_document(doc, fl, root), [])
        rebuilt = gc.build(root, os.path.join(REPO, "fields.json"), REPO)
        self.assertEqual(json.dumps(rebuilt, indent=1, ensure_ascii=False) + "\n",
                         read_text(os.path.join(HERE, "value_classes.json")))


class CommittedClasses(unittest.TestCase):
    def test_every_modelled_or_proven_inert_field_has_classes_each_backed_by_a_comparison(self):
        fl = read_json(os.path.join(REPO, "fields.json"))
        doc = read_json(os.path.join(HERE, "value_classes.json"))
        self.assertEqual(gc.check_document(doc, fl, None), [])

    def test_the_committed_document_states_the_limit_of_what_it_finds(self):
        doc = read_json(os.path.join(HERE, "value_classes.json"))
        self.assertIn("not followed", doc["limits"])


if __name__ == "__main__":
    unittest.main()
