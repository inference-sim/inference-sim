# Original to blis-sglang.
"""Tests for gen_fields.py over a small synthetic SGLang tree.

Run: python3 -I fields/test_gen_fields.py
"""

import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_fields  # noqa: E402

SRT = "python/sglang/srt"

FILES = {
    SRT + "/arg_groups/fields/sched.py": '''
        import msgspec
        from sglang.srt.arg_groups.arg_utils import A, Arg, Derived
        from sglang.srt.arg_groups.choices import POLICY_CHOICES

        class Sched(msgspec.Struct):
            _NS_PATH = "schedule"
            max_running_requests: A[Optional[int], "The maximum number of running requests."] = None
            schedule_policy: A[str, Arg(help="Policy.", choices=POLICY_CHOICES)] = "fcfs"
            kv_mode: A[Optional[Literal["auto", "fp8"]], "Mode."] = None
            tp_size: A[int, Arg(help="TP.", cli_name="--tensor-parallel-size", aliases=["--tp"])] = 1
            enable_dp_attention: A[bool, Arg(help="Old.", no_cli=True)] = False
            plain_field: Optional[int] = None
            dp_rank = Derived(doc="set at runtime")
            attn_tp_size: Optional[int] = Derived(doc="computed after resolution")
    ''',
    SRT + "/arg_groups/fields/mem.py": '''
        import msgspec
        from sglang.srt.arg_groups.arg_utils import A, Arg

        class Mem(msgspec.Struct):
            _NS_PATH = "memory"
            page_size: A[Optional[int], Arg(help="Page.", resolvable=True)] = None
    ''',
    SRT + "/arg_groups/choices.py": '''
        POLICY_CHOICES = ["lpm", "fcfs"]
    ''',
    SRT + "/server_args.py": '''
        from sglang.srt.arg_groups.fields.sched import (
            Sched,
        )
        from sglang.srt.arg_groups.fields.mem import Mem

        class ServerArgs:
            @staticmethod
            def add_cli_args(parser):
                add_cli_args_from_dataclass(parser, ServerArgs)
                parser.add_argument("--config", type=str, help="A file.")
                parser.add_argument("--enable-dp-attention", action=DeprecatedStoreTrueAction, help="Old.")

        _INPUT_NAMESPACES = [
            Sched,
            Mem,
        ]
    ''',
    SRT + "/environ.py": '''
        import os

        class EnvField:
            def get(self):
                return os.getenv(self.name)

        class Envs:
            SGLANG_ENABLE_X = EnvBool(False)
            SGLANG_NEW = EnvIntWithAlias(4, deprecated_name="SGLANG_OLD")

        envs = Envs()

        _DEPRECATED_ENVS = {
            "SGLANG_GONE": _DeprecatedEnv(note="removed"),
        }
    ''',
    SRT + "/managers/scheduler.py": '''
        import os
        from sglang.srt.environ import envs
        from sglang.srt.utils.common import get_bool_env_var

        def f():
            a = os.environ.get("SGLANG_LITERAL_A")
            b = get_bool_env_var("SGLANG_LITERAL_B")
            os.environ["SGLANG_WRITTEN_ONLY"] = "1"
            c = envs.SGLANG_ENABLE_X.get()
            for name in ["SGLANG_LOOP_1", "SGLANG_LOOP_2"]:
                os.getenv(name)
            return a, b, c

        def _flag(name, default):
            return os.getenv(name) or default

        def g():
            return _flag("SGLANG_THROUGH_HELPER", "0")
    ''',
    SRT + "/managers/dyn.py": '''
        import os

        def h(cfg):
            return os.environ.get(cfg.var_name)
    ''',
    "python/sglang/kernels/k.cuh": '''
        const char* kVar = "SGLANG_KERNEL_CONST";
        void f() { std::getenv(kVar); std::getenv("SGLANG_KERNEL_LIT"); }
    ''',
    "python/sglang/utils.py": "",
    "python/sglang/launch_server.py": "",
    "python/sglang/cli/serve.py": "",
    "rust/x/src/lib.rs": '''
        fn f() { std::env::var("SGLANG_RS_VAR"); }
        #[cfg(test)]
        mod tests { fn t() { std::env::var("SGLANG_RS_TEST_ONLY"); } }
    ''',
}

DYN = "python/sglang/srt/managers/dyn.py: os.environ.get(cfg.var_name)"
ENVFIELD = "python/sglang/srt/environ.py: os.getenv(self.name)"


def make_tree(root, files):
    for rel, text in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(text))


def status_for(src, dyn_names=None):
    names = gen_fields.config_names(src)
    envs = gen_fields.env_names(src, [], None)
    fields = {}
    for n in names:
        fields["config:" + n] = {"status": "not_modelled", "evidence": {"issue": "inference-sim/inference-sim#2344"}, "estimation": {"rule": "none"}}
    for n in envs + (dyn_names or []):
        fields["env:" + n] = {"status": "not_modelled", "evidence": {"issue": "inference-sim/inference-sim#2460"}, "estimation": {"rule": "none"}}
    return {"engine_version": "0.5.23", "extra_env": [], "dynamic_reads": {DYN: {"reason": "r", "names": dyn_names or []},
                                                                   ENVFIELD: {"reason": "EnvField.get"}}, "fields": fields}


class GenFieldsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = self.tmp.name
        make_tree(self.src, FILES)

    def tearDown(self):
        self.tmp.cleanup()

    def test_config_names_are_every_declared_field_and_the_hand_registered_flags(self):
        names = gen_fields.config_names(self.src)
        self.assertEqual(names, ["config", "enable_dp_attention", "kv_mode", "max_running_requests", "page_size",
                                 "plain_field", "schedule_policy", "tp_size"])
        self.assertNotIn("dp_rank", names, "a Derived attribute is computed by SGLang, not set")
        self.assertNotIn("attn_tp_size", names, "nor is an annotated one")

    def test_the_no_cli_rule(self):
        decls, cli_only = gen_fields.server_args_declarations(self.src)
        self.assertEqual(decls["enable_dp_attention"]["flags"], [])
        self.assertTrue(decls["enable_dp_attention"]["no_cli"])
        self.assertEqual(decls["enable_dp_attention"]["manual_flags"][0]["flags"], ["--enable-dp-attention"])
        self.assertTrue(decls["enable_dp_attention"]["manual_flags"][0]["deprecated"])
        self.assertEqual(decls["plain_field"]["flags"], [], "no Arg metadata: add_cli_args_from_dataclass skips it")
        self.assertTrue(decls["plain_field"]["no_arg"])
        self.assertEqual(decls["tp_size"]["flags"], ["--tensor-parallel-size", "--tp"])
        self.assertEqual(decls["max_running_requests"]["flags"], ["--max-running-requests"])
        self.assertEqual(list(cli_only), ["config"])

    def test_declarations_carry_type_default_and_choices(self):
        decls, _ = gen_fields.server_args_declarations(self.src)
        self.assertEqual(decls["schedule_policy"]["choices"], ["lpm", "fcfs"], "a choices constant is followed to choices.py")
        self.assertEqual(decls["schedule_policy"]["default"], "fcfs")
        self.assertEqual(decls["kv_mode"]["choices"], ["auto", "fp8"], "a Literal gives the choices")
        self.assertIsNone(decls["kv_mode"]["default"])
        self.assertEqual(decls["page_size"]["namespace"], "memory")
        self.assertTrue(decls["page_size"]["resolvable"])
        self.assertEqual(decls["max_running_requests"]["type"], "Optional[int]")

    def test_env_names_follow_the_read_rule(self):
        names = gen_fields.env_names(self.src, [], None)
        for n in ("SGLANG_ENABLE_X", "SGLANG_NEW", "SGLANG_OLD", "SGLANG_GONE", "SGLANG_LITERAL_A", "SGLANG_LITERAL_B",
                  "SGLANG_LOOP_1", "SGLANG_LOOP_2", "SGLANG_THROUGH_HELPER", "SGLANG_KERNEL_CONST", "SGLANG_KERNEL_LIT",
                  "SGLANG_RS_VAR"):
            self.assertIn(n, names)
        self.assertNotIn("SGLANG_WRITTEN_ONLY", names, "a variable SGLang only writes is not read")
        self.assertNotIn("SGLANG_RS_TEST_ONLY", names, "Rust test code is not read")

    def test_build_writes_the_list_when_every_name_has_a_status(self):
        doc = gen_fields.build(self.src, status_for(self.src, ["SGLANG_DYN"]))
        names = [(f["source"], f["name"]) for f in doc["fields"]]
        self.assertIn(("env", "SGLANG_DYN"), names, "a dynamic read's traced names are listed")
        self.assertEqual(doc["engine"], {"kind": "sglang", "version": "0.5.23"})

    def test_known_bad_a_declared_field_with_no_status(self):
        st = status_for(self.src)
        del st["fields"]["config:page_size"]
        with self.assertRaises(SystemExit) as e:
            gen_fields.build(self.src, st)
        self.assertIn("no status for 1 fields SGLang reads", str(e.exception))
        self.assertIn("config:page_size", str(e.exception))

    def test_known_bad_a_status_for_a_field_sglang_does_not_declare(self):
        st = status_for(self.src)
        st["fields"]["config:not_a_field"] = {"status": "not_modelled", "evidence": {"issue": "inference-sim/inference-sim#2344"},
                                              "estimation": {"rule": "none"}}
        with self.assertRaises(SystemExit) as e:
            gen_fields.build(self.src, st)
        self.assertIn("config:not_a_field", str(e.exception))

    def test_known_bad_an_unlisted_dynamic_read(self):
        st = status_for(self.src)
        del st["dynamic_reads"][DYN]
        with self.assertRaises(SystemExit) as e:
            gen_fields.build(self.src, st)
        self.assertIn(DYN, str(e.exception))

    def test_known_bad_an_envs_attribute_that_is_not_a_field(self):
        files = dict(FILES)
        files[SRT + "/managers/bad.py"] = "from sglang.srt.environ import envs\nx = envs.SGLANG_NOT_DECLARED.get()\n"
        with tempfile.TemporaryDirectory() as d:
            make_tree(d, files)
            with self.assertRaises(SystemExit) as e:
                gen_fields.literal_env_reads(d)
            self.assertIn("SGLANG_NOT_DECLARED", str(e.exception))

    def test_known_bad_a_field_declared_by_two_namespaces(self):
        files = dict(FILES)
        files[SRT + "/arg_groups/fields/mem.py"] = FILES[SRT + "/arg_groups/fields/mem.py"].rstrip() + \
            "\n            tp_size: A[int, \"again\"] = 2\n"
        with tempfile.TemporaryDirectory() as d:
            make_tree(d, files)
            with self.assertRaises(SystemExit) as e:
                gen_fields.config_names(d)
            self.assertIn("tp_size is declared by both", str(e.exception))

    def test_check_mode_fails_on_a_stale_file(self):
        st = status_for(self.src, ["SGLANG_DYN"])
        with tempfile.TemporaryDirectory() as d:
            sp = os.path.join(d, "status.json")
            with open(sp, "w") as f:
                json.dump(st, f)
            good = os.path.join(d, "fields.json")
            with open(good, "w") as f:
                f.write(gen_fields.render(gen_fields.build(self.src, st)))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(gen_fields.main([self.src, sp, "--check", good]), 0)
            with open(good) as f:
                text = f.read()
            with open(good, "w") as f:
                f.write(text.replace('"page_size"', '"page_size_renamed"'))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(gen_fields.main([self.src, sp, "--check", good]), 1)


if __name__ == "__main__":
    unittest.main()
