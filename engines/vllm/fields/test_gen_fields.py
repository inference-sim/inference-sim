# Original to blis-vllm.
"""Tests for gen_fields.py over a small synthetic vLLM tree. Run: python3 -I test_gen_fields.py"""

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

FILES = {
    "vllm/envs.py": """
        import os
        environment_variables = {
            "VLLM_B": lambda: os.getenv("VLLM_B"),
            "VLLM_A": lambda: os.getenv("VLLM_A"),
        }
    """,
    "vllm/config/sched.py": """
        from vllm.config.utils import config
        @config
        class SpeculativeConfig:
            method: str = "eagle"
            num_speculative_tokens: int = 1
            target_model_config: SkipValidation[ModelConfig] = None
        @config
        class KVConfig:
            kv_connector: str | None = None
            internal: int = field(default=0, init=False)
    """,
    "vllm/engine/arg_utils.py": """
        class EngineArgs:
            max_num_seqs: int = 128
            block_size: int | None = None
            speculative_config: dict[str, Any] | None = None
            kv_transfer_config: KVConfig | None = None
            _hidden: int = 0
            def add_cli_args(parser):
                parser.add_argument("--max-num-seqs")
                parser.add_argument("--block-size")
                parser.add_argument("--speculative-config")
                parser.add_argument("--kv-transfer-config")
        class AsyncEngineArgs(EngineArgs):
            enable_log_requests: bool = False
            def add_cli_args(parser):
                parser.add_argument("--enable-log-requests")
    """,
    "vllm/entrypoints/launchers/cli_args.py": """
        class BaseFrontendArgs:
            chat_template: str | None = None
        class FrontendArgs(BaseFrontendArgs):
            port: int = 8000
        def make_arg_parser(parser):
            parser.add_argument("model_tag", nargs="?")
            parser.add_argument("--headless")
            return parser
    """,
    "vllm/distributed/comm.py": """
        import os
        os.environ["NCCL_MAX_CTAS"] = "1"
        x = os.environ.get("SOME_OTHER")
    """,
}

# One file per read pattern of the READ RULE, each naming the variable it reads.
READS = {
    "vllm/a/get.py": 'import os\nx = os.environ.get("R_ENVIRON_GET", "0")\n',
    "vllm/a/getenv.py": 'import os\nx = os.getenv("R_OS_GETENV")\n',
    "vllm/a/imported.py": 'from os import environ, getenv\nx = environ.get("R_IMPORTED_GET")\ny = getenv("R_IMPORTED_GETENV")\n',
    "vllm/a/subscript.py": 'import os\nx = os.environ["R_SUBSCRIPT"]\n',
    "vllm/a/contains.py": 'import os\nif "R_IN" in os.environ or "R_NOT_IN" not in os.environ:\n    pass\n',
    "vllm/a/setdefault.py": 'import os\nos.environ.setdefault("R_SETDEFAULT", "1")\n',
    "vllm/a/pop.py": 'import os\nold = os.environ.pop("R_POP_USED", None)\nos.environ.pop("W_POP_DROPPED", None)\n',
    "vllm/a/is_set.py": 'from vllm import envs\nif envs.is_set("R_IS_SET"):\n    pass\n',
    "vllm/a/const.py": 'import os\nNAME = "R_CONSTANT"\nx = os.environ.get(NAME)\n',
    "vllm/a/ifexp.py": 'import os\nn = "R_IFEXP_A" if flag else "R_IFEXP_B"\nx = os.environ.get(n)\n',
    "vllm/a/loop.py": 'import os\nfor k in ("R_LOOP_A", "R_LOOP_B"):\n    x = os.environ.get(k)\nD = {"R_LOOP_C": 1}\nold = {k: os.environ.get(k) for k in D}\n',
    "vllm/a/alias.py": 'import os\ndef f(environ=None):\n    env = os.environ if environ is None else environ\n    return env.get("R_ALIAS")\n',
    "vllm/a/write.py": 'import os\nos.environ["W_ONLY_WRITTEN"] = "1"\ndel os.environ["W_DELETED"]\n',
    "vllm/a/comm.py": 'pins = {"NCCL_PROTO": "Simple"}\nchild = {}\nchild.setdefault("UCX_TLS", "rc")\n',
    "vllm/a/envs_attr.py": 'from vllm import envs\nx = envs.VLLM_A\n',
    "vllm/platforms/rocm.py": 'import os\nx = os.environ.get("R_PLATFORM_MODULE")\n',
    "csrc/k.cu": 'const char* a = std::getenv("R_CPP_STD");\nconst char* b = getenv("R_CPP_PLAIN");\n'
                 'constexpr const char* kName = "R_CPP_CONST";\nconst char* c = std::getenv(kName);\n',
    "rust/src/server/src/x.rs": 'const N: &str = "R_RUST_CONST";\nfn f() { std::env::var("R_RUST_VAR"); env::var_os(N); }\n'
                                '#[cfg(test)]\nmod tests { fn t() { std::env::var("W_RUST_TEST"); } }\n',
    # A clap attribute reads its variable too; a commented-out one and one in test code do not.
    "rust/src/cmd/src/cli.rs": '#[derive(Parser)]\nstruct A {\n    #[arg(long, env = "R_CLAP_ENV")]\n    a: String,\n'
                               '    #[arg(long, /* env = "W_CLAP_BLOCK_COMMENT" */)]\n    b: String,\n'
                               '    // env = "W_CLAP_LINE_COMMENT",\n    c: String,\n}\n'
                               '#[cfg(test)]\nmod t { #[arg(env = "W_CLAP_TEST")] x: u8 }\n',
    # EXCLUDED: separate commands and build steps, never the serving process.
    "vllm/collect_env.py": 'import os\nx = os.environ.get("X_COLLECT_ENV")\n',
    "vllm/benchmarks/b.py": 'import os\nx = os.environ.get("X_BENCHMARKS")\n',
    "vllm/entrypoints/cli/openai.py": 'import os\nx = os.environ.get("X_CLI_CLIENT")\n',
    "rust/src/bench/src/b.rs": 'fn f() { std::env::var("X_RUST_BENCH"); }\n',
    "rust/proto/build.rs": 'fn f() { std::env::var("X_BUILD_RS"); }\n',
    "csrc/gen.py": 'import os\nx = os.environ.get("X_CSRC_GENERATOR")\n',
    "tests/t.py": 'import os\nx = os.environ.get("X_TESTS")\n',
    "setup.py": 'import os\nx = os.environ.get("X_SETUP")\n',
}
READ_NAMES = sorted(["R_ENVIRON_GET", "R_OS_GETENV", "R_IMPORTED_GET", "R_IMPORTED_GETENV", "R_SUBSCRIPT",
                     "R_IN", "R_NOT_IN", "R_SETDEFAULT", "R_POP_USED", "R_IS_SET", "R_CONSTANT",
                     "R_IFEXP_A", "R_IFEXP_B", "R_ALIAS", "R_LOOP_A", "R_LOOP_B", "R_LOOP_C", "R_PLATFORM_MODULE",
                     "R_CPP_STD", "R_CPP_PLAIN", "R_CPP_CONST", "R_RUST_CONST", "R_RUST_VAR", "R_CLAP_ENV",
                     "NCCL_PROTO", "UCX_TLS"])

CONFIG = ["block_size", "chat_template", "enable_log_requests", "headless",
          "kv_transfer_config.kv_connector", "max_num_seqs", "model_tag", "port",
          "speculative_config.method", "speculative_config.num_speculative_tokens"]
ENV = ["NCCL_MAX_CTAS", "SOME_OTHER", "VLLM_A", "VLLM_B"]


def entry():
    return {"status": "not_modelled", "evidence": {"issue": "inference-sim/blis-vllm#1"},
            "estimation": {"rule": "none"}}


class GenFieldsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = self.tmp.name
        for path, body in FILES.items():
            p = os.path.join(self.src, path)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(textwrap.dedent(body))
        self.status = {"engine_version": "0.32.0", "fields": {}}
        for n in CONFIG:
            self.status["fields"]["config:" + n] = entry()
        for n in ENV:
            self.status["fields"]["env:" + n] = entry()
        self.status_path = os.path.join(self.src, "status.json")
        self.write_status()

    def tearDown(self):
        self.tmp.cleanup()

    def write_status(self):
        with open(self.status_path, "w") as f:
            json.dump(self.status, f)

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = gen_fields.main(list(args))
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_config_names_are_engine_args_expanded_into_their_config_classes(self):
        self.assertEqual(gen_fields.config_names(self.src), CONFIG)

    def test_env_names_are_envs_keys_literal_reads_and_communication_library_writes(self):
        self.assertEqual(gen_fields.env_names(self.src, []), ENV)

    def write(self, files):
        for path, body in files.items():
            p = os.path.join(self.src, path)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(body)

    def reads(self, files):
        self.write(files)
        return gen_fields.literal_env_reads(self.src)

    def test_each_read_pattern_is_found_and_records_where(self):
        reads, dynamic = self.reads(READS)
        for name in READ_NAMES:
            with self.subTest(name=name):
                self.assertIn(name, reads)
        self.assertEqual(reads["R_SUBSCRIPT"], ["vllm/a/subscript.py:2"])
        self.assertEqual(reads["R_CPP_CONST"], ["csrc/k.cu:4"])
        self.assertEqual(reads["R_RUST_CONST"], ["rust/src/server/src/x.rs:2"])
        self.assertEqual(reads["R_CLAP_ENV"], ["rust/src/cmd/src/cli.rs:3"])
        self.assertEqual(dynamic, [])

    def test_a_variable_vllm_only_writes_is_not_a_read(self):
        reads, _ = self.reads(READS)
        for name in ("W_ONLY_WRITTEN", "W_DELETED", "W_POP_DROPPED", "W_RUST_TEST",
                     "W_CLAP_BLOCK_COMMENT", "W_CLAP_LINE_COMMENT", "W_CLAP_TEST"):
            self.assertNotIn(name, reads)

    def test_the_excluded_paths_and_only_those_are_skipped(self):
        reads, _ = self.reads(READS)
        for name in ("X_COLLECT_ENV", "X_BENCHMARKS", "X_CLI_CLIENT", "X_RUST_BENCH", "X_BUILD_RS",
                     "X_CSRC_GENERATOR", "X_TESTS", "X_SETUP"):
            self.assertNotIn(name, reads)
        names = {n for n in reads if n.startswith(("R_", "NCCL_", "UCX_", "SOME_"))}
        self.assertEqual(sorted(names), sorted(READ_NAMES + ["NCCL_MAX_CTAS", "SOME_OTHER"]))

    def test_a_module_named_for_another_platform_is_read(self):
        reads, _ = self.reads(READS)
        self.assertEqual(reads["R_PLATFORM_MODULE"], ["vllm/platforms/rocm.py:2"])

    def test_an_envs_attribute_that_is_not_a_key_stops_the_generator(self):
        self.write({"vllm/a/bad.py": "from vllm import envs\nx = envs.VLLM_NOT_A_KEY\n"})
        with self.assertRaises(SystemExit) as e:
            gen_fields.literal_env_reads(self.src)
        self.assertIn("envs.VLLM_NOT_A_KEY", str(e.exception))

    def test_a_dynamic_read_must_be_listed_and_its_names_are_added(self):
        self.write({"vllm/a/dyn.py": "import os\ndef f(name):\n    return os.environ.get(name)\n"})
        call = "vllm/a/dyn.py: os.environ.get(name)"
        with self.assertRaises(SystemExit) as e:
            gen_fields.env_names(self.src, [], {})
        self.assertIn(call, str(e.exception))
        names = gen_fields.env_names(self.src, [], {call: {"reason": "r", "names": ["R_TRACED"]}})
        self.assertIn("R_TRACED", names)
        with self.assertRaises(SystemExit) as e:
            gen_fields.env_names(self.src, [], {call: {"reason": "r"}, "vllm/gone.py: os.getenv(x)": {"reason": "r"}})
        self.assertIn("vllm/gone.py", str(e.exception))

    def test_output_is_sorted_by_name_then_source(self):
        code, out, _ = self.run_main(self.src, self.status_path)
        self.assertEqual(code, 0)
        doc = json.loads(out)
        keys = [(f["name"], f["source"]) for f in doc["fields"]]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(keys), len(CONFIG) + len(ENV))
        self.assertEqual(list(doc), ["kind", "engine", "fields"])
        self.assertEqual(list(doc["fields"][0]), ["name", "source", "status", "evidence", "estimation"])

    def test_check_fails_on_a_changed_file_and_passes_on_the_generated_one(self):
        _, out, _ = self.run_main(self.src, self.status_path)
        path = os.path.join(self.src, "fields.json")
        with open(path, "w") as f:
            f.write(out)
        self.assertEqual(self.run_main(self.src, self.status_path, "--check", path)[0], 0)
        with open(path, "w") as f:
            f.write(out.replace('"not_modelled"', '"modelled"', 1))
        code, diff, err = self.run_main(self.src, self.status_path, "--check", path)
        self.assertEqual(code, 1)
        self.assertIn("modelled", diff)
        self.assertIn("regenerate", err)

    def test_a_field_vllm_reads_without_a_status_is_refused(self):
        del self.status["fields"]["config:block_size"]
        self.write_status()
        code, _, _ = self.run_main(self.src, self.status_path)
        self.assertIn("config:block_size", str(code))

    def test_a_status_for_a_name_vllm_does_not_read_is_refused(self):
        self.status["fields"]["config:gone"] = entry()
        self.write_status()
        code, _, _ = self.run_main(self.src, self.status_path)
        self.assertIn("config:gone", str(code))


if __name__ == "__main__":
    unittest.main()
