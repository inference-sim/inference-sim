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

CONFIG = ["block_size", "chat_template", "enable_log_requests", "headless",
          "kv_transfer_config.kv_connector", "max_num_seqs", "model_tag", "port",
          "speculative_config.method", "speculative_config.num_speculative_tokens"]
ENV = ["NCCL_MAX_CTAS", "VLLM_A", "VLLM_B"]


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

    def test_env_names_are_envs_keys_and_communication_library_reads(self):
        self.assertEqual(gen_fields.env_names(self.src, []), ENV)

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
