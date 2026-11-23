# Original to blis-vllm.
"""Tests for the engine-block migration: each rewrite, the refusals, and that a second run changes nothing."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import migrate  # noqa: E402
import migrate_go  # noqa: E402

OLD = """pools:
  - role: colocated
    engine:
      cache_dtype: fp8
      cudagraph_mode: FULL
      custom_ops: ["+rms_norm"]
      use_trtllm_attention: false
      speculative: {method: mtp, num_spec_tokens: 3, draft_model: d}
      dbo: {enable_dbo: true, dbo_decode_token_threshold: 32}
      eplb:
        enable_eplb: true
        window_size: 5
      kv_events: {enable_kv_cache_events: true}
      frontend: rust
      env: {A: "1"}
      cudagraph_capture_sizes: [1, 8]
    parallel: {tp: 8}
acceptance:
  value:
    num_spec_tokens: 7
"""
NEW = """pools:
  - role: colocated
    engine:
      kv_cache_dtype: fp8
      speculative_config: {method: mtp, num_speculative_tokens: 3, model: d}
      enable_dbo: true
      dbo_decode_token_threshold: 32
      enable_eplb: true
      eplb_config: {window_size: 5}
      kv_events_config: {enable_kv_cache_events: true}
      env: {A: "1", VLLM_USE_RUST_FRONTEND: "1"}
      cudagraph_capture_sizes: [1, 8]
      compilation_config: {cudagraph_mode: FULL, custom_ops: ["+rms_norm"]}
      attention_config: {use_trtllm_attention: false}
    parallel: {tp: 8}
acceptance:
  value:
    num_spec_tokens: 7
"""


class Migrate(unittest.TestCase):
    def test_rewrites_the_engine_block_and_nothing_else(self):
        self.assertEqual(migrate.migrate_text(OLD), NEW)

    def test_a_second_run_changes_nothing(self):
        self.assertEqual(migrate.migrate_text(NEW), NEW)

    def test_a_frontend_value_is_read_after_its_comment_and_quotes(self):
        for line in ("rust # keep", "'rust'", '"rust"  # keep'):
            got = migrate.migrate_text(f"e:\n  engine:\n    frontend: {line}\n    block_size: 16\n")
            self.assertIn('VLLM_USE_RUST_FRONTEND: "1"', got, line)
        for line in ("python", "python # keep", "'python'"):
            got = migrate.migrate_text(f"e:\n  engine:\n    frontend: {line}\n    block_size: 16\n")
            self.assertNotIn("VLLM_USE_RUST_FRONTEND", got, line)
            self.assertNotIn("frontend", got, line)

    def test_a_frontend_value_that_is_neither_rust_nor_python_is_refused(self):
        for line in ("Rust", "rusty # x", "'rust\"", "go"):
            with self.assertRaises(ValueError, msg=line):
                migrate.migrate_text(f"e:\n  engine:\n    frontend: {line}\n")

    def test_allreduce_backend_is_left_alone(self):
        src = "e:\n  engine:\n    allreduce_backend: nccl\n"
        self.assertEqual(migrate.migrate_text(src), src)

    def test_a_block_valued_key_is_refused_not_guessed(self):
        with self.assertRaises(ValueError):
            migrate.migrate_text("e:\n  engine:\n    custom_ops:\n      - x\n")

    def test_go_string_literals_are_rewritten_and_prose_is_not(self):
        got = migrate_go.fix_literal('"speculative: {method: x, num_spec_tokens: 2}\\ncache_dtype: fp8\\n"')
        self.assertEqual(got, '"speculative_config: {method: x, num_speculative_tokens: 2}\\nkv_cache_dtype: fp8\\n"')
        self.assertEqual(migrate_go.fix_literal('"a sentence about cache_dtype: and nothing"'),
                         '"a sentence about cache_dtype: and nothing"')


if __name__ == "__main__":
    unittest.main()
