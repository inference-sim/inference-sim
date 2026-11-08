# Original to blis-vllm.
"""Tests for the block-size oracle's config.json source (modelconfig.py), with no network: the
fetcher is injected. Usage: python3 -I oracle/blocksize/test_modelconfig.py (stdlib only)."""
import hashlib
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import modelconfig as mc  # noqa: E402

BODY = json.dumps({"num_attention_heads": 48, "head_dim": 128}).encode()
SHA = hashlib.sha256(BODY).hexdigest()
REV = "d" * 40
YAML = f"""name: m
source:
  provider: huggingface
  repo: Owner/Model-1
  revision: {REV}
  retrieved: 2026-10-03
licence:
  id: LicenseRef-x
config:
  sha256: {SHA}
"""
URL = f"https://huggingface.co/Owner/Model-1/resolve/{REV}/config.json"


class Fetch:
    def __init__(self, data=BODY, exc=None):
        self.data, self.exc, self.urls = data, exc, []

    def __call__(self, url):
        self.urls.append(url)
        if self.exc:
            raise self.exc
        return self.data


class ModelConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = os.path.join(self.tmp.name, "model")
        os.makedirs(self.dir)
        self.cache = os.path.join(self.tmp.name, "cache")
        self.old = os.environ.get("BLIS_ORACLE_FETCH_CACHE")
        os.environ["BLIS_ORACLE_FETCH_CACHE"] = self.cache
        self.addCleanup(self.restore)

    def restore(self):
        if self.old is None:
            os.environ.pop("BLIS_ORACLE_FETCH_CACHE", None)
        else:
            os.environ["BLIS_ORACLE_FETCH_CACHE"] = self.old

    def write(self, name, data):
        with open(os.path.join(self.dir, name), "wb") as f:
            f.write(data if isinstance(data, bytes) else data.encode())

    def test_file_present_is_read_without_fetch(self):
        self.write("config.json", b'{"a": 1}')
        f = Fetch()
        self.assertEqual(mc.config_bytes("m", self.dir, f), b'{"a": 1}')
        self.assertEqual(f.urls, [])

    def test_absent_fetched_verified_and_cached(self):
        self.write("model.yaml", YAML)
        f = Fetch()
        self.assertEqual(mc.config_bytes("m", self.dir, f), BODY)
        self.assertEqual(f.urls, [URL])
        with open(os.path.join(self.cache, SHA), "rb") as c:
            self.assertEqual(c.read(), BODY)
        f2 = Fetch(exc=OSError("network down"))
        self.assertEqual(mc.config_bytes("m", self.dir, f2), BODY)   # served from the cache
        self.assertEqual(f2.urls, [])
        self.assertEqual(os.listdir(self.dir), ["model.yaml"])       # nothing written beside the model

    def test_hash_mismatch_refused_naming_both_digests(self):
        self.write("model.yaml", YAML)
        other = b'{"num_attention_heads": 1}'
        with self.assertRaises(mc.ConfigError) as cm:
            mc.config_bytes("m", self.dir, Fetch(other))
        msg = str(cm.exception)
        self.assertIn("m:", msg)
        self.assertIn(SHA, msg)
        self.assertIn(hashlib.sha256(other).hexdigest(), msg)
        self.assertFalse(os.path.exists(os.path.join(self.cache, SHA)))

    def test_model_yaml_lacking_keys_refused(self):
        for drop in ("repo", "revision", "sha256", "provider"):
            with self.subTest(drop=drop):
                text = "".join(l for l in YAML.splitlines(True) if f"  {drop}:" not in l)
                self.write("model.yaml", text)
                with self.assertRaises(mc.ConfigError) as cm:
                    mc.config_bytes("m", self.dir, Fetch())
                self.assertIn(drop, str(cm.exception))

    def test_model_yaml_malformed_values_refused(self):
        for old, new in ((REV, "main"), (SHA, "abc"), ("Owner/Model-1", "../x"), ("huggingface", "github"),
                         (f"revision: {REV}", f"revision: '{REV}'")):
            with self.subTest(new=new):
                self.write("model.yaml", YAML.replace(old, new, 1))
                with self.assertRaises(mc.ConfigError):
                    mc.config_bytes("m", self.dir, Fetch())

    def test_model_yaml_unreadable_refused(self):
        with self.assertRaises(mc.ConfigError):
            mc.config_bytes("m", self.dir, Fetch())

    def test_corrupted_cache_refused(self):
        self.write("model.yaml", YAML)
        mc.config_bytes("m", self.dir, Fetch())
        with open(os.path.join(self.cache, SHA), "wb") as c:
            c.write(b"tampered")
        f = Fetch()
        with self.assertRaises(mc.ConfigError) as cm:
            mc.config_bytes("m", self.dir, f)
        self.assertIn(SHA, str(cm.exception))
        self.assertIn(hashlib.sha256(b"tampered").hexdigest(), str(cm.exception))
        self.assertEqual(f.urls, [])   # no fallback to the network

    def test_fetch_failure_says_so_and_needs_network(self):
        self.write("model.yaml", YAML)
        with self.assertRaises(mc.ConfigError) as cm:
            mc.config_bytes("m", self.dir, Fetch(exc=OSError("network down")))
        self.assertIn("fetching", str(cm.exception))
        self.assertIn("network access", str(cm.exception))

    def test_http_fetcher_refuses_other_hosts(self):
        for url in ("http://huggingface.co/x", "https://example.com/x", "https://evilhuggingface.co/x"):
            with self.subTest(url=url):
                with self.assertRaises(mc.ConfigError):
                    mc.http_fetcher(url)


if __name__ == "__main__":
    unittest.main()
