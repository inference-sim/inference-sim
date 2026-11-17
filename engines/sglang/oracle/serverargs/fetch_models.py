# Original to blis-sglang.
"""Fetch each model of cases.json as configuration only (config.json and the tokenizer and
generation configs, no weights) at the revision cases.json pins, into <dir>/<name>/.

Usage: python -I oracle/serverargs/fetch_models.py <cases.json> <dir>
"""
import json
import sys


def main(cases_path: str, out: str) -> int:
    from huggingface_hub import snapshot_download

    with open(cases_path) as f:
        models = json.load(f)["models"]
    for name, m in sorted(models.items()):
        snapshot_download(m["repo"], revision=m["revision"], local_dir="%s/%s" % (out, name),
                          allow_patterns=["config.json", "generation_config.json", "tokenizer_config.json"])
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
