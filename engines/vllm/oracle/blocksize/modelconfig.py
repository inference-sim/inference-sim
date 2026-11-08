# Original to blis-vllm.
"""A model's config.json for the block-size oracle: the catalog's file, or, for a model whose
licence does not let the catalog redistribute it, the same file fetched at the revision the
catalog's model.yaml pins and checked against the SHA-256 it records.

Standard library only. The fetch is injected (`fetcher`), so tests need no network.
"""
import hashlib
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request

HOSTS_OK = ("huggingface.co", ".huggingface.co", ".hf.co")
TIMEOUT_S = 60
MAX_BYTES = 8 << 20


class ConfigError(Exception):
    """The model's config.json cannot be had and verified; the oracle must not go on."""


def _host_allowed(host):
    host = (host or "").lower()
    return any(host == h or (h.startswith(".") and host.endswith(h)) for h in HOSTS_OK)


def parse_pin(text, model):
    """(repo, revision, sha256) from a model.yaml, read strictly.

    Only the scalar lines of the top-level `source:` and `config:` blocks are read, each a plain
    `key: value` at two spaces of indent. A quoted value, a repeated key or block, a missing key,
    a provider other than huggingface or a malformed value is refused."""
    blocks, cur = {}, None
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            m = re.fullmatch(r"([A-Za-z_]+):\s*(.*)", line)
            if not m:
                raise ConfigError(f"{model}: model.yaml line {n} is not a top-level key: {line!r}")
            cur = None
            if m.group(1) in ("source", "config"):
                if m.group(1) in blocks or m.group(2):
                    raise ConfigError(f"{model}: model.yaml has a repeated or inline {m.group(1)!r}")
                blocks[m.group(1)] = {}
                cur = m.group(1)
            continue
        if cur is None:
            continue
        m = re.fullmatch(r"  ([A-Za-z_][A-Za-z0-9_]*): ([A-Za-z0-9][^'\"#\s]*)\s*", line)
        if not m:
            raise ConfigError(f"{model}: model.yaml line {n} under {cur!r} is not a plain 'key: value': {line!r}")
        if m.group(1) in blocks[cur]:
            raise ConfigError(f"{model}: model.yaml repeats {cur}.{m.group(1)}")
        blocks[cur][m.group(1)] = m.group(2)
    try:
        provider, repo = blocks["source"]["provider"], blocks["source"]["repo"]
        rev, sha = blocks["source"]["revision"], blocks["config"]["sha256"]
    except KeyError as e:
        raise ConfigError(f"{model}: no config.json in the catalog and model.yaml lacks {e.args[0]!r} "
                          f"(needs source.provider, source.repo, source.revision, config.sha256)") from None
    if provider != "huggingface":
        raise ConfigError(f"{model}: source.provider is {provider!r}; only huggingface is fetched")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo):
        raise ConfigError(f"{model}: source.repo {repo!r} is not <owner>/<name>")
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise ConfigError(f"{model}: source.revision {rev!r} is not a 40-hex commit")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ConfigError(f"{model}: config.sha256 {sha!r} is not 64 hex digits")
    return repo, rev, sha


class _HubRedirect(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to https on huggingface.co or its CDN; the token goes to
    huggingface.co itself and nowhere else."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u = urllib.parse.urlsplit(urllib.parse.urljoin(req.full_url, newurl))
        if u.scheme != "https" or not _host_allowed(u.hostname):
            raise ConfigError(f"redirect to {newurl!r} refused: only https on huggingface.co or its CDN is allowed")
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and u.hostname.lower() != "huggingface.co":
            new.remove_header("Authorization")
        return new


def http_fetcher(url):
    """The bytes at url (https on huggingface.co only). HF_TOKEN, if set, is sent as a bearer token."""
    u = urllib.parse.urlsplit(url)
    if u.scheme != "https" or not _host_allowed(u.hostname):
        raise ConfigError(f"refusing to fetch {url!r}: only https on huggingface.co is allowed")
    req = urllib.request.Request(url, headers={"User-Agent": "blis-vllm-oracle"})
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
    with urllib.request.build_opener(_HubRedirect).open(req, timeout=TIMEOUT_S) as r:
        data = r.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ConfigError(f"{url} is larger than {MAX_BYTES} bytes")
    return data


def _cache_dir():
    d = os.environ.get("BLIS_ORACLE_FETCH_CACHE") or os.path.join(tempfile.gettempdir(), "blis-oracle-fetch-cache")
    os.makedirs(d, exist_ok=True)
    return d


def config_bytes(model, model_dir, fetcher=http_fetcher):
    """The bytes of the model's config.json: the catalog's file if present, else the file at the
    pinned revision, verified against the catalog's recorded SHA-256 (cache reads included)."""
    path = os.path.join(model_dir, "config.json")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    ypath = os.path.join(model_dir, "model.yaml")
    try:
        with open(ypath, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise ConfigError(f"{model}: there is no {path} and {ypath} cannot be read: {e}") from e
    repo, rev, want = parse_pin(text, model)
    cached = os.path.join(_cache_dir(), want)
    if os.path.exists(cached):
        with open(cached, "rb") as f:
            data = f.read()
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise ConfigError(f"{model}: the cached config.json {cached} has SHA-256 {got}, expected {want}; "
                              f"remove that cache file and rerun")
        return data
    url = f"https://huggingface.co/{repo}/resolve/{rev}/config.json"
    try:
        data = fetcher(url)
    except ConfigError:
        raise
    except (OSError, urllib.error.URLError) as e:
        raise ConfigError(f"{model}: config.json is not in the catalog (a derived-only model) and fetching {url} "
                          f"failed: {e}. The oracle needs network access to huggingface.co (in CI too) "
                          f"to fetch it at the pinned revision; it uses no other source.") from e
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise ConfigError(f"{model}: config.json fetched from {url} has SHA-256 {got}, "
                          f"but the catalog's model.yaml records {want}; refusing it")
    tmp = cached + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, cached)
    return data
