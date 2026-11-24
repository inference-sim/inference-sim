"""The pinned-curve fetch retries a transient transport or 5xx/429 response, with a capped
backoff, and still verifies every file's hash after the retry, so a retry can only ever accept the
one payload curves.json pins. A non-retryable status (a 404) is not retried, and a run of
persistent failures gives up naming the URL and the attempts.

It injects a fake opener, so nothing is fetched over the network.
"""

from __future__ import annotations

import hashlib
import http.client
import sys
import urllib.error
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import fetch_inferencex_curves as fx  # noqa: E402


class _Resp:
    """A minimal urlopen result: a context manager whose read() returns the body."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return self._body


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url="https://example/x", code=code, msg="transient", hdrs=None, fp=None)


def _opener(seq):
    """An opener that, on each call, raises the next item if it is an exception or returns it."""
    it = iter(seq)
    calls = {"n": 0}

    def opener(url, timeout=0):
        calls["n"] += 1
        item = next(it)
        if isinstance(item, BaseException):
            raise item
        return _Resp(item)

    return opener, calls


def test_read_url_retries_a_transient_503_then_returns_the_payload():
    opener, calls = _opener([_http_error(503), b"body"])
    slept = []
    assert fx.read_url("https://example/x", opener=opener, sleep=slept.append) == b"body"
    assert calls["n"] == 2
    assert slept == [1.0]  # one backoff before the second attempt


def test_read_url_retries_a_connection_reset():
    # A reset is a ConnectionResetError (an OSError), not a urllib.error.URLError.
    opener, calls = _opener([ConnectionResetError("connection reset by peer"), b"body"])
    assert fx.read_url("https://example/x", opener=opener, sleep=lambda _: None) == b"body"
    assert calls["n"] == 2


def test_read_url_retries_a_bare_remote_disconnect():
    # urllib wraps h.request() in URLError but not h.getresponse(); a server that accepts then drops
    # the connection raises a bare http.client.RemoteDisconnected, which must still be retried.
    opener, calls = _opener([http.client.RemoteDisconnected("remote end closed without response"), b"body"])
    assert fx.read_url("https://example/x", opener=opener, sleep=lambda _: None) == b"body"
    assert calls["n"] == 2


def test_read_url_retries_an_incomplete_read():
    # A reset during r.read() raises http.client.IncompleteRead (an HTTPException, not an OSError).
    opener, calls = _opener([http.client.IncompleteRead(partial=b"half"), b"body"])
    assert fx.read_url("https://example/x", opener=opener, sleep=lambda _: None) == b"body"
    assert calls["n"] == 2


def test_read_url_retries_a_wrapped_transport_error():
    # A DNS failure or refused connect reaches us as urllib.error.URLError (an OSError subclass, but
    # not an HTTPError), which _is_retryable treats as a transient transport error. This pins that
    # path so narrowing the except clause to ConnectionError later would be caught here.
    opener, calls = _opener([urllib.error.URLError(OSError("name resolution failed")), b"body"])
    assert fx.read_url("https://example/x", opener=opener, sleep=lambda _: None) == b"body"
    assert calls["n"] == 2


def test_read_url_retries_a_429():
    opener, calls = _opener([_http_error(429), b"body"])
    assert fx.read_url("https://example/x", opener=opener, sleep=lambda _: None) == b"body"
    assert calls["n"] == 2


def test_read_url_gives_up_after_max_attempts_on_persistent_failure():
    opener, calls = _opener([_http_error(503)] * fx.MAX_ATTEMPTS)
    slept = []
    with pytest.raises(SystemExit) as e:
        fx.read_url("https://example/x", opener=opener, sleep=slept.append)
    assert "https://example/x" in str(e.value) and "attempts failed" in str(e.value)
    assert calls["n"] == fx.MAX_ATTEMPTS
    assert len(slept) == fx.MAX_ATTEMPTS - 1  # a backoff between attempts, none after the last


def test_read_url_does_not_retry_a_404():
    opener, calls = _opener([_http_error(404)])
    with pytest.raises(urllib.error.HTTPError):
        fx.read_url("https://example/x", opener=opener, sleep=lambda _: None)
    assert calls["n"] == 1


def test_fetch_retries_then_writes_the_hash_verified_payload(tmp_path):
    payload = b"golden-curve-bytes"
    lock = {"repository": "https://github.com/o/r", "commit": "deadbeef", "path": "p",
            "files": {"f.json": hashlib.sha256(payload).hexdigest()}}
    opener, calls = _opener([_http_error(503), payload])
    n = fx.fetch(tmp_path, lock, opener=opener, sleep=lambda _: None)
    assert n == 1
    assert calls["n"] == 2
    assert (tmp_path / "f.json").read_bytes() == payload


def test_fetch_still_rejects_a_wrong_hash_after_a_retry(tmp_path):
    lock = {"repository": "https://github.com/o/r", "commit": "deadbeef", "path": "p",
            "files": {"f.json": "0" * 64}}
    opener, _ = _opener([_http_error(503), b"not-the-pinned-bytes"])
    with pytest.raises(SystemExit) as e:
        fx.fetch(tmp_path, lock, opener=opener, sleep=lambda _: None)
    assert "SHA-256" in str(e.value)
    assert not (tmp_path / "f.json").exists()
