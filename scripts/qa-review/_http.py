#!/usr/bin/env python3
"""qa-review shared HTTP client — one retrying chat-completions POST.

The single canonical LLM transport for ``answerer.py``, ``questioner.py`` and
``adjudicator.py`` (issue #1833). Each of the three used to issue its own single
``urllib.request.urlopen`` with no ``timeout=`` and no retry, so one transient
gateway error crashed the script outright: on PR #1832 a ``504 Gateway Time-out``
killed the answerer ~30 minutes into its tool loop, which wrote no ``QA-VERDICT``
marker, so the L1 gate read ``QA_VERDICT=MISSING`` and stopped the whole delivery
for a human. Re-dispatching cleared it — the failure self-heals, but the loop
could not retry itself. A transient blip now costs a short backoff instead.

The retry is on the *single failed call*, with the caller's ``messages`` history
untouched, so an answerer that trips a 504 on turn 18 resumes at turn 18 rather
than throwing away seventeen turns of tool work.

Retry policy:

* **Retried** — HTTP ``408``/``429`` and every ``5xx`` (notably ``502``/``503``/
  ``504``), plus connection-level failures (``URLError``, socket timeout,
  connection reset, a disconnect mid-read). These are the errors that self-heal.
* **Not retried** — every other ``4xx``. ``400``/``401``/``403``/``404`` are
  deterministic configuration errors (bad payload, wrong key, unknown model);
  retrying them cannot succeed and only burns CI wall-clock.
* ``Retry-After`` is honoured when present and numeric (capped, so a large or
  hostile value cannot park the job); otherwise exponential backoff with jitter.
* **Fail-closed on genuine exhaustion.** When the attempt budget runs out the
  last error is re-raised after an explicit stderr diagnostic, so the caller
  still dies without a marker and the gate still reads ``MISSING`` =>
  ``needs-human``. Only the single-blip hair-trigger is removed; a sustained
  outage still (correctly) stops for a human.

Env overrides (all optional, for CI tuning; an unusable value is reported on
stderr and ignored rather than crashing the review):

  QA_HTTP_TIMEOUT        per-attempt request timeout, seconds (default 600)
  QA_HTTP_MAX_ATTEMPTS   total attempts including the first (default 5)
  QA_HTTP_BACKOFF        first backoff, seconds, doubled per retry (default 2)
"""

import http.client
import json
import math
import os
import random
import socket
import sys
import time
import urllib.error
import urllib.request

# Transient statuses that are not 5xx: 408 Request Timeout and 429 Too Many
# Requests. Everything >= 500 is treated as transient by is_retryable().
RETRY_STATUSES = frozenset((408, 429))

# Ceilings. BACKOFF_CAP stops the doubling from growing without bound;
# RETRY_AFTER_CAP stops a server-supplied delay from parking the CI job.
BACKOFF_CAP = 60.0
RETRY_AFTER_CAP = 120.0

# Cap on the doubling exponent. 2**30 already dwarfs BACKOFF_CAP, so this changes
# no delay any configuration can actually produce — it only stops a large
# QA_HTTP_MAX_ATTEMPTS from computing 2**N as a bignum and overflowing the float
# multiply into an OverflowError instead of backing off.
BACKOFF_MAX_SHIFT = 30


def _env_number(name, default, cast):
    """Positive finite number from the environment, else default.

    An absent variable is the normal case. A malformed, non-positive or
    non-finite one is reported on stderr and ignored: this is delivery-loop
    infrastructure, and a typo'd tuning knob must not be the thing that turns a
    review into MISSING (R1 — never fail silently, but never fail fatally on a
    knob either). ``nan``/``inf`` parse fine as floats and would otherwise reach
    ``urlopen(timeout=...)`` or ``time.sleep()``, so they are rejected here.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        value = None
    if value is None or not math.isfinite(value) or value <= 0:
        sys.stderr.write(
            "qa-review http: ignoring unusable %s=%r, using %r\n" % (name, raw, default)
        )
        return default
    return value


# 600s per attempt is ~8x the incident's observed average turn (~30 min over up
# to 24 turns), so it bounds a genuinely hung connection without turning a slow
# but working completion into a failure — which would cause the very MISSING
# this change exists to prevent. Before this there was no timeout at all, so a
# hang consumed the whole 120-minute verify job.
TIMEOUT = _env_number("QA_HTTP_TIMEOUT", 600.0, float)
MAX_ATTEMPTS = _env_number("QA_HTTP_MAX_ATTEMPTS", 5, int)
BACKOFF = _env_number("QA_HTTP_BACKOFF", 2.0, float)


def sleep(seconds):
    """Wait between attempts.

    A named module-level seam so tests can exercise the real retry policy
    without spending the real backoff.
    """
    time.sleep(seconds)


def send(req, timeout):
    """Perform one HTTP round-trip and return the decoded response body.

    The only place this module touches the network, and the seam tests replace
    with a fake transport — everything above it (classification, backoff,
    exhaustion) is then covered for real. ``timeout`` is always passed
    explicitly: without it urllib blocks on the global default (usually none),
    so a hung connection hangs the review instead of failing into a retry.
    """
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def is_retryable(exc):
    """True when exc is a transient failure worth another attempt."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRY_STATUSES or exc.code >= 500
    # HTTPError is a URLError subclass, so it must be classified first (above)
    # or a non-retryable 4xx would be swept up as a generic URLError here.
    # URLError covers DNS/connect failures; socket.timeout is what the timeout=
    # above raises; ConnectionError covers a reset peer; HTTPException covers a
    # disconnect or truncated read mid-response.
    return isinstance(
        exc,
        (
            urllib.error.URLError,
            socket.timeout,
            ConnectionError,
            http.client.HTTPException,
        ),
    )


def retry_after_seconds(exc):
    """The response's ``Retry-After`` in seconds, or None when unusable.

    Only the delay-seconds form is honoured. The HTTP-date form is legal but
    would require trusting the gateway's clock against the runner's, so it falls
    through to the backoff schedule instead of guessing at a skew. The value is
    capped so a huge (or hostile) delay cannot park the job.
    """
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    # A header is the one input here an upstream can choose freely, so a negative
    # or non-finite value must fall through to the backoff schedule rather than
    # reach time.sleep().
    if not math.isfinite(value) or value < 0:
        return None
    return min(value, RETRY_AFTER_CAP)


def retry_delay(exc, attempt):
    """Seconds to wait before attempt+1 (attempt is 1-based).

    ``Retry-After`` wins when the server gave a usable one — it knows its own
    rate-limit window. Otherwise exponential backoff with half jitter: the wait
    is drawn from ``[w/2, w]`` for a doubling window ``w``, which spreads
    concurrent retries without ever yielding a ~0 wait that would hammer a
    gateway that is already struggling.
    """
    after = retry_after_seconds(exc)
    if after is not None:
        return after
    window = min(BACKOFF * (2 ** min(attempt - 1, BACKOFF_MAX_SHIFT)), BACKOFF_CAP)
    return window * (0.5 + random.random() / 2.0)


def describe(exc):
    """Short, log-friendly rendering of a transport failure."""
    if isinstance(exc, urllib.error.HTTPError):
        return "HTTP %s %s" % (exc.code, exc.reason)
    return "%s: %s" % (type(exc).__name__, exc)


def post_chat_completion(base_url, api_key, model, messages, tools=None):
    """POST one OpenAI-compatible chat completion, retrying transient failures.

    ``tools`` is omitted from the payload when falsy, so a tool-free caller (the
    questioner) sends exactly the payload it sent before this client existed.

    Returns the parsed response. Raises the last transport error once the
    attempt budget is exhausted, after naming it on stderr — the fail-closed
    contract: no marker, gate reads MISSING, delivery stops for a human.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode("utf-8")

    # Always make at least one attempt. _env_number already rejects a
    # non-positive budget, but a caller that lowered MAX_ATTEMPTS directly would
    # otherwise skip the loop entirely and reach `raise last` with last unset —
    # a TypeError about a None exception, hiding the real configuration mistake.
    attempts = max(1, MAX_ATTEMPTS)
    last = None
    for attempt in range(1, attempts + 1):
        # A fresh Request per attempt: a Request carries per-send state (unredirected
        # headers, host), so reusing one across retries is not guaranteed to be clean.
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + api_key)
        try:
            return json.loads(send(req, TIMEOUT))
        # Broad by design: every failure is classified on the next line, and a
        # deterministic one is re-raised untouched.
        except Exception as exc:
            if not is_retryable(exc):
                # Deterministic failure (a 4xx, or a malformed body): re-raise
                # untouched so the caller sees the same error it always did.
                raise
            last = exc
            if attempt >= attempts:
                break
            delay = retry_delay(exc, attempt)
            sys.stderr.write(
                "qa-review http: %s on attempt %d/%d, retrying in %.1fs\n"
                % (describe(exc), attempt, attempts, delay)
            )
            sleep(delay)

    sys.stderr.write(
        "qa-review http: giving up after %d attempts, last error %s — the script "
        "will exit without a verdict marker, so the gate reads MISSING and stops "
        "for a human\n" % (attempts, describe(last))
    )
    raise last
