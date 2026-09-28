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
  connection reset, a disconnect mid-read), plus a response body that will not
  decode (a proxy blip can answer ``200`` with an HTML error page or a truncated
  read, which urllib reports as a perfectly clean response). These are the
  errors that self-heal.
* **Not retried** — every other ``4xx``. ``400``/``401``/``403``/``404`` are
  deterministic configuration errors (bad payload, wrong key, unknown model);
  retrying them cannot succeed and only burns CI wall-clock. The gateway's own
  response body is written to stderr before the error is re-raised, because the
  status line alone does not say *which* thing the request got wrong.
* ``Retry-After`` is honoured **in full** whenever the response carries a usable
  one — both the delay-seconds and the HTTP-date form, and the value is never
  shortened. Otherwise exponential backoff with jitter.
* **Bounded total wait.** The waiting between attempts is capped in aggregate
  (``MAX_TOTAL_WAIT``), not per wait: a delay that does not fit the remaining
  budget ends the retry loop instead of being truncated, because retrying
  earlier than the gateway asked only earns the same error again. That also
  stops a hostile ``Retry-After: 99999`` from parking the job.
* **Fail-closed on genuine exhaustion.** When the attempt budget (or the wait
  budget) runs out the last error is re-raised after an explicit stderr
  diagnostic, so the caller still dies without a marker and the gate still reads
  ``MISSING`` => ``needs-human``. Only the single-blip hair-trigger is removed; a
  sustained outage still (correctly) stops for a human.

Env overrides (all optional, for CI tuning; an unusable value is reported on
stderr and ignored rather than crashing the review):

  QA_HTTP_TIMEOUT         per-attempt request timeout, seconds (default 600)
  QA_HTTP_MAX_ATTEMPTS    total attempts including the first (default 5)
  QA_HTTP_BACKOFF         first backoff, seconds, doubled per retry (default 2)
  QA_HTTP_MAX_TOTAL_WAIT  total seconds of waiting allowed between attempts,
                          across the whole call (default 900)
"""

import datetime
import email.utils
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

# Ceiling on one backoff window — stops the doubling from growing without bound.
# Retry-After is deliberately NOT capped (see MAX_TOTAL_WAIT).
BACKOFF_CAP = 60.0

# Cap on the doubling exponent. 2**30 already dwarfs BACKOFF_CAP, so this changes
# no delay any configuration can actually produce — it only stops a large
# QA_HTTP_MAX_ATTEMPTS from computing 2**N as a bignum and overflowing the float
# multiply into an OverflowError instead of backing off.
BACKOFF_MAX_SHIFT = 30

# How much of a gateway's error body to reproduce on stderr. Long enough for the
# proxy's message and offending field, short enough not to bury the log.
ERROR_BODY_LIMIT = 2000


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

# Total waiting allowed between attempts, summed over the whole call. 900s
# accommodates any rate-limit window a gateway realistically asks for (a
# `Retry-After: 300` is honoured verbatim) while keeping the parked time well
# inside the 120-minute verify job. A single request for more than the remaining
# budget is not truncated — the client stops retrying and fails closed, which is
# both honest about Retry-After and a firmer bound than a per-wait cap.
MAX_TOTAL_WAIT = _env_number("QA_HTTP_MAX_TOTAL_WAIT", 900.0, float)


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
    # A response body that will not decode is transient too. A proxy blip can
    # answer 200 with an HTML error page, or truncate the body mid-JSON, without
    # urllib raising anything — the decode failure is then the only evidence that
    # this was not the model's answer. A genuinely malformed API response costs a
    # bounded few extra attempts and still fails closed with the same error;
    # treating the recoverable case as fatal costs a whole delivery.
    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        return True
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


def _http_date(raw):
    """Parse an HTTP-date into an aware UTC datetime, or None when unusable."""
    if not raw:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(str(raw).strip())
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        # A "-0000" zone means "no information about the local time zone"; RFC
        # 9110 reads such a timestamp as UTC, which is also what a gateway means.
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


def _date_form_seconds(text, headers):
    """Seconds until an HTTP-date ``Retry-After``, or None when unusable.

    Measured against the response's own ``Date`` header when it carries one:
    both timestamps then come from the gateway, so a skewed runner clock cannot
    turn a 30-second window into a 30-minute wait (or a negative one). Our own
    clock is the fallback, for the rare response with no ``Date``.
    """
    target = _http_date(text)
    if target is None:
        return None
    origin = _http_date(headers.get("Date")) or datetime.datetime.now(datetime.timezone.utc)
    try:
        return (target - origin).total_seconds()
    except (TypeError, OverflowError):
        return None


def retry_after_seconds(exc):
    """The response's ``Retry-After`` in seconds, or None when unusable.

    Both forms RFC 9110 allows are honoured — delay-seconds and HTTP-date — and
    the value is returned **in full**. It is deliberately not capped here:
    retrying before the window the gateway named just earns the same error
    again, so a delay we are unwilling to wait is handled by not retrying at all
    (``MAX_TOTAL_WAIT``, enforced by the caller), never by waiting less than we
    were asked to.
    """
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    text = str(raw).strip()
    try:
        value = float(text)
    except (TypeError, ValueError):
        value = _date_form_seconds(text, headers)
    if value is None:
        return None
    # A header is the one input here an upstream can choose freely. A non-finite
    # value must never reach time.sleep(); a non-positive one (a negative count,
    # or a deadline already past) falls through to the backoff schedule, because
    # waiting LONGER than asked still honours "not before T" whereas retrying
    # instantly hammers a gateway that has just pushed back.
    if not math.isfinite(value) or value <= 0:
        return None
    return value


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


def error_body(exc):
    """The gateway's response body, as a short single-line snippet ("" if none).

    A proxy 400/422 carries the actual reason in its *body* ("model does not
    support tools", "context length exceeded", which field was rejected); the
    status line says only `Bad Request`. Nothing here caught the error before,
    so that payload went straight to the bit bucket and an operator debugging a
    malformed request had nothing to go on.

    Reading consumes the response, so a caller that caught the error would then
    read an empty body. None of the three qa-review scripts catches it, and the
    text is on stderr either way — strictly more than they had.
    """
    reader = getattr(exc, "read", None)
    if reader is None:
        return ""
    try:
        raw = reader()
    except Exception:  # a diagnostic must never replace the error it describes
        return ""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    text = " ".join(str(raw).split())
    if len(text) > ERROR_BODY_LIMIT:
        text = text[:ERROR_BODY_LIMIT] + "... (truncated)"
    return text


def describe(exc):
    """Short, log-friendly rendering of a transport failure."""
    if isinstance(exc, urllib.error.HTTPError):
        return "HTTP %s %s" % (exc.code, exc.reason)
    return "%s: %s" % (type(exc).__name__, exc)


def describe_with_body(exc):
    """``describe`` plus the gateway's own explanation when there is one.

    Used on the two paths that end the call — a non-retryable error and
    exhaustion — where the body is the difference between a diagnosable failure
    and `HTTP 400 Bad Request`. Retry diagnostics use the short form: a
    retryable error is about to be retried, and its body is rarely the reason.
    """
    rendered = describe(exc)
    body = error_body(exc)
    if body:
        rendered += " — gateway said: " + body
    return rendered


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
    made = 0
    waited = 0.0
    for attempt in range(1, attempts + 1):
        made = attempt
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
                # Deterministic failure (a 4xx). Re-raise untouched so the caller
                # sees the same error it always did — but name the gateway's own
                # body first, since the traceback shows only the status line.
                sys.stderr.write(
                    "qa-review http: %s is not transient, not retrying\n" % describe_with_body(exc)
                )
                raise
            last = exc
            if attempt >= attempts:
                break
            delay = retry_delay(exc, attempt)
            if waited + delay > MAX_TOTAL_WAIT:
                # Honouring the delay would blow the wait budget, and retrying
                # sooner than the gateway asked would only earn the same error.
                # So stop here and fail closed — the honest outcome, and the one
                # that denies a hostile Retry-After the job it wanted to park.
                sys.stderr.write(
                    "qa-review http: %s asked for a %.1fs wait on attempt %d/%d, which does not "
                    "fit the %.1fs retry-wait budget (%.1fs already spent) — retrying earlier "
                    "than asked would only earn the same error, so the client stops here (raise "
                    "QA_HTTP_MAX_TOTAL_WAIT to wait it out)\n"
                    % (describe(exc), delay, attempt, attempts, MAX_TOTAL_WAIT, waited)
                )
                break
            sys.stderr.write(
                "qa-review http: %s on attempt %d/%d, retrying in %.1fs\n"
                % (describe(exc), attempt, attempts, delay)
            )
            sleep(delay)
            waited += delay

    sys.stderr.write(
        "qa-review http: giving up after %d attempts, last error %s — the script "
        "will exit without a verdict marker, so the gate reads MISSING and stops "
        "for a human\n" % (made, describe_with_body(last))
    )
    raise last
