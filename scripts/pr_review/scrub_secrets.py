#!/usr/bin/env python3
"""scrub_secrets.py — redact secret-shaped tokens from the review comment (stdin->stdout).

Defence in depth, NOT the primary control: the real control is that the LiteLLM
key never enters the reviewer's session (it lives in the sidecar). This is the
last line in case a reviewer is coaxed into echoing a credential-shaped string it
somehow obtained — it must never reach a public PR comment.

Deliberately conservative: it redacts things that look like credentials and
leaves ordinary prose and code intact. It is a filter, not a guarantee; a
determined encoding will slip past any regex, which is why it is the LAST line
and not the only one.
"""
import re
import sys

# Each pattern replaces the secret-looking run with [REDACTED], keeping any
# surrounding label so the comment still reads sensibly.
_PATTERNS = [
    # `Bearer <token>` / `Authorization: Bearer <token>`
    (re.compile(r'(?i)(bearer\s+)[A-Za-z0-9._\-]{12,}'), r'\1[REDACTED]'),
    # OpenAI/LiteLLM-style keys: sk-..., sk-proj-...
    (re.compile(r'\bsk-[A-Za-z0-9._\-]{8,}'), '[REDACTED]'),
    # x-api-key: <token>  /  api[_-]key = "<token>"
    (re.compile(r'(?i)((?:x-api-key|api[_-]?key|secret|token)\s*[:=]\s*["\']?)[A-Za-z0-9._\-]{12,}'),
     r'\1[REDACTED]'),
    # GitHub tokens
    (re.compile(r'\bgh[pousr]_[A-Za-z0-9]{20,}\b'), '[REDACTED]'),
]


def scrub(text: str) -> str:
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


def main() -> int:
    sys.stdout.write(scrub(sys.stdin.read()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
