#!/usr/bin/env python3
"""Redact sensitive content from plan file."""
import re
import sys

if len(sys.argv) != 3:
    print("Usage: redact.py <input_file> <output_file>", file=sys.stderr)
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
meta_file = output_file + ".meta"

try:
    with open(input_file, "r") as f:
        content = f.read()
except Exception as e:
    print(f"ERROR: Could not read input file: {e}", file=sys.stderr)
    sys.exit(1)

# Count redactions
redaction_count = 0

# Redact private key blocks
private_key_pattern = r'-----BEGIN [A-Z ]+ PRIVATE KEY-----.*?-----END [A-Z ]+ PRIVATE KEY-----'
matches = re.findall(private_key_pattern, content, flags=re.DOTALL)
redaction_count += len(matches)
content = re.sub(
    private_key_pattern,
    '[REDACTED: PRIVATE KEY BLOCK]',
    content,
    flags=re.DOTALL
)

# Redact API key lines (any line with API_KEY=)
api_key_pattern = r'^.*API_KEY.*=.*$'
matches = re.findall(api_key_pattern, content, flags=re.MULTILINE)
redaction_count += len(matches)
content = re.sub(
    api_key_pattern,
    '[REDACTED: API KEY LINE]',
    content,
    flags=re.MULTILINE
)

# Redact Bearer tokens
bearer_pattern = r'(Bearer|Token:)\s+[A-Za-z0-9_-]+'
matches = re.findall(bearer_pattern, content)
redaction_count += len(matches)
content = re.sub(
    bearer_pattern,
    r'\1 [REDACTED]',
    content
)

# Write redacted content
try:
    with open(output_file, "w") as f:
        f.write(content)
except Exception as e:
    print(f"ERROR: Could not write output file: {e}", file=sys.stderr)
    sys.exit(1)

# Write metadata
try:
    with open(meta_file, "w") as f:
        f.write(f"redaction_count={redaction_count}\n")
except Exception as e:
    print(f"ERROR: Could not write metadata file: {e}", file=sys.stderr)
    sys.exit(1)
