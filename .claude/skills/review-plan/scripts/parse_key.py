#!/usr/bin/env python3
"""Parse API key from file (plain text, dotenv, or JSON)."""
import json
import sys

if len(sys.argv) != 2:
    print("Usage: parse_key.py <key_file>", file=sys.stderr)
    sys.exit(1)

key_file = sys.argv[1]

try:
    with open(key_file, "r") as f:
        content = f.read().strip()
except Exception as e:
    print(f"ERROR: Could not read key file: {e}", file=sys.stderr)
    sys.exit(1)

# Detect format and extract key
key = ""
base_url = ""

if content.startswith("{"):
    # JSON format
    try:
        data = json.loads(content)
        key = data.get("OPENAI_API_KEY", "")
        base_url = data.get("OPENAI_BASE_URL", "")
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in key file: {e}", file=sys.stderr)
        sys.exit(1)

elif "=" in content:
    # Dotenv format
    for line in content.split("\n"):
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            # Strip surrounding quotes (single or double)
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1]
            if k == "OPENAI_API_KEY":
                key = v
            elif k == "OPENAI_BASE_URL":
                base_url = v

else:
    # Plain text - just the key on first line
    key = content.split("\n")[0].strip()

# Validate key format (warning only, let API give definitive error)
if key and not key.startswith(("sk-", "sk_")):
    print("WARNING: API key doesn't match expected format (sk-* or sk_*)", file=sys.stderr)
    print("         The key will be used as-is, but may be rejected by the API", file=sys.stderr)

# Output in shell-parseable format (no eval needed)
if key:
    print(f"OPENAI_API_KEY={key}")
if base_url:
    print(f"OPENAI_BASE_URL={base_url}")

# If no key found, exit with error
if not key:
    print("ERROR: No OPENAI_API_KEY found in file", file=sys.stderr)
    sys.exit(1)
