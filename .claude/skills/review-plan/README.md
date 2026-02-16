# `/review-plan` - Zero-Config Plan Reviewer

A production-grade Claude Code skill that sends your implementation plans to an LLM for independent technical review.

## Quick Start

```bash
# Zero-config usage (works out-of-the-box)
/review-plan
```

That's it! If you have an API key for the IBM Research LiteLLM Gateway at `~/.openai-api-key`, it just works.

## Features

✅ **Zero-configuration** - Works with sensible defaults
✅ **Session-safe** - Handles parallel Claude sessions correctly
✅ **Smart plan detection** - Three-tier resolution system
✅ **Security-first** - Automatic redaction of secrets
✅ **Multiple API key formats** - Plain text, dotenv, JSON
✅ **LiteLLM compatible** - Works with proxy servers
✅ **Dry-run mode** - Verify configuration without API calls

## Installation Verification

The skill is installed at:
```
~/.claude/skills/review-plan/
├── SKILL.md              # Skill metadata
├── README.md             # This file
└── scripts/
    ├── review.sh         # Main orchestration script
    ├── parse_key.py      # API key parser (supports 3 formats)
    ├── redact.py         # Secret redaction engine
    └── build_request.py  # JSON request builder
```

All scripts should be executable (`chmod +x`).

## Setup: API Key

Create an API key file in one of these locations (checked in order):

1. `~/.openai-api-key` (recommended)
2. `~/.config/openai/api-key`
3. `~/.config/litellm/keys.env`
4. `~/.anthropic-api-key`

### Supported Formats

**Plain text** (simplest):
```
sk-proj-YOUR-KEY-HERE
```

**Dotenv format**:
```
OPENAI_API_KEY=sk-proj-YOUR-KEY-HERE
OPENAI_BASE_URL=https://api.openai.com
```

**JSON format**:
```json
{
  "OPENAI_API_KEY": "sk-proj-YOUR-KEY-HERE",
  "OPENAI_BASE_URL": "https://api.openai.com"
}
```

**Security**: Set proper permissions:
```bash
chmod 600 ~/.openai-api-key
```

## Usage Examples

### Zero-Config (Auto-Detect Everything)
```bash
/review-plan
```
**Requirements**:
- API key at default location
- Exactly one `.md` file in `~/.claude/plans/`

### Specify Model
```bash
/review-plan gpt-4-turbo
/review-plan gpt-3.5-turbo
/review-plan claude-3-opus  # Works with LiteLLM proxy
```

### Explicit Plan Path
```bash
/review-plan ~/.claude/plans/my-plan.md
/review-plan ./plan.md
/review-plan /absolute/path/to/plan.md
```

### Full Control
```bash
/review-plan ~/plans/feature-x.md gpt-4-turbo
```

### Dry-Run (Verify Configuration)
```bash
/review-plan --dry-run
```
Shows: resolved plan path, API endpoint, model, redaction count, preview

### Disable Redaction (Not Recommended)
```bash
/review-plan --no-redact
```

## Plan Resolution (Three-Tier System)

The skill safely identifies which plan to review using this priority order:

### Tier 1: Explicit Path (Highest Priority)
```bash
/review-plan /path/to/specific-plan.md
```
Uses exactly the path you specify.

### Tier 2: Session Pointer
Location: `~/.claude/plans/current-${CLAUDE_SESSION_ID}.path`

**Create pointer manually**:
```bash
echo "/Users/sri/.claude/plans/my-plan.md" > ~/.claude/plans/current-SESSION_ID.path
```

**Auto-created**: When Tier 3 succeeds, a pointer is automatically created (if permissions allow).

**Use case**: Running parallel Claude Code sessions, each working on different plans.

### Tier 3: Safe Default Detection
- If exactly **one** `.md` file exists in `~/.claude/plans/` → use it (zero-config success!)
- If **zero** plans → friendly error with instructions
- If **multiple** plans → fail loudly with list of options (safety check)

**Safety principle**: Never guess silently; always explicit or fail with guidance.

## Session Safety (Parallel Sessions)

When running multiple Claude Code sessions:

**Session A** (terminal 1):
```bash
export CLAUDE_SESSION_ID="session-aaa"
echo "$HOME/.claude/plans/plan-a.md" > ~/.claude/plans/current-session-aaa.path
/review-plan  # Reviews plan-a.md
```

**Session B** (terminal 2):
```bash
export CLAUDE_SESSION_ID="session-bbb"
echo "$HOME/.claude/plans/plan-b.md" > ~/.claude/plans/current-session-bbb.path
/review-plan  # Reviews plan-b.md
```

Each session maintains its own pointer file, preventing cross-session confusion.

## Security & Privacy

### What Gets Redacted
Before sending to the API, the following are automatically redacted:

1. **Private key blocks**: `-----BEGIN ... PRIVATE KEY-----` blocks
2. **API key lines**: Any line containing `API_KEY=...`
3. **Bearer tokens**: `Bearer xyz` or `Token: xyz`

### What DOESN'T Get Redacted
- Code snippets (unless they contain secrets)
- File paths
- Configuration examples (unless they show actual keys)

### Security Guarantees
- ✅ Secrets never printed to stdout (including key prefixes)
- ✅ API keys loaded securely (no `eval`)
- ✅ Redaction happens BEFORE transmission
- ✅ Temporary files cleaned up on exit

### Privacy Warning
⚠️ **Your plan content (after redaction) is sent to an external API** (OpenAI by default).

If using a custom endpoint (LiteLLM, local model), set `OPENAI_BASE_URL` in your key file.

## LiteLLM Proxy Setup

To route through LiteLLM proxy server:

**Key file** (`~/.openai-api-key`):
```
OPENAI_API_KEY=your-litellm-key
OPENAI_BASE_URL=http://localhost:8000
```

**Note**: BASE_URL should NOT include `/v1` suffix - that's added automatically.

Then use any model name:
```bash
/review-plan claude-3-opus
/review-plan gpt-4
/review-plan custom-model-name
```

## Troubleshooting

### ERROR: OpenAI API key not found
**Solution**: Create API key file at one of the checked locations.

```bash
echo "sk-proj-YOUR-KEY" > ~/.openai-api-key
chmod 600 ~/.openai-api-key
```

### ERROR: Multiple plans found
**Solution**: Use explicit path or set session pointer.

```bash
# Option 1: Explicit path
/review-plan ~/.claude/plans/specific-plan.md

# Option 2: Set pointer (if CLAUDE_SESSION_ID is set)
echo "$HOME/.claude/plans/specific-plan.md" > ~/.claude/plans/current-$CLAUDE_SESSION_ID.path
```

### ERROR: No plan files found
**Solution**: Create a plan first by asking Claude to enter plan mode.

### ERROR: python3 not found
**Solution**: Install Python 3.

```bash
# macOS
brew install python3

# Linux (Debian/Ubuntu)
sudo apt-get install python3

# Linux (Fedora/RHEL)
sudo dnf install python3
```

### ERROR: API call failed with HTTP 401
**Solution**: API key is invalid or expired. Get a new key from https://platform.openai.com/api-keys

### ERROR: API call failed with HTTP 404
**Solution**: Model name is invalid or endpoint URL is wrong.

Check your configuration:
```bash
/review-plan --dry-run
```

### Debugging: Dry-Run Mode
Always start with dry-run to verify configuration:

```bash
/review-plan --dry-run
```

This shows:
- Resolved plan path
- Model being used
- Full API endpoint (BASE_URL + /v1/chat/completions)
- API key file location (NOT the key itself)
- Redaction count
- Preview of redacted content (first 500 chars)

## Testing the Installation

### Test 1: Component Tests
```bash
# Test redaction
cd /tmp
cat > test-plan.md <<'EOF'
OPENAI_API_KEY=sk-fake
Bearer test123
EOF

python3 ~/.claude/skills/review-plan/scripts/redact.py test-plan.md out.txt
cat out.txt  # Should show [REDACTED: API KEY LINE] and Bearer [REDACTED]
cat out.txt.meta  # Should show redaction_count=2

# Test key parsing
echo 'OPENAI_API_KEY="sk-test"' > test-key.env
python3 ~/.claude/skills/review-plan/scripts/parse_key.py test-key.env
# Should output: OPENAI_API_KEY=sk-test (no quotes)
```

### Test 2: Dry-Run Test
```bash
# Create test plan
mkdir -p ~/.claude/plans
echo "# Test Plan" > ~/.claude/plans/test-plan.md

# Run dry-run
/review-plan --dry-run

# Expected: Shows configuration, no API call made
```

### Test 3: Real API Test (if you have a key)
```bash
/review-plan

# Expected: Sends plan to an LLM and displays review
```

## Implementation Details

### Architecture
- **Shell orchestration** (`review.sh`): Main logic, dependency checks, plan resolution
- **Python helpers**: Parse API keys, redact secrets, build JSON requests
- **No eval**: Secure key parsing without shell eval
- **Cleanup**: Automatic temp file removal (trap on exit)

### Endpoint Normalization
```bash
# Configuration (no /v1 in base URL)
API_BASE_URL="https://api.openai.com"
API_PATH="/v1/chat/completions"

# Usage
curl "${API_BASE_URL}${API_PATH}"  # Correct: https://api.openai.com/v1/chat/completions
```

This design supports custom endpoints cleanly:
```bash
# User sets in key file
OPENAI_BASE_URL=http://localhost:8000

# Result: http://localhost:8000/v1/chat/completions
```

### Argument Parsing Rules (Simple & Deterministic)
1. Flags (`--dry-run`, `--help`) detected by prefix
2. If arg contains `/` OR ends with `.md` → plan_path
3. Otherwise → model name
4. First match for each type wins

**No clever heuristics**: No file existence checks during parsing, no model prefix assumptions.

### Redaction Counting (Accurate)
Counts actual occurrences, not rules that matched:
```python
# Count matches BEFORE substitution
matches = re.findall(pattern, content)
redaction_count += len(matches)
content = re.sub(pattern, replacement, content)
```

If a plan has 3 Bearer tokens, it reports `redaction_count=3` (not `1`).

### Tool Permissions (Minimal)
From `SKILL.md` frontmatter:
```yaml
allowed-tools:
  - Bash(~/.claude/skills/review-plan/scripts/review.sh *)
  - Bash(~/.claude/skills/review-plan/scripts/parse_key.py *)
  - Bash(~/.claude/skills/review-plan/scripts/redact.py *)
  - Bash(~/.claude/skills/review-plan/scripts/build_request.py *)
  - Bash(python3 *)
  - Bash(curl *)
  - Bash(jq *)
```

Explicit paths instead of wildcards (least-privilege principle).

## Contributing

This skill follows the production implementation plan from the vLLM repository session.

**Key design principles**:
1. Works out-of-the-box (zero-config)
2. Fails loudly and helpfully (never guess silently)
3. Session-safe (parallel sessions supported)
4. Security-first (automatic redaction, no secret leakage)
5. Simple, deterministic logic (no clever heuristics)

## License

This skill is part of the vLLM project tooling.

## Support

For issues or questions:
1. Run `/review-plan --dry-run` to debug configuration
2. Check this README's troubleshooting section
3. File an issue in the vLLM repository

---

**Version**: 1.0.0 (Production)
**Last Updated**: 2026-01-30
**Claude Code Compatibility**: Requires Claude Code CLI with skills support
