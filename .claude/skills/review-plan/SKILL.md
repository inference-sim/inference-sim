---
name: review-plan
description: Send Claude plan to an LLM for external technical review (zero-config, works out-of-the-box)
allowed-tools:
  - Bash(~/.claude/skills/review-plan/scripts/review.sh *)
  - Bash(python3 *)
  - Bash(curl *)
  - Bash(jq *)
argument-hint: [plan_path] [model] [--dry-run]
---

# Review Plan with an LLM

Get independent technical feedback on your Claude Code plan from an LLM.

**Zero-config usage** — just type:
```
/review-plan
```

No setup required if you have an API key at the configured location.

## Usage Examples

**Zero-config (automatic detection):**
```
/review-plan
```

**Specify model:**
```
/review-plan gpt-4-turbo
```

**Explicit plan path:**
```
/review-plan ~/.claude/plans/my-plan.md
```

**Full control:**
```
/review-plan ~/.claude/plans/my-plan.md claude-3-opus
```

**Dry-run (verify without API call):**
```
/review-plan --dry-run
```

## Common Mistakes

**Positional argument ambiguity:**

The parsing rule is deterministic: if an argument contains `/` OR ends with `.md`, it's treated as a plan path. Otherwise, it's treated as a model name.

❌ `/review-plan my-plan` - interpreted as model name "my-plan"
✅ `/review-plan my-plan.md` - interpreted as plan file
✅ `/review-plan ./my-plan.md` - interpreted as plan file
✅ `/review-plan ~/.claude/plans/my-plan.md` - interpreted as plan file
✅ `/review-plan` - auto-detects plan (recommended for most use cases)

**Tip**: If you're specifying a plan, always include the `.md` extension or a path separator (`/` or `./`).

## Arguments

All arguments are optional:

- **Plan path** (optional): Path to plan file
  - If omitted: auto-detects using session pointer or single-plan fallback
  - Detected by: contains "/" OR ends with ".md"

- **Model** (optional): Model name
  - Default: `Azure/gpt-4o`
  - Examples: `gpt-4-turbo`, `gpt-3.5-turbo`, `claude-3-opus`

- **Flags**:
  - `--dry-run`: Show configuration without calling API
  - `--help`: Show usage help
  - `--no-redact`: Disable redaction (not recommended)

## What This Skill Does

1. **Dependency check**: Ensures python3, curl, jq are available
2. **Load API key**: Searches configured locations
3. **Locate plan**: Three-tier resolution (explicit > pointer > safe default)
4. **Redact secrets**: Removes API keys, private keys, tokens
5. **Call an LLM**: Sends redacted plan via curl
6. **Display review**: Structured feedback

## Session Safety

When multiple Claude Code sessions run in parallel, this skill safely identifies which plan to review:

1. **Explicit path** (highest priority): Use specified path
2. **Session pointer**: `~/.claude/plans/current-${CLAUDE_SESSION_ID}.path`
3. **Safe default**: If exactly one plan exists, use it
4. **Fail loudly**: If ambiguous, list options and show exact command

## Security & Privacy

- ✅ Redacts API keys, private keys, tokens before transmission
- ✅ Never prints secrets to stdout
- ✅ API keys loaded securely from file
- ⚠️  Plan content is sent to external API (OpenAI by default)

## Troubleshooting

**"ERROR: OpenAI API key not found"**
→ Ensure API key is at configured location (check error message for paths)

**"ERROR: Multiple plans found"**
→ Use explicit path: `/review-plan ~/.claude/plans/specific-plan.md`

**"ERROR: python3 not found"**
→ Install Python 3: `brew install python3` (macOS) or `apt install python3` (Linux)

For more help: Run `/review-plan --dry-run` to see configuration.

## Implementation

This skill executes:
```bash
bash ~/.claude/skills/review-plan/scripts/review.sh $ARGUMENTS
```
