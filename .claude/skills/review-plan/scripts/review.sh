#!/bin/bash

################################################################################
# review.sh - Zero-Config Plan Reviewer for Claude Code
#
# Design philosophy: WORKS OUT-OF-THE-BOX with sensible defaults.
# Fails loudly and helpfully when safety requires it.
#
# Usage:
#   review.sh [plan_path] [model] [--dry-run]
#
# All arguments are optional. Defaults work for common setups.
#
# Dependencies:
#   - python3 (for helper scripts)
#   - curl (for API calls)
#   - jq (for JSON parsing)
#
################################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ============================================================
# CONFIGURATION
# ============================================================

# API endpoint (NO /v1 suffix - we append it below)
API_BASE_URL="https://ete-litellm.ai-models.vpc-int.res.ibm.com"
API_PATH="/v1/chat/completions"

# Default model
DEFAULT_MODEL="Azure/gpt-4o"

# API key file search paths (in priority order)
API_KEY_SEARCH_PATHS=(
    "$HOME/.openai-api-key"
    "$HOME/.config/openai/api-key"
    "$HOME/.config/litellm/keys.env"
    "$HOME/.anthropic-api-key"
)

# Plans directory
PLANS_DIR="$HOME/.claude/plans"

# Redaction enabled by default
REDACTION_ENABLED=true

# Dry-run mode off by default
DRY_RUN=false

# Script directory (for helper scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# PARSE ARGUMENTS (SIMPLE, DETERMINISTIC)
# ============================================================

EXPLICIT_PLAN_PATH=""
MODEL="$DEFAULT_MODEL"
MODEL_OVERRIDE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-redact)
            REDACTION_ENABLED=false
            shift
            ;;
        --help|-h)
            cat <<'HELP'
review.sh - Plan Reviewer (Zero-Config)

Usage:
  review.sh [plan_path] [model] [flags]

Arguments (all optional):
  plan_path    Path to plan file (auto-detected if omitted)
               Detected by: contains "/" or ends with ".md"
  model        model name (default: Azure/gpt-4o)

Flags:
  --dry-run    Show config without calling API
  --no-redact  Disable secret redaction (not recommended)
  --help       Show this help

Examples:
  review.sh
  review.sh gpt-4-turbo
  review.sh ~/plans/my-plan.md
  review.sh ./plan.md gpt-4
  review.sh --dry-run

HELP
            exit 0
            ;;
        *)
            # Positional argument
            # Rule: Contains "/" or ends with ".md" → plan_path
            #       Otherwise → model
            if [ -z "$EXPLICIT_PLAN_PATH" ] && [[ "$1" == *"/"* || "$1" == *".md" ]]; then
                EXPLICIT_PLAN_PATH="$1"
            elif [ "$MODEL_OVERRIDE" = false ]; then
                MODEL="$1"
                MODEL_OVERRIDE=true
            fi
            shift
            ;;
    esac
done

# ============================================================
# DISPLAY HEADER
# ============================================================

echo "═══════════════════════════════════════════════════════════"
echo "  Plan Review"
echo "═══════════════════════════════════════════════════════════"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "  [DRY-RUN MODE - No API call will be made]"
    echo ""
fi

# ============================================================
# STEP 1: DEPENDENCY CHECK
# ============================================================

echo "[1/6] Checking dependencies..."

MISSING_DEPS=()

if ! command -v python3 &> /dev/null; then
    MISSING_DEPS+=("python3")
fi

if ! command -v curl &> /dev/null; then
    MISSING_DEPS+=("curl")
fi

if ! command -v jq &> /dev/null; then
    MISSING_DEPS+=("jq")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: Missing required dependencies: ${MISSING_DEPS[*]}"
    echo ""
    echo "Install them:"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  brew install python3 curl jq"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "  sudo apt-get install python3 curl jq"
    fi
    echo ""
    exit 1
fi

echo "   ✓ python3: $(python3 --version | cut -d' ' -f2)"
echo "   ✓ curl: $(curl --version | head -1 | cut -d' ' -f2)"
echo "   ✓ jq: $(jq --version | cut -d'-' -f2)"
echo ""

# ============================================================
# STEP 2: LOAD API KEY
# ============================================================

echo "[2/6] Loading API key..."

# Search for API key file
API_KEY_FILE=""
for path in "${API_KEY_SEARCH_PATHS[@]}"; do
    if [ -f "$path" ]; then
        API_KEY_FILE="$path"
        break
    fi
done

if [ -z "$API_KEY_FILE" ]; then
    echo ""
    echo "ERROR: OpenAI API key not found"
    echo ""
    echo "Checked locations:"
    for path in "${API_KEY_SEARCH_PATHS[@]}"; do
        echo "  ✗ $path"
    done
    echo ""
    echo "Quick setup:"
    echo "  echo \"sk-proj-YOUR-KEY\" > $HOME/.openai-api-key"
    echo "  chmod 600 $HOME/.openai-api-key"
    echo ""
    echo "Get your API key: https://platform.openai.com/api-keys"
    echo ""
    exit 1
fi

# Parse key file using helper script
# This script outputs: OPENAI_API_KEY=...
#                      OPENAI_BASE_URL=... (optional)
KEY_VARS=$(python3 "$SCRIPT_DIR/parse_key.py" "$API_KEY_FILE")

# Parse output (no eval, secure)
OPENAI_API_KEY=""
USER_BASE_URL=""

while IFS='=' read -r key value; do
    if [ "$key" = "OPENAI_API_KEY" ]; then
        OPENAI_API_KEY="$value"
    elif [ "$key" = "OPENAI_BASE_URL" ]; then
        USER_BASE_URL="$value"
    fi
done <<< "$KEY_VARS"

# Validate key was loaded
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "ERROR: Could not extract OPENAI_API_KEY from $API_KEY_FILE"
    echo ""
    echo "Supported formats:"
    echo "  Plain text: sk-proj-..."
    echo "  Dotenv:     OPENAI_API_KEY=sk-proj-..."
    echo '  JSON:       {"OPENAI_API_KEY": "sk-proj-..."}'
    echo ""
    exit 1
fi

# Use user-provided base URL if specified, otherwise use default
if [ -n "$USER_BASE_URL" ]; then
    API_BASE_URL="$USER_BASE_URL"
fi

# Construct full endpoint (base + path)
API_ENDPOINT="${API_BASE_URL}${API_PATH}"

echo "   ✓ API key loaded from: $API_KEY_FILE"
echo "   ✓ Key status: Valid format detected"
echo ""

# ============================================================
# STEP 3: LOCATE PLAN FILE (THREE-TIER RESOLUTION)
# ============================================================

echo "[3/6] Locating plan file..."

PLAN_FILE=""

# Tier 1: Explicit path argument
if [ -n "$EXPLICIT_PLAN_PATH" ]; then
    echo "   [Tier 1] Using explicit path"
    PLAN_FILE="$EXPLICIT_PLAN_PATH"

    if [ ! -f "$PLAN_FILE" ]; then
        echo ""
        echo "ERROR: Plan file not found at: $PLAN_FILE"
        exit 1
    fi
    echo "   ✓ Plan: $PLAN_FILE"

# Tier 2: Session pointer file
elif [ -n "${CLAUDE_SESSION_ID:-}" ]; then
    echo "   [Tier 2] Checking session pointer..."

    POINTER_FILE="$PLANS_DIR/current-${CLAUDE_SESSION_ID}.path"

    if [ -f "$POINTER_FILE" ]; then
        PLAN_FILE=$(head -1 "$POINTER_FILE" | tr -d '\n')

        if [ -f "$PLAN_FILE" ]; then
            echo "   ✓ Found pointer: $POINTER_FILE"
            echo "   ✓ Plan: $PLAN_FILE"
        else
            echo "   ✗ Pointer exists but plan not found: $PLAN_FILE"
            echo "   Falling back to Tier 3..."
            PLAN_FILE=""
        fi
    else
        echo "   ✗ No pointer file at: $POINTER_FILE"
        echo "   Falling back to Tier 3..."
    fi
fi

# Tier 3: Safe default detection
if [ -z "$PLAN_FILE" ]; then
    echo "   [Tier 3] Safe default detection..."

    PLAN_COUNT=$(find "$PLANS_DIR" -maxdepth 1 -name "*.md" -type f 2>/dev/null | wc -l | tr -d ' ')

    if [ "$PLAN_COUNT" -eq 0 ]; then
        echo ""
        echo "ERROR: No plan files found in $PLANS_DIR"
        echo ""
        echo "Create a plan first by asking Claude to enter plan mode."
        exit 1

    elif [ "$PLAN_COUNT" -eq 1 ]; then
        PLAN_FILE=$(find "$PLANS_DIR" -maxdepth 1 -name "*.md" -type f 2>/dev/null)
        echo "   ✓ Found exactly one plan: $(basename "$PLAN_FILE")"

        # Attempt to create pointer for future use
        if [ -n "${CLAUDE_SESSION_ID:-}" ]; then
            POINTER_FILE="$PLANS_DIR/current-${CLAUDE_SESSION_ID}.path"
            if echo "$PLAN_FILE" > "$POINTER_FILE" 2>/dev/null; then
                echo "   ✓ Created session pointer for next time"
            else
                echo "   ✗ Could not create pointer (permission denied)"
                echo "   To set manually:"
                echo "     echo \"$PLAN_FILE\" > $POINTER_FILE"
            fi
        fi

    else
        echo ""
        echo "ERROR: Multiple plans found, cannot auto-detect safely"
        echo ""
        echo "Plans in $PLANS_DIR:"
        find "$PLANS_DIR" -maxdepth 1 -name "*.md" -type f -exec ls -lh {} \; 2>/dev/null | \
            awk '{print "  • " $9 " (modified " $6 " " $7 " " $8 ")"}'
        echo ""

        if [ -n "${CLAUDE_SESSION_ID:-}" ]; then
            echo "Current session: $CLAUDE_SESSION_ID"
            echo ""
            echo "Choose one explicitly:"
            echo "  /review-plan $PLANS_DIR/<plan-name>.md"
            echo ""
            echo "Or set session pointer:"
            echo "  echo \"$PLANS_DIR/<plan-name>.md\" > $PLANS_DIR/current-${CLAUDE_SESSION_ID}.path"
        else
            echo "Specify plan explicitly:"
            echo "  /review-plan $PLANS_DIR/<plan-name>.md"
        fi
        echo ""
        exit 1
    fi
fi

echo ""

# ============================================================
# STEP 4: REDACT SENSITIVE CONTENT
# ============================================================

echo "[4/6] Redacting sensitive content..."

if [ "$REDACTION_ENABLED" = false ]; then
    echo "   ⚠️  Redaction DISABLED (--no-redact flag)"
    REDACTED_PLAN="$PLAN_FILE"
    REDACTION_COUNT=0
else
    # Create temporary files (using mktemp for security)
    REDACTED_PLAN=$(mktemp /tmp/claude-plan-redacted.XXXXXX)
    chmod 600 "$REDACTED_PLAN"

    # Use helper script for clean redaction
    python3 "$SCRIPT_DIR/redact.py" "$PLAN_FILE" "$REDACTED_PLAN"

    # Read metadata
    REDACTION_COUNT=$(grep "redaction_count=" "$REDACTED_PLAN.meta" | cut -d= -f2)

    if [ "$REDACTION_COUNT" -gt 0 ]; then
        echo "   ✓ Redacted $REDACTION_COUNT sensitive item(s)"
    else
        echo "   ✓ No sensitive content detected"
    fi

    # Ensure cleanup on exit
    trap "rm -f $REDACTED_PLAN $REDACTED_PLAN.meta" EXIT
fi

echo ""

# ============================================================
# DRY-RUN MODE: SHOW CONFIG AND EXIT
# ============================================================

if [ "$DRY_RUN" = true ]; then
    echo "═══════════════════════════════════════════════════════════"
    echo "  DRY-RUN SUMMARY"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "Configuration:"
    echo "  Plan file:    $PLAN_FILE"
    echo "  Model:        $MODEL"
    echo "  API endpoint: $API_ENDPOINT"
    echo "  Key file:     $API_KEY_FILE"
    echo "  Redaction:    $([ "$REDACTION_ENABLED" = true ] && echo "ENABLED ($REDACTION_COUNT items redacted)" || echo "DISABLED")"
    echo ""

    if [ "$REDACTION_ENABLED" = true ]; then
        echo "Redaction preview (first 500 chars):"
        echo "────────────────────────────────────────────────────────────"
        head -c 500 "$REDACTED_PLAN"
        echo ""
        echo "────────────────────────────────────────────────────────────"
        echo ""
    fi

    echo "✓ Dry-run complete (no API call made)"
    echo ""
    echo "To perform actual review:"
    echo "  /review-plan"
    exit 0
fi

# ============================================================
# STEP 5: BUILD JSON REQUEST AND CALL API
# ============================================================

echo "[5/6] Sending to $MODEL for review..."

# Build request using helper script
REQUEST_JSON=$(python3 "$SCRIPT_DIR/build_request.py" "$REDACTED_PLAN" "$MODEL")

# Make API call
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_ENDPOINT" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d "$REQUEST_JSON")

# Extract HTTP status code
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$RESPONSE" | sed '$d')

# Check for errors
if [ "$HTTP_CODE" -ne 200 ]; then
    echo ""
    echo "ERROR: API call failed with HTTP $HTTP_CODE"
    echo ""
    echo "Response:"
    echo "$RESPONSE_BODY" | jq . 2>/dev/null || echo "$RESPONSE_BODY"
    echo ""

    case "$HTTP_CODE" in
        401)
            echo "This usually means your API key is invalid or expired."
            echo "Get a new key: https://platform.openai.com/api-keys"
            ;;
        404)
            echo "This usually means the model name is invalid or endpoint URL is wrong."
            echo "Check model: $MODEL"
            echo "Check endpoint: $API_ENDPOINT"
            ;;
        429)
            echo "Rate limit exceeded. Wait a moment and try again."
            ;;
    esac
    exit 1
fi

echo "   ✓ Received response from API"
echo ""

# ============================================================
# STEP 6: PARSE AND DISPLAY RESPONSE
# ============================================================

echo "[6/6] Processing review..."
echo ""

# Extract review content
REVIEW_CONTENT=$(echo "$RESPONSE_BODY" | jq -r '.choices[0].message.content // .error.message // "ERROR: Could not parse response"')

if [ "$REVIEW_CONTENT" = "ERROR: Could not parse response" ]; then
    echo "ERROR: Unexpected API response format"
    echo ""
    echo "Raw response:"
    echo "$RESPONSE_BODY" | jq . 2>/dev/null || echo "$RESPONSE_BODY"
    exit 1
fi

# Display review
echo "═══════════════════════════════════════════════════════════"
echo "  REVIEW"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "$REVIEW_CONTENT"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Review Details"
echo "═══════════════════════════════════════════════════════════"
echo "  Plan reviewed: $PLAN_FILE"
echo "  Model used:    $MODEL"
echo "  API endpoint:  $API_ENDPOINT"
echo "  Redaction:     $([ "$REDACTION_ENABLED" = true ] && echo "ON ($REDACTION_COUNT items)" || echo "OFF")"
echo "  ⚠️  Plan content was sent to external API for review"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "✓ Review complete"
