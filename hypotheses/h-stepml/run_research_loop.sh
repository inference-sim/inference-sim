#!/usr/bin/env bash
#
# StepML Research Loop Orchestrator
#
# Runs the WP1→WP2→WP3→WP4 research loop autonomously for up to MAX_ROUNDS rounds.
# Each round is broken into PHASES — each phase is a separate `claude -p` invocation
# with its own fresh context window. On-disk artifacts are the communication channel.
#
# Phase decomposition per round (each is one `claude -p` session):
#   Phase 1 (WP1+WP2): Read problem.md → /research-ideas → research.md → scaffold HYPOTHESIS.md files
#   Phase 2..N (WP3):  One session per idea — run sub-hypotheses → write FINDINGS_SUMMARY.md
#   Phase final (WP4): Read all FINDINGS_SUMMARY.md → leaderboard → convergence-review → update problem.md → STATUS
#
# This design ensures no single session exceeds the context window.
# Artifacts on disk (research.md, HYPOTHESIS.md, FINDINGS_SUMMARY.md) bridge sessions.
#
# Usage:
#   ./run_research_loop.sh [START_ROUND] [MAX_ROUNDS] [--dry-run]
#
# Examples:
#   ./run_research_loop.sh          # Start from round 3, run up to 5 rounds total
#   ./run_research_loop.sh 3 25     # Start from round 3, run up to 25 rounds total
#   ./run_research_loop.sh 3 5 --dry-run  # Simulate round 3 without invoking claude
#
# Prerequisites:
#   - `claude` CLI installed and authenticated
#   - Working directory: BLIS-research repo root
#   - WP0 infrastructure already complete
#   - problem.md up to date with prior round findings (if START_ROUND > 1)

set -euo pipefail

# --- Configuration ---
DRY_RUN=false
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    fi
done
# Strip --dry-run from positional args
positional=()
for arg in "$@"; do
    [[ "$arg" != "--dry-run" ]] && positional+=("$arg")
done
START_ROUND="${positional[0]:-3}"
MAX_ROUNDS="${positional[1]:-5}"
MAX_TURNS_PER_PHASE=200

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STEPML_DIR="${REPO_ROOT}/hypotheses/h-stepml"
LOG_FILE="${STEPML_DIR}/orchestrator.log"
WORKTREE_BASE="${REPO_ROOT}/../stepml-worktrees"

# --- Functions ---
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

die() {
    log "FATAL: $*"
    exit 1
}

# Run a claude -p session. Args: phase_name, log_file, prompt
run_claude() {
    local phase_name="$1"
    local phase_log="$2"
    local prompt="$3"

    if $DRY_RUN; then
        log "  [${phase_name}] DRY-RUN: Would invoke claude -p (skipped)"
        echo "[DRY-RUN] Prompt for ${phase_name}:" > "$phase_log"
        echo "$prompt" >> "$phase_log"
        return 0
    fi

    log "  [${phase_name}] Invoking claude -p ..."
    if claude -p "$prompt" \
        --dangerously-skip-permissions \
        --output-format stream-json \
        --verbose \
        --max-turns "$MAX_TURNS_PER_PHASE" \
        < /dev/null > "$phase_log" 2>&1; then
        log "  [${phase_name}] Session completed successfully"
        return 0
    else
        local exit_code=$?
        log "  [${phase_name}] WARNING: Session exited with code ${exit_code}"
        log "  [${phase_name}] Check log: ${phase_log}"
        return $exit_code
    fi
}

# Create mock artifacts for dry-run mode
dry_run_phase1() {
    local round_dir="$1"
    local round_num="$2"
    log "  [DRY-RUN] Creating mock Phase 1 artifacts..."
    for i in 1 2 3; do
        local idea_dir="${round_dir}/idea-${i}-mock-idea-${i}"
        mkdir -p "$idea_dir"
        cat > "${idea_dir}/HYPOTHESIS.md" <<HYPEOF
# Mock Idea ${i} (Round ${round_num} Dry Run)
## Sub-hypotheses
- H1: Mock sub-hypothesis 1
- H2: Mock sub-hypothesis 2
HYPEOF
    done
    touch "${round_dir}/.phase1_done"
    log "  [DRY-RUN] Created 3 mock idea directories + marker"
}

# Create mock FINDINGS_SUMMARY.md for dry-run mode
dry_run_wp3() {
    local idea_dir="$1"
    local idea_name="$2"
    log "  [DRY-RUN] Creating mock FINDINGS_SUMMARY.md for ${idea_name}..."
    cat > "${idea_dir}/FINDINGS_SUMMARY.md" <<SUMEOF
# FINDINGS_SUMMARY: ${idea_name} (Dry Run)
## Results
- Mock results: all sub-hypotheses simulated
## Best BLIS E2E: ~15% mean error (mock)
## Binding Constraints: N/A (dry run)
SUMEOF
}

# Create mock WP4 artifacts for dry-run mode
dry_run_wp4() {
    local round_dir="$1"
    local round_num="$2"
    log "  [DRY-RUN] Creating mock WP4 artifacts..."
    cat > "${round_dir}/FINDINGS_ROUND${round_num}.md" <<FREOF
# FINDINGS_ROUND${round_num} (Dry Run)
## Leaderboard
| Idea | Mean Error |
|------|-----------|
| idea-1-mock-idea-1 | 15% (mock) |
| idea-2-mock-idea-2 | 12% (mock) |
| idea-3-mock-idea-3 | 18% (mock) |
## Decision: ITERATE (mock)
FREOF
    echo "ITERATE" > "${round_dir}/STATUS"
    log "  [DRY-RUN] Wrote FINDINGS_ROUND${round_num}.md + STATUS=ITERATE"
}

# --- Worktree management ---

# Create an isolated git worktree for an idea's experimentation.
# Args: round_num, idea_name
# Sets: WORKTREE_DIR (global) to the created worktree path
create_idea_worktree() {
    local round_num="$1"
    local idea_name="$2"
    local round_dir="${STEPML_DIR}/round${round_num}"

    WORKTREE_DIR="${WORKTREE_BASE}/round${round_num}-${idea_name}"

    # Remove stale worktree if it exists (failed prior run)
    if [[ -d "$WORKTREE_DIR" ]]; then
        log "    Removing stale worktree: ${WORKTREE_DIR}"
        git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
    fi

    mkdir -p "$WORKTREE_BASE"

    # Create worktree from current HEAD
    local current_branch
    current_branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)
    git -C "$REPO_ROOT" worktree add "$WORKTREE_DIR" HEAD --detach 2>/dev/null \
        || die "Failed to create worktree at ${WORKTREE_DIR}"
    log "    Created worktree: ${WORKTREE_DIR} (detached at ${current_branch} HEAD)"

    # Copy unstaged files that the worktree needs:
    # 1. The entire round directory (HYPOTHESIS.md files, research.md, Phase 1 artifacts)
    local wt_round_dir="${WORKTREE_DIR}/hypotheses/h-stepml/round${round_num}"
    mkdir -p "$wt_round_dir"
    cp -r "${round_dir}/"* "$wt_round_dir/" 2>/dev/null || true
    # 2. Shared infrastructure (may have unstaged changes)
    cp -r "${STEPML_DIR}/shared/"* "${WORKTREE_DIR}/hypotheses/h-stepml/shared/" 2>/dev/null || true
    # 3. problem.md
    cp "${STEPML_DIR}/problem.md" "${WORKTREE_DIR}/hypotheses/h-stepml/problem.md"
    # 4. research.md
    [[ -f "${STEPML_DIR}/research.md" ]] && cp "${STEPML_DIR}/research.md" "${WORKTREE_DIR}/hypotheses/h-stepml/research.md"
    # 5. Any unstaged Go files (stepml.go etc.)
    cp "${REPO_ROOT}/sim/latency/stepml.go" "${WORKTREE_DIR}/sim/latency/stepml.go" 2>/dev/null || true
    cp "${REPO_ROOT}/sim/latency/stepml_test.go" "${WORKTREE_DIR}/sim/latency/stepml_test.go" 2>/dev/null || true

    log "    Copied round artifacts + shared infra into worktree"
}

# After an idea completes, copy artifacts back and save Go patches.
# Args: round_num, idea_name, idea_dir (in main repo)
collect_idea_artifacts() {
    local round_num="$1"
    local idea_name="$2"
    local idea_dir="$3"  # destination in main repo
    local wt_idea_dir="${WORKTREE_DIR}/hypotheses/h-stepml/round${round_num}/${idea_name}"

    # Copy all experiment artifacts back to main repo
    if [[ -d "$wt_idea_dir" ]]; then
        cp -r "${wt_idea_dir}/"* "${idea_dir}/" 2>/dev/null || true
        log "    Copied artifacts from worktree → ${idea_dir}"
    fi

    # Save any Go code changes as a patch for reproducibility
    local go_diff
    go_diff=$(git -C "$WORKTREE_DIR" diff -- "*.go" 2>/dev/null || true)
    if [[ -n "$go_diff" ]]; then
        echo "$go_diff" > "${idea_dir}/go_changes.patch"
        log "    Saved Go changes → ${idea_dir}/go_changes.patch"
    fi

    # Save list of all modified files
    local all_changes
    all_changes=$(git -C "$WORKTREE_DIR" diff --name-only 2>/dev/null || true)
    if [[ -n "$all_changes" ]]; then
        echo "$all_changes" > "${idea_dir}/modified_files.txt"
    fi
}

# Remove worktree after artifacts collected.
# Args: (uses global WORKTREE_DIR)
cleanup_idea_worktree() {
    if [[ -n "${WORKTREE_DIR:-}" ]] && [[ -d "$WORKTREE_DIR" ]]; then
        git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
        log "    Removed worktree: ${WORKTREE_DIR}"
    fi
    WORKTREE_DIR=""
}

# Write REPRODUCE.md in each idea's directory with exact reproduction steps.
# Args: round_num, idea_name, idea_dir (in main repo)
write_idea_reproduce() {
    local round_num="$1"
    local idea_name="$2"
    local idea_dir="$3"
    local reproduce_file="${idea_dir}/REPRODUCE.md"
    local git_sha
    git_sha=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
    local branch_name
    branch_name=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    local rel_idea="hypotheses/h-stepml/round${round_num}/${idea_name}"

    cat > "$reproduce_file" <<REPEOF
# Reproduction Guide: ${idea_name} (Round ${round_num})

> Auto-generated by the orchestrator after WP3 completed for this idea.

## Base State

- **Branch:** ${branch_name}
- **Commit:** ${git_sha}
- **Date:** $(date '+%Y-%m-%d %H:%M:%S')

## Step 1: Create isolated worktree

\`\`\`bash
git worktree add ../reproduce-${idea_name} ${git_sha} --detach
cd ../reproduce-${idea_name}
\`\`\`

## Step 2: Copy experiment artifacts

\`\`\`bash
# Copy this idea's artifacts (HYPOTHESIS.md, sub-hypothesis dirs, trained models)
cp -r ${rel_idea}/ ${rel_idea}/

# Copy shared infrastructure
cp -r hypotheses/h-stepml/shared/ hypotheses/h-stepml/shared/

# Copy problem.md (input context)
cp hypotheses/h-stepml/problem.md hypotheses/h-stepml/problem.md
\`\`\`

## Step 3: Apply Go changes (if any)

REPEOF

    if [[ -f "${idea_dir}/go_changes.patch" ]]; then
        cat >> "$reproduce_file" <<REPEOF
This idea modified Go code. Apply the patch:

\`\`\`bash
git apply ${rel_idea}/go_changes.patch
go build -o simulation_worker main.go
go test ./sim/latency/...
\`\`\`

### Patch contents

\`\`\`diff
$(cat "${idea_dir}/go_changes.patch")
\`\`\`
REPEOF
    else
        echo "No Go changes were made. The existing \`sim/latency/stepml.go\` was used as-is." >> "$reproduce_file"
        echo "" >> "$reproduce_file"
    fi

    # List modified files
    cat >> "$reproduce_file" <<REPEOF

## Step 4: Modified files

REPEOF

    if [[ -f "${idea_dir}/modified_files.txt" ]]; then
        {
            echo '```'
            cat "${idea_dir}/modified_files.txt"
            echo '```'
        } >> "$reproduce_file"
    else
        echo "No modified_files.txt recorded (idea may not have changed any tracked files)." >> "$reproduce_file"
    fi

    # List experiment scripts and trained models
    cat >> "$reproduce_file" <<REPEOF

## Step 5: Re-run experiments

Sub-hypothesis directories and their key files:

\`\`\`
REPEOF

    find "$idea_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | while read -r sub_dir; do
        local sub_name
        sub_name=$(basename "$sub_dir")
        echo "${sub_name}/" >> "$reproduce_file"
        # List Python scripts, shell scripts, and model files
        find "$sub_dir" -maxdepth 2 \( -name "*.py" -o -name "*.sh" -o -name "*.json" -o -name "*.pkl" -o -name "*.joblib" -o -name "FINDINGS.md" \) 2>/dev/null | sort | while read -r f; do
            echo "  $(basename "$f")" >> "$reproduce_file"
        done
    done

    echo '```' >> "$reproduce_file"

    cat >> "$reproduce_file" <<REPEOF

## Step 6: Validate via BLIS

\`\`\`bash
# Activate Python environment
source hypotheses/h-stepml/shared/../../../.inference-sim-env/bin/activate

# Run BLIS validation (adjust paths to trained model/coefficients)
cd ${rel_idea}
# Look for validate_blis.py invocations in the sub-hypothesis FINDINGS.md files
\`\`\`

## Key Results

See \`FINDINGS_SUMMARY.md\` in this directory for the full results table.
REPEOF

    log "    Wrote ${reproduce_file}"
}

# Discover idea directories for a given round
get_idea_dirs() {
    local round_dir="$1"
    # Find directories matching idea-* pattern, sorted
    find "$round_dir" -maxdepth 1 -type d -name 'idea-*' | sort
}

# Write BEST_RESULT.md summarizing the best idea across all completed rounds
# with full reproduction instructions.
write_best_result_summary() {
    local summary_file="${STEPML_DIR}/BEST_RESULT.md"

    log ""
    log "--- Writing best-result summary ---"

    cat > "$summary_file" <<'HEADER'
# StepML Research — Best Result Summary

> Auto-generated by the orchestrator at the end of a research run.
> This file identifies the best-performing idea and provides exact reproduction steps.

HEADER

    # Scan all rounds for FINDINGS_SUMMARY.md files and extract E2E error info
    local best_idea=""
    local best_round=""
    local best_error=""
    local best_idea_dir=""

    for rd in "${STEPML_DIR}"/round[0-9]*; do
        [[ -d "$rd" ]] || continue
        local rnum
        rnum=$(basename "$rd" | grep -o '[0-9]*$')

        for id in "$rd"/idea-*; do
            [[ -d "$id" ]] || continue
            local fs="${id}/FINDINGS_SUMMARY.md"
            [[ -f "$fs" ]] || continue

            local iname
            iname=$(basename "$id")

            # Try to extract a numeric E2E error percentage from the summary.
            # Look for patterns like "X% mean E2E" or "mean E2E error: X%" or "E2E: X%"
            local error_val
            error_val=$(grep -iEo '[0-9]+\.?[0-9]*%?\s*(mean\s+)?E2E|E2E[^0-9]*[0-9]+\.?[0-9]*%' "$fs" 2>/dev/null \
                | grep -oE '[0-9]+\.?[0-9]*' | head -1)

            if [[ -n "$error_val" ]]; then
                # Compare (bash doesn't do float comparison, use awk)
                if [[ -z "$best_error" ]] || awk "BEGIN{exit !($error_val < $best_error)}" 2>/dev/null; then
                    best_error="$error_val"
                    best_idea="$iname"
                    best_round="$rnum"
                    best_idea_dir="$id"
                fi
            fi
        done
    done

    # --- Leaderboard: all ideas across all rounds ---
    {
        echo "## Leaderboard (All Rounds)"
        echo ""
        echo "| Round | Idea | E2E Error (extracted) | Has Go Patch | FINDINGS_SUMMARY |"
        echo "|-------|------|-----------------------|--------------|------------------|"

        for rd in "${STEPML_DIR}"/round[0-9]*; do
            [[ -d "$rd" ]] || continue
            local rnum
            rnum=$(basename "$rd" | grep -o '[0-9]*$')

            for id in "$rd"/idea-*; do
                [[ -d "$id" ]] || continue
                local iname fs error_val has_patch
                iname=$(basename "$id")
                fs="${id}/FINDINGS_SUMMARY.md"
                [[ -f "$fs" ]] || continue

                error_val=$(grep -iEo '[0-9]+\.?[0-9]*%?\s*(mean\s+)?E2E|E2E[^0-9]*[0-9]+\.?[0-9]*%' "$fs" 2>/dev/null \
                    | grep -oE '[0-9]+\.?[0-9]*' | head -1)
                [[ -z "$error_val" ]] && error_val="N/A"

                has_patch="No"
                [[ -f "${id}/go_changes.patch" ]] && has_patch="Yes"

                local marker=""
                [[ "$iname" == "$best_idea" && "$rnum" == "$best_round" ]] && marker=" **BEST**"

                echo "| ${rnum} | ${iname}${marker} | ${error_val}% | ${has_patch} | round${rnum}/${iname}/FINDINGS_SUMMARY.md |"
            done
        done
        echo ""
    } >> "$summary_file"

    # --- Best result details ---
    if [[ -n "$best_idea" ]]; then
        {
            echo "## Best Result"
            echo ""
            echo "- **Round:** ${best_round}"
            echo "- **Idea:** ${best_idea}"
            echo "- **E2E Error:** ${best_error}%"
            echo "- **Artifacts:** \`hypotheses/h-stepml/round${best_round}/${best_idea}/\`"
            echo ""

            # Include the full FINDINGS_SUMMARY for the best idea
            echo "### FINDINGS_SUMMARY (verbatim)"
            echo ""
            echo '```'
            cat "${best_idea_dir}/FINDINGS_SUMMARY.md"
            echo '```'
            echo ""

            # --- Reproduction instructions ---
            echo "## Reproduction Steps"
            echo ""
            echo "### 1. Locate artifacts"
            echo '```bash'
            echo "ls hypotheses/h-stepml/round${best_round}/${best_idea}/"
            echo '```'
            echo ""
            echo "### 2. Apply Go changes (if any)"
            if [[ -f "${best_idea_dir}/go_changes.patch" ]]; then
                echo '```bash'
                echo "# Review the patch first"
                echo "cat hypotheses/h-stepml/round${best_round}/${best_idea}/go_changes.patch"
                echo ""
                echo "# Apply to a clean worktree"
                echo "git worktree add ../stepml-reproduce HEAD --detach"
                echo "cd ../stepml-reproduce"
                echo "git apply hypotheses/h-stepml/round${best_round}/${best_idea}/go_changes.patch"
                echo "go build -o simulation_worker main.go"
                echo '```'
                echo ""
                echo "### Go changes summary"
                echo '```diff'
                head -50 "${best_idea_dir}/go_changes.patch"
                echo '```'
            else
                echo "No Go changes were made for this idea — uses existing \`stepml.go\`."
            fi
            echo ""

            echo "### 3. Re-run experiments"
            echo '```bash'
            echo "# Re-run the Python experiments from the idea directory"
            echo "cd hypotheses/h-stepml/round${best_round}/${best_idea}/"
            echo "# Check each sub-hypothesis directory for run scripts or Python files"
            echo "ls -d h*/"
            echo '```'
            echo ""

            echo "### 4. Modified files"
            if [[ -f "${best_idea_dir}/modified_files.txt" ]]; then
                echo '```'
                cat "${best_idea_dir}/modified_files.txt"
                echo '```'
            else
                echo "No modified_files.txt recorded."
            fi
            echo ""

            echo "### 5. Trained model coefficients"
            echo "Check for coefficient/model files in the idea's sub-hypothesis directories:"
            echo '```bash'
            find "$best_idea_dir" -name "*.json" -o -name "*.pkl" -o -name "*.joblib" -o -name "*coefficients*" -o -name "*model*" 2>/dev/null | while read -r f; do
                echo "  $(echo "$f" | sed "s|${STEPML_DIR}/||")"
            done
            echo '```'
        } >> "$summary_file"
    else
        echo "No ideas with extractable E2E error found across rounds." >> "$summary_file"
    fi

    # --- Round history ---
    {
        echo ""
        echo "## Round History"
        echo ""
        echo "| Round | STATUS | FINDINGS_ROUND |"
        echo "|-------|--------|----------------|"
        for rd in "${STEPML_DIR}"/round[0-9]*; do
            [[ -d "$rd" ]] || continue
            local rnum st
            rnum=$(basename "$rd" | grep -o '[0-9]*$')
            st="(no STATUS)"
            [[ -f "${rd}/STATUS" ]] && st=$(cat "${rd}/STATUS" | tr -d '[:space:]')
            local fr="(not written)"
            [[ -f "${rd}/FINDINGS_ROUND${rnum}.md" ]] && fr="round${rnum}/FINDINGS_ROUND${rnum}.md"
            echo "| ${rnum} | ${st} | ${fr} |"
        done
    } >> "$summary_file"

    log "Wrote: ${summary_file}"
}

# --- Auth ---
# Source LiteLLM proxy credentials if available
LITELLM_RC="${HOME}/litellm.sh"
if [[ -f "$LITELLM_RC" ]]; then
    # shellcheck disable=SC1090
    source "$LITELLM_RC"
    log "Sourced auth from ${LITELLM_RC}"
fi

# --- Validation ---
cd "$REPO_ROOT" || die "Cannot cd to repo root: $REPO_ROOT"
if ! $DRY_RUN; then
    command -v claude >/dev/null 2>&1 || die "'claude' CLI not found in PATH"
fi
[[ -f "${STEPML_DIR}/problem.md" ]] || die "problem.md not found at ${STEPML_DIR}/problem.md"
[[ -f "CLAUDE.md" ]] || die "CLAUDE.md not found in repo root"

# --- Main Loop ---
log "=========================================="
log "StepML Research Loop Orchestrator (phased)"
log "Start round: ${START_ROUND}"
log "Max rounds:  ${MAX_ROUNDS}"
log "Max turns per phase: ${MAX_TURNS_PER_PHASE}"
log "Dry run:     ${DRY_RUN}"
log "Repo root: ${REPO_ROOT}"
log "=========================================="

round=$START_ROUND

while (( round <= MAX_ROUNDS )); do
    log ""
    log "=========== Round ${round} ==========="
    round_dir="${STEPML_DIR}/round${round}"
    status_file="${round_dir}/STATUS"
    mkdir -p "$round_dir"

    # Check if this round already completed (resume after orchestrator restart)
    if [[ -f "$status_file" ]]; then
        existing_status=$(cat "$status_file" | tr -d '[:space:]')
        log "Round ${round} already has STATUS=${existing_status}, skipping"
        case "$existing_status" in
            CONVERGED)
                log "Previously converged. Exiting."
                exit 0 ;;
            ABORT)
                log "Previously aborted. Exiting."
                exit 0 ;;
            ITERATE)
                log "Previously decided to iterate. Moving to round $((round + 1))."
                round=$((round + 1))
                continue ;;
        esac
    fi

    # ============================================================
    # PHASE 1: WP1 (Ideation) + WP2 (Scaffolding)
    # ============================================================
    # These are lightweight enough to share a session.
    # Input:  problem.md
    # Output: research.md + round<N>/idea-*/HYPOTHESIS.md
    # ============================================================

    phase1_marker="${round_dir}/.phase1_done"
    if [[ -f "$phase1_marker" ]]; then
        log "  Phase 1 (WP1+WP2) already complete, skipping"
    elif $DRY_RUN; then
        log "  Phase 1: WP1 (Ideation) + WP2 (Scaffolding) [DRY-RUN]"
        run_claude "round${round}-phase1" "${round_dir}/phase1_wp1_wp2.log" \
"[DRY-RUN] Would execute WP1+WP2 for round ${round}" \
        || true
        dry_run_phase1 "$round_dir" "$round"
    else
        log "  Phase 1: WP1 (Ideation) + WP2 (Scaffolding)"
        run_claude "round${round}-phase1" "${round_dir}/phase1_wp1_wp2.log" \
"Execute WP1 and WP2 for Round ${round} of the StepML research loop.

Follow the macro plan at docs/plans/2026-02-26-stepml-research-macro-plan.md strictly.

STEP 1 — WP1 (Ideation):
- Read hypotheses/h-stepml/problem.md — it is the sole input with all prior round learnings.
- Run /research-ideas to generate 3+ ideas. Ideas must address binding constraints from prior rounds and must not repeat eliminated approaches listed in problem.md.
- The output research.md should be saved in the h-stepml directory.

STEP 2 — WP2 (Scaffolding):
- Extract the top 3 ideas from research.md.
- Create directories under hypotheses/h-stepml/round${round}/idea-<N>-<name>/ for each idea.
- Write HYPOTHESIS.md in each idea directory with 2-3 sub-hypotheses.
- Each HYPOTHESIS.md must reference prior round findings if round > 1.

When done, create the marker file hypotheses/h-stepml/round${round}/.phase1_done to signal completion.

Do NOT proceed to WP3 (experimentation). Do NOT ask for confirmation. Just do WP1 and WP2." \
        || log "  Phase 1 had non-zero exit, checking marker..."

        if [[ ! -f "$phase1_marker" ]]; then
            # Try to check if the artifacts exist even without the marker
            idea_count=$(find "$round_dir" -maxdepth 1 -type d -name 'idea-*' 2>/dev/null | wc -l | tr -d ' ')
            if (( idea_count > 0 )); then
                log "  Phase 1 artifacts found (${idea_count} ideas) but marker missing — creating marker"
                touch "$phase1_marker"
            else
                die "Phase 1 failed for round ${round} — no idea directories created. Check ${round_dir}/phase1_wp1_wp2.log"
            fi
        fi

        # Archive research.md into the round directory.
        # /research-ideas writes it as a sibling of problem.md (h-stepml/research.md),
        # so each round overwrites the same file. Copy it to preserve per-round ideas.
        if [[ -f "${STEPML_DIR}/research.md" ]]; then
            cp "${STEPML_DIR}/research.md" "${round_dir}/research.md"
            log "  Archived research.md → ${round_dir}/research.md"
        fi
    fi

    # ============================================================
    # PHASE 2..N: WP3 (Experimentation) — one session per idea
    # ============================================================
    # Each idea gets its own fresh context window.
    # Input:  idea's HYPOTHESIS.md + shared infrastructure
    # Output: idea's FINDINGS_SUMMARY.md
    # ============================================================

    idea_dirs=$(get_idea_dirs "$round_dir")
    if [[ -z "$idea_dirs" ]]; then
        die "No idea directories found in ${round_dir}"
    fi

    idea_num=0
    while IFS= read -r idea_dir; do
        idea_num=$((idea_num + 1))
        idea_name=$(basename "$idea_dir")
        summary_file="${idea_dir}/FINDINGS_SUMMARY.md"

        if [[ -f "$summary_file" ]]; then
            log "  Phase 2.${idea_num}: ${idea_name} — FINDINGS_SUMMARY.md exists, skipping"
            continue
        fi

        if $DRY_RUN; then
            log "  Phase 2.${idea_num}: WP3 for ${idea_name} [DRY-RUN]"
            log "    [DRY-RUN] Would create worktree at ${WORKTREE_BASE}/round${round}-${idea_name}"
            dry_run_wp3 "$idea_dir" "$idea_name"
            continue
        fi

        log "  Phase 2.${idea_num}: WP3 for ${idea_name}"

        # Create isolated worktree for this idea
        create_idea_worktree "$round" "$idea_name"
        local phase_log="${round_dir}/phase2_${idea_name}.log"

        # Run claude -p inside the worktree (cd in subshell)
        log "    Running in worktree: ${WORKTREE_DIR}"
        (
            cd "$WORKTREE_DIR"
            if $DRY_RUN; then
                echo "[DRY-RUN] Would run WP3 for ${idea_name} in worktree" > "$phase_log"
            else
                log "  [round${round}-${idea_name}] Invoking claude -p ..."
                claude -p "Execute WP3 (experimentation) for a SINGLE idea in Round ${round} of the StepML research loop.

Follow the macro plan at docs/plans/2026-02-26-stepml-research-macro-plan.md strictly.

YOUR TASK: Run all sub-hypotheses for the idea at: hypotheses/h-stepml/round${round}/${idea_name}/

STEPS:
1. Read the HYPOTHESIS.md in that directory to understand the idea and its sub-hypotheses.
2. For each sub-hypothesis, implement the experiment (write code if needed), run it, analyze results, and write FINDINGS.md in the sub-hypothesis directory.
3. Use the shared infrastructure at hypotheses/h-stepml/shared/ for data loading, evaluation, and BLIS validation.
4. After ALL sub-hypotheses are complete, write FINDINGS_SUMMARY.md in the idea's root directory (hypotheses/h-stepml/round${round}/${idea_name}/FINDINGS_SUMMARY.md).

The FINDINGS_SUMMARY.md must contain:
1. Idea recap
2. Sub-hypothesis results table (status, key metric, takeaway)
3. Best BLIS E2E result — full per-experiment error table
4. What worked (specific techniques)
5. What failed and why (root causes)
6. Binding constraints
7. Data insights discovered
8. Comparison to baseline
9. Go integration feasibility

You are running in an ISOLATED GIT WORKTREE. You may freely modify Go code (e.g. sim/latency/stepml.go) — changes are isolated to this worktree and will not affect other ideas. Any Go changes will be saved as a patch file for reproducibility.

Do NOT proceed to WP4 or any other idea. Do NOT ask for confirmation. Just run this one idea's experiments and write the summary." \
                    --dangerously-skip-permissions \
                    --output-format stream-json \
                    --verbose \
                    --max-turns "$MAX_TURNS_PER_PHASE" \
                    < /dev/null > "$phase_log" 2>&1 \
                || log "  [round${round}-${idea_name}] Session exited with non-zero code"
            fi
        )

        # Collect artifacts back to main repo
        collect_idea_artifacts "$round" "$idea_name" "$idea_dir"

        # Write per-idea reproduction guide
        write_idea_reproduce "$round" "$idea_name" "$idea_dir"

        # Check for summary in main repo (was copied back by collect_idea_artifacts)
        if [[ ! -f "$summary_file" ]]; then
            log "  WARNING: FINDINGS_SUMMARY.md not created for ${idea_name}. Check ${phase_log}"
            log "  Continuing to next idea (WP4 will handle incomplete ideas)..."
        fi

        # Clean up worktree
        cleanup_idea_worktree
    done <<< "$idea_dirs"

    # ============================================================
    # PHASE FINAL: WP4 (Leaderboard + Convergence Review + Wrap-up)
    # ============================================================
    # Input:  All FINDINGS_SUMMARY.md files from this round
    # Output: FINDINGS_ROUND<N>.md, updated problem.md, STATUS file
    # ============================================================

    log "  Phase final: WP4 (Leaderboard + Convergence Review)"

    # Build list of available summaries for the prompt
    summaries=""
    while IFS= read -r idea_dir; do
        idea_name=$(basename "$idea_dir")
        sf="${idea_dir}/FINDINGS_SUMMARY.md"
        if [[ -f "$sf" ]]; then
            summaries="${summaries}\n  - hypotheses/h-stepml/round${round}/${idea_name}/FINDINGS_SUMMARY.md"
        fi
    done <<< "$idea_dirs"

    if $DRY_RUN; then
        run_claude "round${round}-wp4" "${round_dir}/phase_final_wp4.log" \
"[DRY-RUN] Would execute WP4 for round ${round}" \
        || true
        dry_run_wp4 "$round_dir" "$round"
    else
    run_claude "round${round}-wp4" "${round_dir}/phase_final_wp4.log" \
"Execute WP4 (Leaderboard, Convergence Review, and Round Wrap-up) for Round ${round} of the StepML research loop.

Follow the macro plan at docs/plans/2026-02-26-stepml-research-macro-plan.md strictly — especially the WP4 procedure (Steps 1-6).

AVAILABLE FINDINGS SUMMARIES (read all of these):
$(echo -e "$summaries")

Also read: hypotheses/h-stepml/problem.md (current problem statement with prior round context)

YOUR TASKS (in order):
1. LEADERBOARD: Read all FINDINGS_SUMMARY.md files. Rank ideas by BLIS E2E mean error. Compare against blackbox baseline and prior rounds.
2. FINDINGS_ROUND${round}.md: Write comprehensive round findings to hypotheses/h-stepml/round${round}/FINDINGS_ROUND${round}.md with all 12 sections per the macro plan.
3. CONVERGENCE REVIEW: Run /convergence-review on FINDINGS_ROUND${round}.md.
4. LOOP DECISION: Decide CONVERGED, ITERATE, or ABORT based on results and convergence review.
5. UPDATE problem.md (if ITERATE): Update hypotheses/h-stepml/problem.md with ALL accumulated knowledge per the macro plan's Step 5 specification (baseline results, solved/unsolved experiments, successful techniques, data characteristics, eliminated approaches, binding constraints, key questions, prescribed focus areas, cumulative round history).
6. WRITE STATUS FILE (MANDATORY LAST ACTION): Write hypotheses/h-stepml/round${round}/STATUS containing exactly one word: CONVERGED, ITERATE, or ABORT.

Do NOT ask for confirmation. Execute all steps and ensure the STATUS file is written as the very last action." \
    || log "  WP4 phase had non-zero exit, checking STATUS..."
    fi

    # --- Read STATUS ---
    if [[ ! -f "$status_file" ]]; then
        log "ERROR: No STATUS file after WP4 for round ${round}"
        log "Check: ${round_dir}/phase_final_wp4.log"
        log "To resume WP4 only: re-run this script (it will skip completed phases)"
        die "Missing STATUS file for round ${round}"
    fi

    status=$(cat "$status_file" | tr -d '[:space:]')
    log "Round ${round} STATUS: ${status}"

    case "$status" in
        CONVERGED)
            log ""
            log "=========================================="
            log "CONVERGED at round ${round}!"
            log "Target <10% E2E achieved."
            log "See: ${round_dir}/FINDINGS_ROUND${round}.md"
            log "=========================================="
            write_best_result_summary
            exit 0
            ;;
        ITERATE)
            log "Iterating to round $((round + 1))..."
            ;;
        ABORT)
            log ""
            log "=========================================="
            log "ABORTED at round ${round}."
            log "No improvement path identified."
            log "See: ${round_dir}/FINDINGS_ROUND${round}.md"
            log "=========================================="
            write_best_result_summary
            exit 0
            ;;
        *)
            die "Invalid STATUS: '${status}' (expected CONVERGED, ITERATE, or ABORT)"
            ;;
    esac

    round=$((round + 1))
done

log ""
log "=========================================="
log "Max rounds (${MAX_ROUNDS}) reached without convergence."
log "Best results in most recent FINDINGS_ROUND file."
log "=========================================="
write_best_result_summary
exit 0
