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

# Validate that an idea's FINDINGS_SUMMARY.md exists and contains generalization results.
# Returns: 0=valid, 1=FINDINGS_SUMMARY.md missing, 2=generalization missing
validate_idea_artifacts() {
    local idea_dir="$1"
    local idea_name="$2"
    local summary="${idea_dir}/FINDINGS_SUMMARY.md"

    # Check 1: FINDINGS_SUMMARY.md exists
    if [[ ! -f "$summary" ]]; then
        log "    VALIDATION FAIL: ${idea_name} — FINDINGS_SUMMARY.md missing"
        return 1
    fi

    # Infrastructure-only ideas can use GENERALIZATION_NOTE.md as escape hatch
    if [[ -f "${idea_dir}/GENERALIZATION_NOTE.md" ]]; then
        log "    VALIDATION OK: ${idea_name} — uses GENERALIZATION_NOTE.md (infrastructure-only)"
        return 0
    fi

    # Check 2: Contains LOMO results
    local has_lomo has_lowo
    has_lomo=$(grep -ciE 'LOMO|leave.one.model.out' "$summary" 2>/dev/null || echo 0)
    has_lowo=$(grep -ciE 'LOWO|leave.one.workload.out' "$summary" 2>/dev/null || echo 0)

    if (( has_lomo == 0 )) || (( has_lowo == 0 )); then
        log "    VALIDATION FAIL: ${idea_name} — FINDINGS_SUMMARY.md missing generalization (LOMO mentions: ${has_lomo}, LOWO mentions: ${has_lowo})"
        return 2
    fi

    log "    VALIDATION OK: ${idea_name} — FINDINGS_SUMMARY.md + LOMO + LOWO present"
    return 0
}

# Retry a failed idea with a targeted prompt. Runs in the existing worktree.
# Args: round_num, idea_name, idea_dir, validation_code (1=no summary, 2=no generalization)
retry_idea_in_worktree() {
    local round_num="$1"
    local idea_name="$2"
    local idea_dir="$3"
    local val_code="$4"
    local retry_log="${idea_dir}/../phase2_${idea_name}_retry.log"
    local retry_prompt

    if (( val_code == 1 )); then
        log "    RETRY: ${idea_name} — writing missing FINDINGS_SUMMARY.md"
        retry_prompt="The WP3 session for ${idea_name} completed but did NOT produce a FINDINGS_SUMMARY.md. This is a CRITICAL artifact.

YOUR TASK: Examine the experiment outputs and write FINDINGS_SUMMARY.md.

1. Read hypotheses/h-stepml/round${round_num}/${idea_name}/HYPOTHESIS.md for context.
2. Check each sub-hypothesis directory (h1-*, h2-*, etc.) for:
   - output/ directories with results (CSV, JSON)
   - FINDINGS.md files with analysis
   - Python scripts that can be re-run if outputs are missing
3. If sub-hypothesis experiments have not been run at all, run them now using the shared infrastructure at hypotheses/h-stepml/shared/.
4. After examining/running all sub-hypotheses, write FINDINGS_SUMMARY.md at:
   hypotheses/h-stepml/round${round_num}/${idea_name}/FINDINGS_SUMMARY.md

The FINDINGS_SUMMARY.md MUST contain these sections:
1. Idea recap  2. Sub-hypothesis results table  3. Best BLIS E2E result table
4. What worked  5. What failed and why  6. Binding constraints
7. Data insights  8. Comparison to baseline  9. Go integration feasibility
10. Generalization Results (MANDATORY):
    - LOMO: Leave-one-model-out 4-fold CV results (target <80% MAPE per fold)
    - LOWO: Leave-one-workload-out 3-fold CV results (target <50% MAPE per fold)
    - Use hypotheses/h-stepml/shared/splits.py for data splitting
    - If infrastructure-only, write GENERALIZATION_NOTE.md instead

Do NOT ask for confirmation. Write the summary."
    else
        log "    RETRY: ${idea_name} — executing missing LOMO/LOWO generalization"
        retry_prompt="The WP3 session for ${idea_name} produced a FINDINGS_SUMMARY.md but it is MISSING mandatory LOMO and/or LOWO generalization results. The orchestrator requires these.

YOUR TASK: Execute the generalization experiments and update FINDINGS_SUMMARY.md.

1. Read hypotheses/h-stepml/round${round_num}/${idea_name}/HYPOTHESIS.md — find the LOMO and LOWO sub-hypotheses.
2. If LOMO/LOWO sub-hypothesis directories exist, check for run scripts and execute them.
3. If they don't exist, create and run them:
   - LOMO: Train on 3 models, test on held-out 4th (4-fold: llama-2-7b, llama-2-70b, codellama-34b, mixtral-8x7b). Target <80% MAPE per fold.
   - LOWO: Train on 2 workloads, test on held-out 3rd (3-fold: general, codegen, roleplay). Target <50% MAPE per fold.
4. Use hypotheses/h-stepml/shared/splits.py for LOMO/LOWO data splitting.
5. UPDATE the existing FINDINGS_SUMMARY.md — add a 'Generalization Results' section (section 10) with LOMO and LOWO result tables.
   - If this idea is infrastructure-only (not a trainable model), write GENERALIZATION_NOTE.md with cross-experiment evidence instead.

Do NOT ask for confirmation. Execute and update the summary."
    fi

    (
        cd "$WORKTREE_DIR"
        log "  [round${round_num}-${idea_name}-retry] Invoking claude -p ..."
        claude -p "$retry_prompt" \
            --dangerously-skip-permissions \
            --output-format stream-json \
            --verbose \
            --max-turns 100 \
            < /dev/null > "$retry_log" 2>&1 \
        || log "  [round${round_num}-${idea_name}-retry] Session exited with non-zero code"
    )
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

# --- Environment ---
# Prevent "nested session" errors when this script is launched from inside Claude Code
unset CLAUDECODE 2>/dev/null || true

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
- Also read the macro plan at docs/plans/2026-02-26-stepml-research-macro-plan.md for context on the research loop.
- THOROUGHLY explore context before generating ideas:
  (a) Codebase: Read sim/latency/stepml.go, sim/latency/latency.go, sim/simulator.go, hypotheses/h-stepml/shared/ scripts. Understand exactly how BLIS computes E2E latency and what the LatencyModel interface provides.
  (b) Prior rounds: Read all FINDINGS_SUMMARY.md and FINDINGS_ROUND files from prior rounds (hypotheses/h-stepml/round*/). Understand what was tried, what worked, what failed, and why.
  (c) Ground-truth data: Examine the training data structure and characteristics in hypotheses/h-stepml/shared/.
  (d) Web search: Search for recent papers and techniques on LLM inference latency prediction, step-time modeling, simulator calibration, vLLM performance modeling (e.g., Vidur, MIST, Frontier, Revati, Splitwise, DistServe). Look for techniques that address the specific binding constraints in problem.md.
  (e) Adjacent approaches: Search for queueing theory calibration, discrete-event simulation tuning, trace-driven simulation, online calibration methods.
- Generate exactly 3 novel, concrete research ideas that:
  (a) Address the binding constraints listed in problem.md
  (b) Build on successful techniques from prior rounds
  (c) Do NOT repeat any eliminated approaches listed in problem.md
  (d) Each idea must be distinct and attack a different aspect of the problem
  (e) Are grounded in the codebase exploration and literature findings
- For each idea, write: title, rationale (citing specific literature or codebase findings), method sketch, expected outcome, why it differs from prior attempts.
- Write all ideas to hypotheses/h-stepml/round${round}/research.md

DO NOT use /research-ideas or any interactive skills. DO NOT use AskUserQuestion. Generate ideas directly.

STEP 2 — WP2 (Scaffolding):
- Take the 3 ideas from the research.md you just wrote.
- Create directories under hypotheses/h-stepml/round${round}/idea-<N>-<short-name>/ for each idea.
- Write HYPOTHESIS.md in each idea directory with sub-hypotheses that MUST include:
  - Core accuracy sub-hypotheses (2-3)
  - A LOMO generalization sub-hypothesis (leave-one-model-out, 4-fold, target <80% MAPE per fold)
  - A LOWO generalization sub-hypothesis (leave-one-workload-out, 3-fold, target <50% MAPE per fold)
  Use hypotheses/h-stepml/shared/splits.py for LOMO/LOWO data splitting functions.
- Each HYPOTHESIS.md must reference prior round findings if round > 1.

When done, create the marker file hypotheses/h-stepml/round${round}/.phase1_done to signal completion.

Do NOT proceed to WP3 (experimentation). Do NOT use AskUserQuestion. Do NOT ask for confirmation. Just do WP1 and WP2." \
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

        # research.md is now written directly to round<N>/research.md by Phase 1.
        # Copy to h-stepml/research.md for worktree seeding, then clean up after use.
        if [[ -f "${round_dir}/research.md" ]]; then
            cp "${round_dir}/research.md" "${STEPML_DIR}/research.md"
            log "  Copied research.md → ${STEPML_DIR}/research.md"
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

    # --- PARALLEL WP3: Launch all ideas concurrently, then collect ---
    # Each idea runs in its own worktree + claude session. No shared mutable state.

    # Phase 2a: Launch all ideas in parallel
    # Use parallel indexed arrays (bash 3.2 compatible — no associative arrays)
    wp3_names=()       # [i] -> idea_name
    wp3_pids=()        # [i] -> background PID
    wp3_worktrees=()   # [i] -> worktree path
    wp3_idea_dirs=()   # [i] -> idea dir in main repo
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

        log "  Phase 2.${idea_num}: WP3 for ${idea_name} [PARALLEL]"

        # Create isolated worktree for this idea
        create_idea_worktree "$round" "$idea_name"
        wt_dir="$WORKTREE_DIR"  # capture before next iteration overwrites global
        idx=${#wp3_names[@]}
        wp3_names+=("$idea_name")
        wp3_worktrees+=("$wt_dir")
        wp3_idea_dirs+=("$idea_dir")

        phase_log="${round_dir}/phase2_${idea_name}.log"

        # Launch claude -p in background subshell
        log "    Launching in worktree: ${wt_dir}"
        (
            cd "$wt_dir"
            log "  [round${round}-${idea_name}] Invoking claude -p ..."
            claude -p "Execute WP3 (experimentation) for a SINGLE idea in Round ${round} of the StepML research loop.

Follow the macro plan at docs/plans/2026-02-26-stepml-research-macro-plan.md strictly.

YOUR TASK: Run all sub-hypotheses for the idea at: hypotheses/h-stepml/round${round}/${idea_name}/

STEPS:
1. Read the HYPOTHESIS.md in that directory to understand the idea and its sub-hypotheses.
2. For each sub-hypothesis, implement the experiment (write code if needed), run it, analyze results, and write FINDINGS.md in the sub-hypothesis directory.
3. Use the shared infrastructure at hypotheses/h-stepml/shared/ for data loading, evaluation, and BLIS validation.
4. After ALL sub-hypotheses are complete, write FINDINGS_SUMMARY.md in the idea's root directory (hypotheses/h-stepml/round${round}/${idea_name}/FINDINGS_SUMMARY.md).

The FINDINGS_SUMMARY.md must contain ALL of these sections:
1. Idea recap
2. Sub-hypothesis results table (status, key metric, takeaway)
3. Best BLIS E2E result — full per-experiment error table
4. What worked (specific techniques)
5. What failed and why (root causes)
6. Binding constraints
7. Data insights discovered
8. Comparison to baseline
9. Go integration feasibility
10. Generalization Results (MANDATORY — the orchestrator validates this section exists):
    - LOMO table: Leave-one-model-out cross-validation results (4-fold, target <80% MAPE per fold)
    - LOWO table: Leave-one-workload-out cross-validation results (3-fold, target <50% MAPE per fold)
    - If infrastructure-only idea, write GENERALIZATION_NOTE.md in the idea directory instead
    - Use hypotheses/h-stepml/shared/splits.py for data splitting

CRITICAL: The orchestrator will REJECT a FINDINGS_SUMMARY.md that is missing section 10. If you skip generalization, the orchestrator will re-invoke you to add it.

You are running in an ISOLATED GIT WORKTREE. You may freely modify Go code (e.g. sim/latency/stepml.go) — changes are isolated to this worktree and will not affect other ideas. Any Go changes will be saved as a patch file for reproducibility.

Do NOT proceed to WP4 or any other idea. Do NOT ask for confirmation. Just run this one idea's experiments and write the summary." \
                --dangerously-skip-permissions \
                --output-format stream-json \
                --verbose \
                --max-turns "$MAX_TURNS_PER_PHASE" \
                < /dev/null > "$phase_log" 2>&1 \
            || log "  [round${round}-${idea_name}] Session exited with non-zero code"
        ) &
        wp3_pids+=($!)
        log "    PID ${wp3_pids[$idx]} launched for ${idea_name}"
    done <<< "$idea_dirs"

    # Phase 2b: Wait for all parallel sessions to finish
    if (( ${#wp3_names[@]} > 0 )); then
        log "  Waiting for ${#wp3_names[@]} parallel WP3 sessions: ${wp3_names[*]}"
        for i in $(seq 0 $(( ${#wp3_names[@]} - 1 ))); do
            log "    Waiting for ${wp3_names[$i]} (PID ${wp3_pids[$i]})..."
            wait "${wp3_pids[$i]}" 2>/dev/null || true
            log "    ${wp3_names[$i]} (PID ${wp3_pids[$i]}) finished"
        done
        log "  All parallel WP3 sessions complete"
    fi

    # Phase 2c: Collect artifacts, validate, retry failures, write REPRODUCE.md
    for i in $(seq 0 $(( ${#wp3_names[@]} - 1 ))); do
        idea_name="${wp3_names[$i]}"
        idea_dir="${wp3_idea_dirs[$i]}"
        WORKTREE_DIR="${wp3_worktrees[$i]}"

        # Collect artifacts back to main repo
        collect_idea_artifacts "$round" "$idea_name" "$idea_dir"

        # Validate: FINDINGS_SUMMARY.md exists + contains LOMO/LOWO
        validate_idea_artifacts "$idea_dir" "$idea_name"
        val_result=$?

        if (( val_result != 0 )); then
            # Retry once with a targeted prompt (worktree is still alive)
            retry_idea_in_worktree "$round" "$idea_name" "$idea_dir" "$val_result"

            # Re-collect artifacts after retry
            collect_idea_artifacts "$round" "$idea_name" "$idea_dir"

            # Final validation (warn but don't block — WP4 will flag incomplete ideas)
            validate_idea_artifacts "$idea_dir" "$idea_name"
            if (( $? != 0 )); then
                log "  WARNING: ${idea_name} still incomplete after retry. WP4 will flag it."
            fi
        fi

        # Write per-idea reproduction guide (after retry, so it captures final state)
        write_idea_reproduce "$round" "$idea_name" "$idea_dir"

        # Clean up worktree
        cleanup_idea_worktree
    done

    # Clean up root-level research.md now that all worktrees have been seeded
    if [[ -f "${STEPML_DIR}/research.md" ]]; then
        rm -f "${STEPML_DIR}/research.md"
        log "  Cleaned up ${STEPML_DIR}/research.md (canonical copy in ${round_dir}/research.md)"
    fi

    # ============================================================
    # PHASE FINAL-A: WP4 Core (Leaderboard + Decision + Wrap-up)
    # ============================================================
    # Input:  All FINDINGS_SUMMARY.md files from this round
    # Output: FINDINGS_ROUND<N>.md, updated problem.md, STATUS file
    #
    # This phase produces ALL critical artifacts. It does NOT run
    # convergence-review (which dispatches 10 agents and can blow
    # the context window). Convergence review runs in Phase FINAL-B.
    # ============================================================

    log "  Phase final-a: WP4 Core (Leaderboard + Decision + Wrap-up)"

    # Build list of available summaries for the prompt
    summaries=""
    while IFS= read -r idea_dir; do
        idea_name=$(basename "$idea_dir")
        sf="${idea_dir}/FINDINGS_SUMMARY.md"
        if [[ -f "$sf" ]]; then
            summaries="${summaries}\n  - hypotheses/h-stepml/round${round}/${idea_name}/FINDINGS_SUMMARY.md"
        fi
    done <<< "$idea_dirs"

    wp4a_marker="${round_dir}/.wp4a_done"
    if [[ -f "$wp4a_marker" ]]; then
        log "  Phase final-a already complete, skipping"
    elif $DRY_RUN; then
        run_claude "round${round}-wp4a" "${round_dir}/phase_final_wp4a.log" \
"[DRY-RUN] Would execute WP4 core for round ${round}" \
        || true
        dry_run_wp4 "$round_dir" "$round"
    else
    run_claude "round${round}-wp4a" "${round_dir}/phase_final_wp4a.log" \
"Execute WP4 Core (Leaderboard, Decision, and Round Wrap-up) for Round ${round} of the StepML research loop.

Follow the macro plan at docs/plans/2026-02-26-stepml-research-macro-plan.md strictly.

AVAILABLE FINDINGS SUMMARIES (read all of these):
$(echo -e "$summaries")

Also read: hypotheses/h-stepml/problem.md (current problem statement with prior round context)

YOUR TASKS (in order):
0. GENERALIZATION GATE (pre-leaderboard): Before ranking, verify that EVERY idea has either (a) LOMO + LOWO results in FINDINGS_SUMMARY.md section 10, or (b) a GENERALIZATION_NOTE.md file. Flag any idea missing both as 'INCOMPLETE — no generalization' in the leaderboard. An incomplete idea CANNOT be the winning idea.
1. LEADERBOARD: Read all FINDINGS_SUMMARY.md files. Rank ideas by BLIS E2E mean error. Compare against blackbox baseline and prior rounds. Include LOMO and LOWO columns in the leaderboard table.
2. FINDINGS_ROUND${round}.md: Write comprehensive round findings to hypotheses/h-stepml/round${round}/FINDINGS_ROUND${round}.md with all 12 sections per the macro plan. Include a dedicated 'Generalization Results' section summarizing LOMO/LOWO across all ideas.
3. LOOP DECISION: Based on the results, decide CONVERGED (<10% E2E target met AND LOMO <80% AND LOWO <50%), ITERATE (progress but not converged), or ABORT (stagnant for 2+ rounds). Note: an idea cannot be declared CONVERGED if it fails the generalization targets.
4. UPDATE problem.md (if ITERATE): Update hypotheses/h-stepml/problem.md with ALL accumulated knowledge per the macro plan's Step 5 specification (baseline results, solved/unsolved experiments, successful techniques, data characteristics, eliminated approaches, binding constraints, key questions, prescribed focus areas, cumulative round history). Include LOMO/LOWO results in the cumulative round history table.
5. WRITE STATUS FILE: Write hypotheses/h-stepml/round${round}/STATUS containing exactly one word: CONVERGED, ITERATE, or ABORT.
6. WRITE MARKER FILE: Write hypotheses/h-stepml/round${round}/.wp4a_done containing 'done'.

IMPORTANT: Do NOT run /convergence-review in this session. Convergence review runs in a separate session to avoid context overflow.
Do NOT ask for confirmation. Execute all steps. STATUS file is the critical deliverable." \
    || log "  WP4 core phase had non-zero exit, checking STATUS..."

    # Create marker if STATUS was written (session may have crashed after STATUS but before marker)
    if [[ -f "$status_file" ]] && [[ ! -f "$wp4a_marker" ]]; then
        echo "done" > "$wp4a_marker"
        log "  WP4 core: STATUS exists, created marker"
    fi
    fi

    # ============================================================
    # PHASE FINAL-B: Convergence Review (optional, best-effort)
    # ============================================================
    # Input:  FINDINGS_ROUND<N>.md
    # Output: Review feedback (logged). May update STATUS in rare cases.
    #
    # This phase is non-critical. If it fails (context overflow,
    # timeout, etc.), the loop continues with the STATUS from
    # Phase FINAL-A.
    # ============================================================

    findings_round_file="${round_dir}/FINDINGS_ROUND${round}.md"
    if [[ -f "$findings_round_file" ]] && [[ -f "$status_file" ]]; then
        log "  Phase final-b: Convergence Review (best-effort)"

        if $DRY_RUN; then
            log "  [DRY-RUN] Would run convergence review for round ${round}"
        else
        run_claude "round${round}-wp4b" "${round_dir}/phase_final_wp4b.log" \
"Run a LIGHTWEIGHT convergence review for Round ${round} of the StepML research loop.

Read hypotheses/h-stepml/round${round}/FINDINGS_ROUND${round}.md and review it for:
1. Are the experimental results internally consistent?
2. Are the binding constraints correctly identified?
3. Is the ITERATE/CONVERGE/ABORT decision well-justified?
4. Are there any critical gaps in the findings that would mislead the next round?

Write a brief review (max 50 lines) to hypotheses/h-stepml/round${round}/CONVERGENCE_REVIEW.md with:
- Overall assessment (1-2 sentences)
- Any CRITICAL issues found (list, or 'None')
- Any IMPORTANT suggestions for the next round (max 3)
- Verdict: APPROVE or REVISE

If verdict is REVISE with critical issues, update the STATUS file at hypotheses/h-stepml/round${round}/STATUS if the decision should change (e.g., ITERATE→ABORT). Otherwise, leave STATUS unchanged.

IMPORTANT: Do NOT use /convergence-review or dispatch parallel review agents. Do this review yourself in a single pass to stay within context limits. Keep the review concise and focused.
Do NOT ask for confirmation." \
        || log "  Convergence review phase had non-zero exit (non-critical, continuing)"
        fi
    else
        log "  Phase final-b: SKIPPED (missing FINDINGS_ROUND${round}.md or STATUS file)"
    fi

    # --- Read STATUS ---
    if [[ ! -f "$status_file" ]]; then
        log "ERROR: No STATUS file after WP4 for round ${round}"
        log "Check: ${round_dir}/phase_final_wp4a.log"
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
