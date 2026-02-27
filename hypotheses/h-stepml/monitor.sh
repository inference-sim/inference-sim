#!/usr/bin/env bash
#
# StepML Research Loop Monitor
# Usage: ./monitor.sh [ROUND]   (default: auto-detects latest round)
#

STEPML_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-detect latest round
if [[ -n "$1" ]]; then
    ROUND="$1"
else
    ROUND=$(ls -d "${STEPML_DIR}"/round[0-9]* 2>/dev/null | sort -t'd' -k2 -n | tail -1 | grep -o '[0-9]*$')
    if [[ -z "$ROUND" ]]; then
        echo "No rounds found."
        exit 1
    fi
fi

ROUND_DIR="${STEPML_DIR}/round${ROUND}"
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
RED="\033[31m"
RESET="\033[0m"

clear
echo -e "${BOLD}=== StepML Round ${ROUND} Monitor ===${RESET}"
echo ""

# --- Round status ---
if [[ -f "${ROUND_DIR}/STATUS" ]]; then
    status=$(cat "${ROUND_DIR}/STATUS" | tr -d '[:space:]')
    echo -e "${BOLD}Round Status:${RESET} ${GREEN}${status}${RESET}"
    echo ""
fi

# --- Determine current phase ---
phase1_done=false
all_ideas_done=true
wp4_done=false

[[ -f "${ROUND_DIR}/.phase1_done" ]] && phase1_done=true
[[ -f "${ROUND_DIR}/FINDINGS_ROUND${ROUND}.md" ]] && wp4_done=true

# Check ideas
idea_dirs=$(find "$ROUND_DIR" -maxdepth 1 -type d -name 'idea-*' 2>/dev/null | sort)
idea_count=$(echo "$idea_dirs" | grep -c . 2>/dev/null || echo 0)
ideas_with_summary=0
current_idea=""

if [[ -n "$idea_dirs" ]]; then
    while IFS= read -r d; do
        name=$(basename "$d")
        if [[ -f "${d}/FINDINGS_SUMMARY.md" ]]; then
            ideas_with_summary=$((ideas_with_summary + 1))
        else
            [[ -z "$current_idea" ]] && current_idea="$name"
            all_ideas_done=false
        fi
    done <<< "$idea_dirs"
fi

# Determine active phase
if ! $phase1_done; then
    active_phase="WP1+WP2 (Ideation + Scaffolding)"
    active_log="${ROUND_DIR}/phase1_wp1_wp2.log"
elif ! $all_ideas_done; then
    active_phase="WP3 (Experimentation) — ${current_idea}"
    active_log="${ROUND_DIR}/phase2_${current_idea}.log"
elif ! $wp4_done; then
    active_phase="WP4 (Leaderboard + Convergence Review)"
    active_log="${ROUND_DIR}/phase_final_wp4.log"
else
    active_phase="Complete"
    active_log=""
fi

echo -e "${BOLD}Active Phase:${RESET} ${CYAN}${active_phase}${RESET}"
echo ""

# --- Phase 1 detail ---
echo -e "${BOLD}Phase 1 (WP1+WP2):${RESET}"
if $phase1_done; then
    echo -e "  ${GREEN}Done${RESET} — ${idea_count} ideas scaffolded"
else
    # Check sub-progress from log
    if [[ -f "${ROUND_DIR}/phase1_wp1_wp2.log" ]]; then
        log="${ROUND_DIR}/phase1_wp1_wp2.log"
        log_size=$(wc -c < "$log" | tr -d ' ')

        research_ideas_started=false
        research_ideas_done=false
        research_md_written=false
        hypothesis_files=0

        if grep -q "research-ideas" "$log" 2>/dev/null; then
            research_ideas_started=true
        fi
        # Check if research.md exists AND actually contains ideas (not just problem context)
        for rmd in "${ROUND_DIR}/research.md" "${STEPML_DIR}/research.md"; do
            if [[ -f "$rmd" ]] && grep -qi "idea\|approach\|## [0-9]\|strategy\|proposal" "$rmd" 2>/dev/null; then
                research_md_written=true
                research_ideas_done=true
                break
            fi
        done
        hypothesis_files=$(find "$ROUND_DIR" -name "HYPOTHESIS.md" 2>/dev/null | wc -l | tr -d ' ')

        if (( hypothesis_files > 0 )); then
            echo -e "  ${YELLOW}In progress${RESET} — WP2: ${hypothesis_files} HYPOTHESIS.md written"
        elif $research_ideas_done; then
            echo -e "  ${YELLOW}In progress${RESET} — research.md complete, scaffolding ideas..."
        elif $research_ideas_started; then
            echo -e "  ${YELLOW}In progress${RESET} — /research-ideas running..."
        elif (( log_size > 0 )); then
            echo -e "  ${YELLOW}In progress${RESET} — reading problem.md and planning..."
        else
            echo -e "  ${YELLOW}Waiting${RESET} — session starting..."
        fi
    else
        echo -e "  ${YELLOW}Not started${RESET}"
    fi
fi

# --- Phase 2 (WP3) detail ---
echo ""
echo -e "${BOLD}Phase 2 (WP3 — Experimentation):${RESET}"
if [[ -z "$idea_dirs" ]] || [[ "$idea_dirs" == "" ]]; then
    echo -e "  ${YELLOW}Waiting for Phase 1${RESET}"
else
    while IFS= read -r d; do
        name=$(basename "$d")
        if [[ -f "${d}/FINDINGS_SUMMARY.md" ]]; then
            echo -e "  ${GREEN}Done${RESET}  ${name}"
        elif [[ -f "${ROUND_DIR}/phase2_${name}.log" ]]; then
            # Count sub-hypothesis FINDINGS.md files
            findings_count=$(find "$d" -mindepth 2 -name "FINDINGS.md" 2>/dev/null | wc -l | tr -d ' ')
            sub_dirs=$(find "$d" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
            echo -e "  ${YELLOW}Running${RESET} ${name}  (${findings_count}/${sub_dirs} sub-hypotheses done)"
        else
            echo -e "  ${RED}Pending${RESET} ${name}"
        fi
    done <<< "$idea_dirs"
fi

# --- Phase final (WP4) detail ---
echo ""
echo -e "${BOLD}Phase Final (WP4):${RESET}"
if $wp4_done; then
    echo -e "  ${GREEN}Done${RESET} — FINDINGS_ROUND${ROUND}.md written"
    if [[ -f "${ROUND_DIR}/STATUS" ]]; then
        echo -e "  STATUS: ${GREEN}$(cat "${ROUND_DIR}/STATUS")${RESET}"
    fi
elif $all_ideas_done && [[ -f "${ROUND_DIR}/phase_final_wp4.log" ]]; then
    echo -e "  ${YELLOW}Running${RESET}"
else
    echo -e "  ${RED}Pending${RESET}"
fi

# --- Files created ---
echo ""
echo -e "${BOLD}Artifacts:${RESET}"
find "$ROUND_DIR" -type f ! -name "*.log" ! -name ".phase1_done" 2>/dev/null | sort | while read -r f; do
    relpath="${f#$ROUND_DIR/}"
    size=$(wc -c < "$f" | tr -d ' ')
    echo "  ${relpath} (${size}b)"
done

# --- Recent activity from active log ---
if [[ -n "$active_log" ]] && [[ -f "$active_log" ]]; then
    echo ""
    echo -e "${BOLD}Recent tool calls:${RESET}"
    # Extract recent tool uses from stream-json
    grep '"tool_use"' "$active_log" 2>/dev/null | tail -5 | while read -r line; do
        tool=$(echo "$line" | jq -r '.tool_name // .tool // empty' 2>/dev/null)
        # Try to get a useful identifier from the input
        detail=$(echo "$line" | jq -r '
            .input //  {} |
            (.file_path // .pattern // .command // .skill // .prompt // "" | .[:80])
        ' 2>/dev/null)
        if [[ -n "$tool" ]]; then
            echo -e "  ${CYAN}${tool}${RESET} ${detail}"
        fi
    done
fi
