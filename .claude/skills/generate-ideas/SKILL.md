---
name: generate-ideas
description: You are a researcher generating ideas to be developed and published in top conferences. Your ideas are creative and novel as well as practical.
argument-hint: <problem_dir> [num_iterations]
allowed-tools:
  - Skill(review-plan *)
  - Bash(.claude/skills/review-plan/scripts/review.sh *)
  - Bash(python3 *)
  - Bash(curl *)
  - Bash(jq *)
---
# ARGUMENTS

- `[PROBLEM_DIR]` (required): Directory containing `problem.md` and `background/`
- `[NUM_ITERATIONS]` (optional, default: 1): Number of ideas to generate

# TASK

Propose solutions to `[PROBLEM_DIR]/problem.md`

# BACKGROUND INFORMATION

Background inforamtion is found in `[PROBLEM_DIR]/background`.

# STEPS:

**CRITICAL CONSTRAINT: Each iteration MUST fully complete (including all reviews and feedback) before starting the next iteration. This ensures each new idea can build on feedback from all previous ideas.**

For i = 1 to [NUM_ITERATIONS], execute the following steps **strictly sequentially**:

  ## Step 1: Generate Idea i

  Generate an idea to solve the problem `[PROBLEM_DIR]/problem.md` using:
  - All available background information in `[PROBLEM_DIR]/background/`
  - ALL previously generated ideas AND their reviewer feedback (from `[PROBLEM_DIR]/idea-1.md` through `[PROBLEM_DIR]/idea-<i-1>.md`)

  The new idea should address weaknesses identified in previous ideas' reviews.

  ## Step 2: Write Idea i

  Write the idea to `[PROBLEM_DIR]/idea-<i>.md`

  ## Step 3: Get Reviews (parallel within this step only)

  **Create the review file using this exact procedure:**
  1. Use the Read tool to read `[PROBLEM_DIR]/problem.md` - copy the ENTIRE output
  2. Use the Read tool to read ALL files in `[PROBLEM_DIR]/background/*` - copy the ENTIRE output of each
  3. Use the Read tool to read `[PROBLEM_DIR]/idea-<i>.md` - copy the ENTIRE output
  4. Use the Write tool to create a temp file (e.g., `[PROBLEM_DIR]/tmp-review-<i>.md`) containing ALL content from steps 1-3, separated by headers like:
     - `# PROBLEM (verbatim from problem.md)`
     - `# BACKGROUND: <filename> (verbatim)`
     - `# IDEA <i> (verbatim from idea-<i>.md)`

  ---
  **BLOCKING REQUIREMENT - YOU MUST NOT TAKE SHORTCUTS:**

  - You MUST copy the Read tool output EXACTLY as received - every single line
  - You MUST NOT write placeholders like `[Same as above]`, `[See previous]`, or `[Same background as previous reviews]`
  - You MUST NOT paraphrase, summarize, or abbreviate ANY content
  - EVERY LINE from EVERY source file MUST appear in the temp file
  - If you find yourself typing `[` followed by a summary description, STOP IMMEDIATELY - you are violating this requirement
  - This applies to ALL iterations, not just the first one - do NOT skip content because "reviewers already saw it"

  **WHY THIS MATTERS:** Each reviewer agent runs independently with NO memory of previous reviews. They can ONLY see what's in the temp file you create. If you summarize the background, reviewers cannot do their job properly.

  ---

  **Step 3a: VERIFY the temp file (REQUIRED before proceeding):**

  After creating the temp file, you MUST verify it:
  1. Run: `wc -l [PROBLEM_DIR]/tmp-review-<i>.md`
  2. Run: `wc -l [PROBLEM_DIR]/problem.md [PROBLEM_DIR]/background/* [PROBLEM_DIR]/idea-<i>.md`
  3. Compare: The temp file MUST have AT LEAST 90% as many lines as the sum of all source files
  4. If the temp file is too small, DELETE it and REDO steps 1-4 with FULL content

  **FAILURE CHECK:** If line count of tmp-review file is less than 90% of source files combined, you have summarized. Delete and redo.

  ---

  Ask 3 judges to review idea-<i> **in parallel** using the /review-plan skill, passing it the temp file path containing the problem statement, backgroud and the idea.
  Each judge should use a different model for each idea evaluation. You can select from aws/claude-opus-4-6, Azure/gpt-4o, GCP/gemini-2.5-flash. 
  
  YOU MUST USE THE /review-plan SKILL AND INVOKE THE TWO NON-ANTHROPIC MODELS. The judges must provide independent review feedback. They must not share context with each other.

  ## Step 4: Append Feedback

  **WAIT for ALL reviewers to complete.** Then append each reviewer's feedback to `[PROBLEM_DIR]/idea-<i>.md`.

  ## Step 5: Verify Completion (BLOCKING)

  **DO NOT proceed to iteration i+1 until:**
  - `[PROBLEM_DIR]/idea-<i>.md` contains the full idea AND all reviewer feedback
  - You have read and understood the feedback to inform the next idea

  Only after confirming step 5 is complete, proceed to iteration i+1.

---

# FINAL STEP: Executive Summary

After ALL [NUM_ITERATIONS] iterations are complete, create `[PROBLEM_DIR]/summary.md` containing:

1. **Problem Statement**: Brief recap of the problem being solved
2. **Ideas Overview**: One-paragraph summary of each idea generated
3. **Comparison Table**: Side-by-side comparison of all ideas across key dimensions (parameters, complexity, feasibility, etc.)
4. **Reviewer Consensus**: Key themes and agreements across all reviewer feedback
5. **Recommendation**: Which idea (or combination) is recommended and why
6. **Next Steps**: Concrete actions to pursue the recommended approach

The summary should help a reader quickly understand all ideas and make an informed decision without reading every idea document in full.
