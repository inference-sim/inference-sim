---
name: implement-issue
description: Create a PR from a GitHub issue. Reads the issue, plans implementation, writes code following project standards (TDD, invariants, parity), creates the PR, then triggers blis-pr-review for self-review.
---

# Implement Issue

Create a pull request that implements a GitHub issue end-to-end.

---

## Step 0 — Identify the Issue

Resolve the issue using the first matching source:

1. **Skill argument**: if a URL or `#number` was passed, parse owner/repo/number.
2. **GitHub Actions context**: if running inside `claude-code-action` on an issue trigger, the issue body is already in conversation context.
3. **Fallback**: ask the user for the issue URL.

Fetch full issue context:

```bash
gh issue view <NUMBER> --repo <OWNER>/<REPO> --json title,body,labels,assignees,milestone
```

If the issue references a parent/tracking issue, fetch that too for full context:

```bash
gh issue view <PARENT_NUMBER> --repo <OWNER>/<REPO> --json title,body
```

---

## Step 1 — Assess Scope and Applicable Standards

Read the issue and determine:

1. **Issue type**: bug fix, feature (policy template, subsystem module, backend swap, tier composition), refactoring, hardening
2. **Affected modules**: which BLIS modules are touched (router, scheduler, KV cache, latency model, workload, batch formation, etc.)
3. **Invariants at risk**: which of INV-1 through INV-13 (plus PD disaggregation INV-PD-* / pool INV-P2-*) could be affected
4. **Cross-path parity**: does this change apply to `run`, `replay`, `observe`, or multiple paths?
5. **Extension type**: if a new feature, which extension recipe applies (see `docs/contributing/extension-recipes.md`)

Read relevant standards:

```bash
cat docs/contributing/standards/invariants.md
cat docs/contributing/standards/rules.md
cat docs/contributing/standards/principles.md
```

---

## Step 2 — Create Feature Branch

```bash
git checkout -b issue-<NUMBER>-<short-description> main
```

Use a descriptive branch name derived from the issue title.

---

## Step 3 — Plan Implementation

Before writing code, produce a brief implementation plan (do NOT write a file — keep it in conversation context):

1. **Behavioral contracts**: what invariants/guarantees must the implementation satisfy
2. **TDD sequence**: list test scenarios to write FIRST (table-driven, behavioral, not structural)
3. **Files to modify/create**: enumerate with brief rationale
4. **Cross-path impact**: for each of `run`/`replay`/`observe`, state whether the change applies and why
5. **Risk assessment**: preemption safety, timeout consistency, metric correctness

---

## Step 4 — Implement with TDD

For each logical unit of work:

1. **Write the test first** — behavioral, table-driven, testing observable outcomes not internal structure
2. **Run the test** — verify it fails for the right reason
3. **Write the implementation** — minimal code to pass the test
4. **Run all tests** — ensure no regressions

```bash
go test ./... -count=1
```

Follow project rules:
- R1: No silent `continue` — handle or propagate errors
- R4: Canonical constructors — struct literals in exactly one place
- R8: No exported mutable maps
- R9: Pointer types for YAML optional fields
- R10: Strict YAML parsing (`KnownFields(true)`)
- R13: Behavioral contracts, not implementation-specific interfaces
- R14: Single-module methods

---

## Step 5 — Verify

Run the full verification suite:

```bash
go build ./...
go test ./... -count=1
golangci-lint run ./...
```

All three must pass before proceeding.

---

## Step 6 — Commit and Push

```bash
git add <specific-files>
git commit -m "<type>(<scope>): <description>

Closes #<NUMBER>

Co-Authored-By: Claude <noreply@anthropic.com>"

git push -u origin issue-<NUMBER>-<short-description>
```

---

## Step 7 — Create the Pull Request

```bash
gh pr create --title "<concise title under 70 chars>" --body "$(cat <<'EOF'
## Summary

<1-3 bullet points describing what changed and why>

## Issue

Closes #<NUMBER>

## Cross-Path Parity

| Path | Applies | Implemented | Notes |
|------|---------|-------------|-------|
| `blis run` | yes/no | yes/no/N/A | ... |
| `blis replay` | yes/no | yes/no/N/A | ... |
| `blis observe` | yes/no | yes/no/N/A | ... |

## Invariants Affected

- INV-X: <brief note on how it's preserved>

## Test Plan

- [ ] Unit tests pass (`go test ./...`)
- [ ] Lint passes (`golangci-lint run ./...`)
- [ ] <specific behavioral tests added>

---

Generated with [Claude Code](https://claude.ai/claude-code)
EOF
)"
```

---

## Step 8 — Self-Review and Fix Loop (Converge to READY TO MERGE)

After the PR is created, YOU (implement-issue) own the fix loop. The review skill only produces findings — it never modifies code.

**Loop (max 3 rounds):**

1. **Invoke the review:**
   ```
   /blis-pr-review
   ```
   This posts a verdict as a PR comment. It does NOT fix anything.

2. **Read the verdict.** If **READY TO MERGE** → exit loop. Done.

3. **If CHANGES REQUIRED — fix the findings yourself:**
   - Fix all CRITICAL findings immediately.
   - Fix IMPORTANT findings if straightforward.
   - Re-run verification:
     ```bash
     go build ./...
     go test ./... -count=1
     golangci-lint run ./...
     ```
   - Commit and push:
     ```bash
     git add <fixed-files>
     git commit -m "fix: address review findings (round N)"
     git push
     ```

4. **Loop back to step 1** for re-review.

5. If max rounds exhausted without convergence, post a comment requesting human guidance.

---

## Responding to Human Review Comments

When this skill is triggered on a PR (via `@claude` in a review comment), instead of creating a new PR:

1. Read the review comments and requested changes
2. Implement the requested fixes
3. Re-run verification (`go build && go test && golangci-lint`)
4. Commit and push fixes
5. Re-invoke `/blis-pr-review` to verify the fix doesn't introduce new issues
6. Reply to each review comment confirming the fix or explaining why an alternative approach was taken

---

## Error Handling

- If `go test` fails: fix the issue, do not skip tests
- If `golangci-lint` fails: fix lint issues, do not suppress with `//nolint`
- If the issue is ambiguous: comment on the issue asking for clarification rather than guessing
- If the scope is too large for a single PR: comment on the issue proposing a decomposition, implement only the first piece
- If self-review loop exceeds 3 iterations: stop, post remaining findings, request human guidance
