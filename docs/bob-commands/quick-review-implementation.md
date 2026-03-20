# Quick-Review Implementation Guide

**Status:** Active (v1.0 — created 2026-03-19)

This document provides implementation details for the `/quick-review` Bob command, including perspective mappings, execution algorithms, and usage examples.

---

## Overview

The `/quick-review` command provides multi-perspective code reviews using the same 43 perspective prompts from the convergence-review skill, but with sequential execution (no parallel agents) and no convergence loop. This bridges the gap between the general `/review` command and the formal `/convergence-review` skill.

**Key differences from other review commands:**

| Feature | `/review` | `/quick-review` | `/convergence-review` |
|---------|-----------|----------------|----------------------|
| Tool | Bob command | Bob command | Claude Code skill |
| Perspectives | 4 general | 5-10 domain-specific | 5-10 domain-specific |
| Execution | Single session | Single session | Parallel agents |
| Loop | No | No | Yes (Phase A/B) |
| Output | Bob Findings | Bob Findings | Conversation + state file |
| Token cost | Low | Medium | High |
| Defaults | No | Yes | No |
| Aliases | No | Yes | No |
| Fuzzy matching | No | Yes | No |
| Use case | General review | Domain spot-check | Formal gate |

---

## Bob Command Format

The command is defined in `.bob/commands/quick-review.md` with two main sections:

### Frontmatter (YAML)

```yaml
---
description: Execute multi-perspective review using convergence-review perspectives (sequential, no loop)
argument-hint: <gate-type> [artifact-path]
---
```

**Fields:**
- `description`: Brief description shown in Bob's command menu
- `argument-hint`: Shows expected arguments (displayed in gray in command menu)

### Command Prompt

The command prompt contains instructions for Bob to execute the workflow:
1. Parse arguments (`$1` for gate-type, `$2` for artifact-path)
2. Normalize gate type (aliases + fuzzy matching)
3. Load artifact (with default search if path not provided)
4. Read perspective prompts from `.bob/prompts/review-perspectives.md`
5. Execute each perspective sequentially
6. Extract and accumulate findings
7. Report via `submit_review_findings`
8. Exit (no Phase B, no state file)

---

## Gate Types

| Gate | Perspectives | Artifact Type | Aliases | Default Artifact |
|------|-------------|---------------|---------|------------------|
| `pr-plan` | 10 | Micro plan file | `plan` | `docs/plans/pr*-plan.md` matching branch |
| `pr-code` | 10 | Git diff | `code`, `implementation` | `git diff HEAD` |
| `design` | 8 | Design doc | - | `docs/plans/*-design.md` |
| `macro-plan` | 8 | Macro plan | - | `docs/plans/*-plan.md` (non-PR) |
| `h-design` | 5 | Conversation context | - | Conversation context |
| `h-code` | 5 | Experiment scripts | - | Current directory if in `hypotheses/h-*/` (runs inside experiment branch) |
| `h-findings` | 10 | FINDINGS.md | `findings`, `discovery` | `FINDINGS.md` in current/hypothesis dir (runs inside experiment branch) |

---

## Perspective Sources

### PR Plan Review (10 perspectives)

**Source:** `.bob/prompts/review-perspectives.md` (Section: PR Plan Review Perspectives)

| ID | Perspective | Focus |
|----|------------|-------|
| PP-1 | Substance & Design | Logic bugs, mathematical errors, scale mismatches |
| PP-2 | Cross-Document Consistency | Scope match, stale references, deviation log |
| PP-3 | Architecture Boundary Verification | Import cycles, boundary violations, construction sites |
| PP-4 | Codebase Readiness | Stale comments, pre-existing bugs, dependencies |
| PP-5 | Structural Validation | Task dependencies, template completeness, clarity |
| PP-6 | DES Expert | Event ordering, clock monotonicity, heap priority |
| PP-7 | vLLM/SGLang Expert | Batching semantics, KV cache, chunked prefill |
| PP-8 | Distributed Platform Expert | Multi-instance coordination, routing, admission |
| PP-9 | Performance & Scalability | Algorithmic complexity, allocations, memory growth |
| PP-10 | Security & Robustness | Input validation, panic paths, resource exhaustion |

### PR Code Review (10 perspectives)

**Source:** `.bob/prompts/review-perspectives.md` (Section: PR Code Review Perspectives)

| ID | Perspective | Focus |
|----|------------|-------|
| PC-1 | Substance & Design | Logic bugs, design mismatches, regressions |
| PC-2 | Code Quality + Antipattern Check | Error handling, map iteration, construction sites |
| PC-3 | Test Behavioral Quality | Test coverage, edge cases, behavioral contracts |
| PC-4 | Cross-Document Consistency | Plan-code alignment, stale references |
| PC-5 | Architecture Boundary Verification | Import cycles, boundary violations |
| PC-6 | DES Expert | Event ordering, clock monotonicity |
| PC-7 | vLLM/SGLang Expert | Batching, KV cache, preemption |
| PC-8 | Distributed Platform Expert | Multi-instance coordination, routing |
| PC-9 | Performance & Scalability | Algorithmic complexity, hot paths |
| PC-10 | Security & Robustness | Input validation, panic paths |

### Design Review (8 perspectives)

**Source:** `.bob/prompts/review-perspectives.md` (Section: Design Review Perspectives)

| ID | Perspective | Focus |
|----|------------|-------|
| DD-1 | Substance & Design | Design soundness, mathematical correctness |
| DD-2 | Cross-Document Consistency | Scope match, stale references |
| DD-3 | Architecture Boundary Verification | Boundary violations, abstraction levels |
| DD-4 | Codebase Readiness | Pre-existing issues, dependencies |
| DD-5 | DES Expert | Event ordering, simulation semantics |
| DD-6 | vLLM/SGLang Expert | Serving semantics, batching |
| DD-7 | Distributed Platform Expert | Multi-instance coordination |
| DD-8 | Performance & Scalability | Algorithmic complexity, scalability |

### Macro Plan Review (8 perspectives)

**Source:** `.bob/prompts/review-perspectives.md` (Section: Macro Plan Review Perspectives)

| ID | Perspective | Focus |
|----|------------|-------|
| MP-1 | Substance & Design | Plan soundness, logical consistency |
| MP-2 | Cross-Document Consistency | Scope match, references |
| MP-3 | Architecture Boundary Verification | Boundary violations |
| MP-4 | Codebase Readiness | Dependencies, insertion points |
| MP-5 | DES Expert | Event ordering implications |
| MP-6 | vLLM/SGLang Expert | Serving semantics implications |
| MP-7 | Distributed Platform Expert | Multi-instance implications |
| MP-8 | Performance & Scalability | Performance implications |

### Hypothesis Design Review (5 perspectives)

**Source:** `.bob/prompts/review-perspectives.md` (Section: Hypothesis Design Review Perspectives)

| ID | Perspective | Focus |
|----|------------|-------|
| DR-1 | Hypothesis Clarity | Clear, testable hypothesis |
| DR-2 | Experimental Design | Control experiments, variables |
| DR-3 | Measurement Strategy | Metrics, data collection |
| DR-4 | Validity Threats | Confounding factors, biases |
| DR-5 | Reproducibility | Replication, documentation |

### Hypothesis Code Review (5 perspectives)

**Source:** `.bob/prompts/review-perspectives.md` (Section: Hypothesis Code Review Perspectives)

| ID | Perspective | Focus |
|----|------------|-------|
| CR-1 | Script Correctness | Logic bugs, edge cases |
| CR-2 | Data Collection | Metrics, logging |
| CR-3 | Analysis Quality | Statistical methods, visualization |
| CR-4 | Reproducibility | Seeds, configuration |
| CR-5 | Documentation | Comments, README |

### Hypothesis Findings Review (10 perspectives)

**Source:** `.bob/prompts/review-perspectives.md` (Section: Hypothesis Findings Review Perspectives)

| ID | Perspective | Focus |
|----|------------|-------|
| FR-1 | Claim-Evidence Alignment | Claims supported by data |
| FR-2 | Statistical Rigor | Proper statistical methods |
| FR-3 | Visualization Quality | Clear, accurate charts |
| FR-4 | Alternative Explanations | Confounding factors addressed |
| FR-5 | Generalizability | Scope of conclusions |
| FR-6 | Reproducibility | Replication instructions |
| FR-7 | Documentation Quality | Clear, complete |
| FR-8 | Cross-Reference Consistency | Links to code, data |
| FR-9 | Actionability | Clear next steps |
| FR-10 | Presentation Quality | Professional, clear |

---

## Sequential Execution Algorithm

### Pseudocode

```
function quick_review(gate_type, artifact_path):
    # 1. Normalize gate type
    normalized_gate = normalize_gate_type(gate_type)
    if normalized_gate is None:
        show_error_and_exit()
    
    # 2. Load artifact
    artifact = load_artifact(normalized_gate, artifact_path)
    if artifact is None:
        show_error_and_exit()
    
    # 3. Read perspective prompts
    prompts = read_perspective_prompts(normalized_gate)
    
    # 4. Execute perspectives sequentially
    findings = []
    for prompt in prompts:
        # Apply prompt to artifact
        analysis = apply_perspective(prompt, artifact)
        
        # Extract findings
        perspective_findings = extract_findings(analysis)
        
        # Accumulate
        findings.extend(perspective_findings)
    
    # 5. Report findings
    submit_review_findings(findings)
    
    # 6. Exit
    return

function normalize_gate_type(input):
    # Handle empty/null
    if input is empty or null:
        return None
    
    # Check exact aliases (case-insensitive)
    aliases = {
        "plan": "pr-plan",
        "code": "pr-code",
        "implementation": "pr-code",
        "findings": "h-findings",
        "discovery": "h-findings"
    }
    if input.lower() in aliases:
        return aliases[input.lower()]
    
    # Check canonical names
    canonical = ["pr-plan", "pr-code", "design", "macro-plan", 
                 "h-design", "h-code", "h-findings"]
    if input in canonical:
        return input
    
    # Fuzzy matching using Levenshtein distance
    best_match, similarity = find_closest_match(input, canonical)
    if similarity > 70:
        suggest_match(best_match)
        return None  # Wait for user confirmation
    else:
        return None  # Show error

function load_artifact(gate, path):
    if gate == "pr-code":
        return execute_command("git diff HEAD")
    
    if path is provided:
        return read_file(path)
    
    # Default artifact search
    if gate == "pr-plan":
        branch = get_current_branch()
        matches = find_files("docs/plans/pr*-plan.md", contains=branch)
    elif gate == "design":
        matches = find_files("docs/plans/*-design.md")
    elif gate == "macro-plan":
        matches = find_files("docs/plans/*-plan.md", exclude="pr*")
    elif gate == "h-code":
        if current_dir contains "hypotheses/h-":
            return current_dir
    elif gate == "h-findings":
        matches = find_files("FINDINGS.md")
    elif gate == "h-design":
        return conversation_context
    
    if len(matches) == 0:
        return None
    elif len(matches) == 1:
        return read_file(matches[0])
    else:
        # Multiple matches: use most recently modified
        return read_file(most_recent(matches))

function read_perspective_prompts(gate):
    # Map gate to source file and section
    source_map = {
        "pr-plan": ("pr-prompts.md", "Section A"),
        "pr-code": ("pr-prompts.md", "Section B"),
        "design": ("design-prompts.md", "Section A"),
        "macro-plan": ("design-prompts.md", "Section B"),
        "h-design": ("../hypothesis-experiment/review-prompts.md", "Section A"),
        "h-code": ("../hypothesis-experiment/review-prompts.md", "Section B"),
        "h-findings": ("../hypothesis-experiment/review-prompts.md", "Section C")
    }
    
    file, section = source_map[gate]
    content = read_file(".bob/prompts/review-perspectives.md")
    
    # Extract prompts from markdown code blocks
    prompts = []
    in_code_block = False
    current_prompt = []
    
    for line in content.lines:
        if line.startswith("```"):
            if in_code_block:
                # End of code block
                prompts.append("\n".join(current_prompt))
                current_prompt = []
                in_code_block = False
            else:
                # Start of code block
                in_code_block = True
        elif in_code_block:
            current_prompt.append(line)
    
    return prompts
```

---

## Finding Extraction

Each finding must include these required fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `category` | string | One of: maintainability, security, performance, functionality, style | `"maintainability"` |
| `type` | string | Specific issue type | `"magic-numbers-strings"` |
| `severity` | string | One of: critical, high, medium, low | `"high"` |
| `title` | string | Brief title | `"Magic number should be constant"` |
| `message` | string | Detailed explanation | `"The magic number 42 should be extracted to a named constant"` |
| `path` | string | File path relative to workspace | `"docs/plans/pr760-quick-review-plan.md"` |
| `line` | number | Line number (1-based) | `179` |
| `issueScope` | string | "Single File" or "Multiple Files" | `"Single File"` |

Optional fields:

| Field | Type | Description |
|-------|------|-------------|
| `column` | number | Column number |
| `endLine` | number | End line number |
| `endColumn` | number | End column number |
| `suggestion` | string | Fix recommendation |

---

## Usage Examples

### Basic Usage (with defaults)

```bash
# Review current diff with pr-code perspectives (10 perspectives)
/quick-review

# Review plan with auto-detected file (10 perspectives)
/quick-review plan

# Review using alias (10 perspectives)
/quick-review implementation

# Review findings with auto-detection (10 perspectives)
/quick-review findings
```

### Explicit Usage

```bash
# Review specific plan file (10 perspectives)
/quick-review pr-plan docs/plans/pr760-quick-review-plan.md

# Review design doc (8 perspectives)
/quick-review design docs/plans/roofline-design.md

# Review macro plan (8 perspectives)
/quick-review macro-plan docs/plans/2026-03-15-convergence-review-redesign-plan.md

# Review hypothesis findings (10 perspectives) — run inside experiment branch
/quick-review h-findings hypotheses/h-<name>/FINDINGS.md

# Review hypothesis code (5 perspectives) — run inside experiment branch
/quick-review h-code hypotheses/h-<name>/
```

### Fuzzy Matching Examples

```bash
# High similarity (>70%) - suggests match
/quick-review discovry
# Output: Did you mean 'h-findings'? Valid gates: pr-plan, pr-code, design, macro-plan, h-design, h-code, h-findings. Aliases: plan, code, implementation, findings, discovery

# High similarity (>70%) - suggests match
/quick-review macro
# Output: Did you mean 'macro-plan'? Valid gates: pr-plan, pr-code, design, macro-plan, h-design, h-code, h-findings. Aliases: plan, code, implementation, findings, discovery

# Low similarity (≤70%) - shows error
/quick-review xyz
# Output: Invalid gate type 'xyz'. Valid gates: pr-plan, pr-code, design, macro-plan, h-design, h-code, h-findings. Aliases: plan, code, implementation, findings, discovery
```

### Error Cases

```bash
# Empty diff
/quick-review pr-code
# Output: No changes detected since last commit. Nothing to review.
#         Suggestion: Stage new files or commit changes before invoking.

# Missing file
/quick-review pr-plan nonexistent.md
# Output: File not found: nonexistent.md
#         Searched: docs/plans/
#         Suggestion: Provide explicit path as second argument

# Default search failed
/quick-review design
# Output: No matching files found for pattern: docs/plans/*-design.md
#         Searched in: docs/plans/
#         Suggestion: Provide explicit path as second argument
```

---

## Comparison with Other Review Commands

### When to Use Each Command

**Use `/review`** when:
- You need a quick general review (4 perspectives)
- You want basic bug/security scanning
- You don't need domain-specific expertise
- Token cost is a primary concern

**Use `/quick-review`** when:
- You want domain-specific review quality (5-10 perspectives)
- You're doing iterative development and need spot-checks
- You want convergence-review quality without the convergence loop
- You can tolerate medium token cost for better quality

**Use `/convergence-review`** (Claude Code) when:
- You're at a formal gate (Step 2.5 or 4.5 in PR workflow)
- You need guaranteed convergence (0 CRITICAL + 0 IMPORTANT)
- You want automatic fix-and-rerun loop
- Token cost is not a concern
- You're using Claude Code (not Bob)

---

## Implementation Notes

### Token Cost Analysis

**Parallel dispatch (convergence-review):**
- N perspectives × full context per perspective
- Example: 10 perspectives × 10K tokens = 100K tokens per round

**Sequential execution (quick-review):**
- Shared context + N perspective prompts
- Example: 10K tokens (context) + 10 × 1K tokens (prompts) = 20K tokens
- **Savings: ~80% compared to parallel dispatch**

Note: Actual savings depend on context size and perspective complexity. The shared context reduces per-perspective overhead.

### Fuzzy Matching Implementation

Use Levenshtein distance normalized to 0-100% similarity:

```python
def levenshtein_similarity(s1, s2):
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 100 * (1 - distance / max_len)
```

Threshold of 70% provides good balance between helpfulness and precision.

### Prompt Extraction

Extract prompts from markdown code blocks:

```python
def extract_prompts(file_content):
    prompts = []
    in_block = False
    current = []
    
    for line in file_content.split('\n'):
        if line.strip() == '```':
            if in_block:
                prompts.append('\n'.join(current))
                current = []
            in_block = not in_block
        elif in_block:
            current.append(line)
    
    return prompts
```

---

## Maintenance

### Adding New Perspectives

To add a new perspective to an existing gate:

1. Add the perspective prompt to `.bob/prompts/review-perspectives.md` in the appropriate section
2. Update the perspective count in this document
3. No changes needed to the Bob command file (it reads prompts dynamically)

### Adding New Gate Types

To add a new gate type:

1. Add perspective prompts to appropriate source file
2. Update the Bob command file (`.bob/commands/quick-review.md`):
   - Add to canonical names list
   - Add to gate type table
   - Add artifact loading logic
   - Add to source map
3. Update this implementation guide:
   - Add to gate types table
   - Add to perspective sources section
   - Add usage examples

---

## Related Documentation

- [PR Workflow](../contributing/pr-workflow.md) - Full PR development process
- [Convergence Protocol](../contributing/convergence.md) - Convergence rules and invariants
- Convergence Review Skill - Original source of perspective prompts (now self-contained in `.bob/prompts/review-perspectives.md`)
- [Bob Command Documentation](https://internal.bob.ibm.com/docs/ide/basic-usage/slash-commands) - Bob's command system