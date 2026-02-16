#!/usr/bin/env python3
"""Build JSON API request for an LLM review."""
import json
import sys

if len(sys.argv) != 3:
    print("Usage: build_request.py <plan_file> <model>", file=sys.stderr)
    sys.exit(1)

plan_file = sys.argv[1]
model = sys.argv[2]

try:
    with open(plan_file, "r") as f:
        plan_content = f.read()
except Exception as e:
    print(f"ERROR: Could not read plan file: {e}", file=sys.stderr)
    sys.exit(1)

# Build request
request = {
    "model": model,
    "messages": [
        {
            "role": "system",
            "content": "You are a technical reviewer analyzing implementation plans. Provide structured, constructive feedback."
        },
        {
            "role": "user",
            "content": f"""Review this implementation plan and provide:

1. OVERALL ASSESSMENT
   - Is the plan comprehensive and actionable?
   - Does it address the core requirements?
   - Rate: Strong / Adequate / Needs Work

2. POTENTIAL ISSUES
   - What gaps, risks, or problems do you see?
   - What assumptions might be incorrect?
   - What edge cases are missing?

3. SUGGESTIONS FOR IMPROVEMENT
   - What would strengthen this plan?
   - What should be clarified or expanded?
   - What alternatives should be considered?

4. IMPLEMENTATION RISKS
   - What areas need careful monitoring?
   - What could go wrong during execution?
   - What dependencies are fragile?

PLAN:
{plan_content}

Provide specific, actionable feedback with concrete examples where possible."""
        }
    ],
    "temperature": 0.3,
    "max_tokens": 2000
}

# Output JSON
print(json.dumps(request))
